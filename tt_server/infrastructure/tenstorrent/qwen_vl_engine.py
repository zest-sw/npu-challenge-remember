import logging
import os

import torch
import ttnn
from transformers import AutoProcessor

from llama_models.llama3.api.datatypes import ImageMedia, UserMessage

from .model_env import select_hf_model
from .vlm_backends import load_vlm_backend

# stdlib logging, not loguru: in this process every loguru record after the ttnn import
# vanished -- the server log holds exactly one python-loguru line (ttnn's own, at import)
# and none of ours, while runtime.py's stdlib records came through fine. Reason unknown;
# this uses the logger that demonstrably reaches the log.
logger = logging.getLogger(__name__)


class QwenVLEngine:
    def __init__(self, mesh_device, model_name: str, max_seq_len: int):
        self.mesh_device = mesh_device
        self.model_name = model_name
        self.max_seq_len = max_seq_len

    def load(self) -> None:
        backend = load_vlm_backend(self.model_name)
        self._encode = backend["hf_multimodal_encode"]
        self._get_batch_sampler = backend["get_batch_sampler"]
        select_hf_model(self.model_name)
        logger.info(f"[VLM] loading {self.model_name} ...")

        num_devices = (
            self.mesh_device.get_num_devices()
            if isinstance(self.mesh_device, ttnn.MeshDevice)
            else 1
        )
        generator_args = {
            "num_devices": num_devices,
            "data_parallel": 1,
            "mesh_device": self.mesh_device,
            "max_batch_size": 1,
            "max_seq_len": self.max_seq_len,
        }
        if backend["DecodersPrecision"] is not None:
            generator_args["optimizations"] = lambda args: backend[
                "DecodersPrecision"
            ].performance(args.n_layers, args.model_name)

        model_args, model = backend["prepare_generator_args"](**generator_args)
        self.model_args = model_args
        self.processor = AutoProcessor.from_pretrained(model_args[0].CKPT_DIR)
        self.tokenizer = self.processor.tokenizer
        self.generator = backend["Generator"](model, model_args, self.mesh_device)
        warmup_xattn_caches = [None for _ in self.generator.model]
        # Our shape first, tt-metal's second. tt-metal's warmup ends with
        # decode_forward(), whose enable_trace defaults to True, so it leaves a 224x224
        # decode trace live -- which is what the "allocating buffers is unsafe due to an
        # active trace" warning is about. Going first means the request shape is prefilled
        # with no trace resident, which separates "the size is too big" from "the trace
        # was in the way". Set TT_VLM_WARMUP_LAST=1 to go back to the original order.
        warmup_last = os.environ.get("TT_VLM_WARMUP_LAST") == "1"
        if not warmup_last:
            self._warm_request_shape()
        backend["warmup_model"](
            self.generator,
            model_args,
            warmup_xattn_caches,
            HF_MODEL=True,
            processor=self.processor,
        )
        if warmup_last:
            self._warm_request_shape()
        logger.info("[VLM] ready.")

    def _warm_request_shape(self) -> None:
        """Warm the shape real requests will actually have.

        tt-metal's warmup uses a hardcoded 224x224 dummy. That was invisible while the
        HTTP layer squashed every frame to 224x224 -- and it means **every caption this
        server has ever produced went in at the warmup shape**. Every attempt at another
        size has failed. Whether that is the size or the mismatch has never been
        separated, because the two always moved together.

        This removes the mismatch so the size can be tested on its own.

        `letterbox` and `stretch` emit exactly long_side x long_side whatever the client
        sends, so the shape is known here. `contain` keeps the client's aspect ratio and
        the server cannot know it -- set TT_VLM_WARMUP_SHAPE=WxH for that case (CODa
        frames are all 1224x1024, so at long_side 576 they arrive as 576x480).
        """
        import time

        from PIL import Image

        shape = self._request_shape()
        if shape is None:
            logger.info("[VLM] request shape unknown (TT_VLM_IMAGE_FIT=contain without "
                        "TT_VLM_WARMUP_SHAPE); warmed at tt-metal's 224x224 only")
            return
        width, height = shape
        logger.info(f"[VLM] warming the request shape {width}x{height} -- first compile "
                    f"of a new shape can take minutes")
        started = time.perf_counter()
        try:
            self.describe([Image.new("RGB", (width, height), (128, 128, 128))],
                          "Describe what you see.", max_gen_len=8)
        except Exception as error:  # noqa: BLE001
            # The traceback is the point of the run when TT_METAL_OPERATION_TIMEOUT_SECONDS
            # fires: it names the vision block and the ttnn op the device died on (#70).
            logger.exception(f"[VLM] warmup at {width}x{height} failed ({error}); serving anyway")
            return
        logger.info(f"[VLM] warm at {width}x{height} in {time.perf_counter() - started:.0f}s")

    @staticmethod
    def _request_shape():
        """(width, height) every caption will arrive as, or None if it varies."""
        told = os.environ.get("TT_VLM_WARMUP_SHAPE")
        if told:
            try:
                width, height = (int(part) for part in told.lower().split("x"))
                return width, height
            except ValueError:
                logger.warning(f"[VLM] TT_VLM_WARMUP_SHAPE={told!r} is not WIDTHxHEIGHT")
                return None
        if os.environ.get("TT_VLM_IMAGE_FIT", "stretch") not in ("letterbox", "stretch"):
            return None
        side = int(os.environ.get("TT_VLM_IMAGE_LONG_SIDE", "224"))
        return side, side

    def describe(self, images, prompt: str, max_gen_len: int = 128) -> str:
        import time as _time

        started = _time.perf_counter()
        sizes = ", ".join(f"{image.width}x{image.height}" for image in images)
        logger.info(f"[VLM] caption in: {sizes}  max_gen_len={max_gen_len}")

        xattn_caches = [None for _ in self.generator.model]
        sampler = self._get_batch_sampler(0, 0.9, self.model_args[0].tokenizer)
        content = [ImageMedia(image=image) for image in images] + [prompt]
        model_input = self._encode([UserMessage(content=content)], self.processor)

        image_grid_thw = [model_input.image_grid_thw]
        vision_images = [model_input.vision.images]
        vision_mask = [model_input.vision.mask]
        prompt_tokens = model_input.tokens
        prefill_lens = torch.tensor([len(prompt_tokens)], dtype=torch.long)
        total_lens = prefill_lens + max_gen_len

        tokens = torch.full(
            (1, int(max(total_lens))),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        tokens[0, : len(prompt_tokens)] = prompt_tokens.detach().clone().to(torch.long)
        batch_logits, prefill_xmask, prefill_tmask, decode_xmask, decode_tmask = (
            self.generator.prefill_forward(
                vision_images,
                vision_mask,
                tokens,
                xattn_caches,
                total_lens,
                prefill_lens,
                image_grid_thw=image_grid_thw,
            )
        )

        next_tokens, _ = sampler(batch_logits)
        tokens[0, prefill_lens[0]] = next_tokens[0]
        all_outputs = [*tokens[0][: prefill_lens[0]].tolist(), next_tokens[0].item()]
        stop_tokens = {
            "<|eot_id|>",
            "<|eom_id|>",
            "<|im_end|>",
            "<|im_start|>",
            "addCriterion",
        }

        generation_index = 0
        while generation_index < max_gen_len - 1:
            position_id = prefill_lens + generation_index
            logits = self.generator.decode_forward(
                position_id,
                next_tokens.reshape(1, 1),
                prefill_xmask,
                prefill_tmask,
                decode_xmask,
                decode_tmask,
                xattn_caches,
                enable_trace=True,
            )
            next_tokens, _ = sampler(logits)
            tokens[torch.arange(1), position_id + 1] = next_tokens
            token = int(next_tokens[0].item())
            all_outputs.append(token)
            generation_index += 1
            if self.tokenizer.decode([token]) in stop_tokens:
                break

        output_tokens = [
            token for token in all_outputs if token != self.tokenizer.pad_token_id
        ]
        full_text = self.tokenizer.decode(output_tokens)
        prompt_text = self.tokenizer.decode(model_input.tokens.tolist())
        answer = full_text.replace(prompt_text, "", 1)
        for token in stop_tokens:
            answer = answer.replace(token, "")
        ttnn.synchronize_device(self.mesh_device)
        answer = answer.strip()
        # The caption only ever went out over HTTP, so a run watched from the server
        # showed progress bars and nothing about what the model actually said.
        logger.info(f"[VLM] caption out: prefill={int(prefill_lens[0])} tok, "
                    f"{generation_index + 1} generated, "
                    f"{_time.perf_counter() - started:.1f}s")
        logger.info(f"[VLM] {answer}")
        return answer
