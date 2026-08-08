import os
import time

import torch
import ttnn
from loguru import logger
from PIL import Image
from transformers import AutoProcessor

from llama_models.llama3.api.datatypes import ImageMedia, UserMessage

from tt_server.domain.models import check_generation_fits

from .model_env import select_hf_model
from .vlm_backends import load_vlm_backend


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
        self.xattn_caches = [None for _ in self.generator.model]
        backend["warmup_model"](
            self.generator,
            model_args,
            self.xattn_caches,
            HF_MODEL=True,
            processor=self.processor,
        )
        self._warm_real_shape()
        logger.info("[VLM] ready.")

    def _warm_real_shape(self) -> None:
        """Compile for the frame shape this server will actually be sent.

        tt-metal's warmup uses a hardcoded 224x224 dummy, so the first request at any
        other size compiles then -- minutes of silence that reads as a hang, since the
        worker holds the GIL (#62). Paying it at load puts it before /health reports
        ready, with no client on a socket.

            TT_VLM_IMAGE_LONG_SIDE=1224 TT_VLM_WARMUP_SIZE=1224x1024

        The size is what the engine receives, i.e. after the HTTP layer's cap. Unset
        skips it."""
        size = os.environ.get("TT_VLM_WARMUP_SIZE")
        if not size:
            logger.info(
                "[VLM] TT_VLM_WARMUP_SIZE unset -- tt-metal warmed 224x224 only, so the "
                "first caption at any other size pays its compile (can be minutes)"
            )
            return
        try:
            width, height = (int(part) for part in size.lower().split("x"))
        except ValueError:
            logger.warning(f"[VLM] TT_VLM_WARMUP_SIZE={size!r} is not WIDTHxHEIGHT; skipping")
            return
        logger.info(f"[VLM] warming the real frame shape {width}x{height} "
                    f"-- first compile of a new shape can take minutes")
        started = time.perf_counter()
        try:
            self.describe([Image.new("RGB", (width, height), (128, 128, 128))],
                          "Describe what you see.", max_gen_len=8)
        except Exception as error:  # noqa: BLE001
            # A failed warmup is worth reporting but not worth refusing to serve over:
            # the shape may simply not fit, and the request-time guard says so clearly.
            logger.warning(f"[VLM] warmup at {width}x{height} failed ({error}); serving anyway")
            return
        logger.info(f"[VLM] warm at {width}x{height} in {time.perf_counter() - started:.0f}s")

    def describe(self, images, prompt: str, max_gen_len: int = 128) -> str:
        sampler = self._get_batch_sampler(0, 0.9, self.model_args[0].tokenizer)
        content = [ImageMedia(image=image) for image in images] + [prompt]
        model_input = self._encode([UserMessage(content=content)], self.processor)

        image_grid_thw = [model_input.image_grid_thw]
        vision_images = [model_input.vision.images]
        vision_mask = [model_input.vision.mask]
        prompt_tokens = model_input.tokens
        prefill_lens = torch.tensor([len(prompt_tokens)], dtype=torch.long)
        total_lens = prefill_lens + max_gen_len
        check_generation_fits(int(prefill_lens[0]), max_gen_len, self.max_seq_len)

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
                self.xattn_caches,
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
                self.xattn_caches,
                # No device tracing. tt-metal caches the decode trace on
                # `hasattr(self, "trace_ids_text")` alone, so it outlives the request
                # that captured it and later prefills allocate buffers beside it --
                # tt-metal warns those may be corrupted. Same bug as #21. Costs a
                # slower decode; a wedged board is not faster.
                enable_trace=False,
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
        return answer.strip()
