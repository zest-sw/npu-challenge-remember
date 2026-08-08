import torch
import ttnn
from loguru import logger
from transformers import AutoProcessor

from llama_models.llama3.api.datatypes import ImageMedia, UserMessage

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
        logger.info("[VLM] ready.")

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
        return answer.strip()
