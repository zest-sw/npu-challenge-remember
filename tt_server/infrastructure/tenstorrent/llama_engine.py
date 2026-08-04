import torch
import ttnn
from loguru import logger

from models.bos_model.llama32.demo_llama32 import create_tt_page_table
from models.bos_model.llama32.tt.common import (
    PagedAttentionConfig,
    create_tt_model,
    preprocess_inputs_prefill,
)
from models.bos_model.llama32.tt.generator import Generator as LlamaGenerator
from models.bos_model.llama32.tt.generator import create_submeshes
from models.bos_model.llama32.tt.model_config import DecodersPrecision

from .model_env import select_hf_model
from .sampling import JsonTokenEnforcerFactory, masked_sample


class LlamaEngine:
    def __init__(
        self,
        mesh_device,
        model_name: str,
        max_seq_len: int,
        debug_decode: bool = False,
        debug_sample: bool = False,
    ):
        self.mesh_device = mesh_device
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.debug_decode = debug_decode
        self.debug_sample = debug_sample
        self.page_params = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}

    def load(self) -> None:
        select_hf_model(self.model_name)
        logger.info(f"[LLM] loading {self.model_name} ...")
        paged_attention_config = PagedAttentionConfig(
            block_size=self.page_params["page_block_size"],
            max_num_blocks=self.page_params["page_max_num_blocks_per_dp"],
        )
        submesh_devices = create_submeshes(self.mesh_device, 1)
        model_args, model, tt_kv_cache = [], [], []
        state_dict = None
        for submesh in submesh_devices:
            args, loaded_model, cache, state_dict = create_tt_model(
                mesh_device=submesh,
                instruct=True,
                max_batch_size=1,
                optimizations=lambda margs: DecodersPrecision.performance(
                    margs.n_layers, margs.model_name
                ),
                max_seq_len=self.max_seq_len,
                paged_attention_config=paged_attention_config,
                dtype=ttnn.bfloat8_b,
                state_dict=state_dict,
            )
            model_args.append(args)
            model.append(loaded_model)
            tt_kv_cache.append(cache)

        self.model_args = model_args
        self.model = model
        self.tt_kv_cache = tt_kv_cache
        self.page_table = create_tt_page_table(
            global_batch_size=1,
            data_parallel=1,
            paged_attention_config=paged_attention_config,
        )
        self.tokenizer = model_args[0].tokenizer
        self.generator = LlamaGenerator(model, model_args, self.mesh_device, tokenizer=self.tokenizer)
        self.precision = getattr(getattr(model_args[0], "optimizations", None), "__name__", None)
        self.json_enforcers = JsonTokenEnforcerFactory(
            self.tokenizer,
            vocab_size=model_args[0].vocab_size,
        )
        logger.info(f"[LLM] ready. model={self.model_name} precision={self.precision}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        force_json: bool = False,
        json_schema=None,
        temperature: float = 0.0,
    ) -> str:
        model_args, model, tokenizer = self.model_args, self.model, self.tokenizer
        generator, page_table, tt_kv_cache = self.generator, self.page_table, self.tt_kv_cache

        max_new_tokens = max(1, min(max_new_tokens, self.max_seq_len // 2))
        reserve = max_new_tokens + 48
        budget = self.max_seq_len - reserve
        prompt_ids = tokenizer.encode(prompt)
        if len(prompt_ids) > budget:
            logger.warning(f"[LLM] prompt {len(prompt_ids)} tok > budget {budget}; truncating")
            prompt = tokenizer.decode(prompt_ids[:budget])

        input_tokens, encoded_prompts, decoding_pos, _ = preprocess_inputs_prefill(
            [prompt],
            tokenizer,
            model_args,
            True,
            max_new_tokens,
            max_prefill_len=self.max_seq_len,
        )
        max_prompt_len = max(len(encoded) for encoded in encoded_prompts)
        max_new_tokens = max(1, min(max_new_tokens, self.max_seq_len - max_prompt_len))

        self._clear_kv_cache()
        input_tokens = torch.stack(input_tokens).view(1, -1)
        logits = generator.prefill_forward_text(
            input_tokens,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
        )

        # One decode path only, on purpose. There used to be a second one that let
        # the device do the argmax (`sampling_params=SamplingParams(...)`), taken
        # for plain temperature=0 requests because it skips the per-token logits
        # readback. It has to go: tt-metal caches the decode trace on
        # `hasattr(self, "trace_ids_text")` alone (generator.py:370), so the trace
        # is NOT keyed on what it returns. Whichever kind of request arrived first
        # captured the trace, and every request of the other kind then replayed it
        # and read the wrong buffer -- for the rest of the process's life:
        #
        #   fresh server -> plain then JSON : JSON returns '{"!!!!!!!!...'
        #                                     (token 0 wins the masked argmax)
        #   fresh server -> JSON then plain : plain returns 'Hello9001/+,,+,**+++'
        #   fresh server -> one kind only   : correct
        #
        # That silently poisoned measurements: a single plain warm-up call made
        # every later schema-constrained answer garbage. Keeping only the host
        # path means one trace shape is ever captured, so the mismatch cannot
        # happen. Cost is a 513 KB logits read per token (~20 us over PCIe)
        # against a measured 293.5 ms/token decode -- under 0.1%, and the earlier
        # end-to-end comparison of the two paths came out at ~3%. See issue #21.
        enforcer = self.json_enforcers.create(json_schema) if force_json else None
        return self._decode_on_host(
            logits,
            decoding_pos[0],
            max_new_tokens,
            temperature,
            enforcer,
        )

    def _clear_kv_cache(self) -> None:
        for loaded_model in self.model:
            for layer in loaded_model.layers:
                key_cache, value_cache = layer.attention.layer_past
                ttnn.mul(key_cache, 0, output_tensor=key_cache)
                ttnn.mul(value_cache, 0, output_tensor=value_cache)

    def _decode_on_host(
        self,
        logits,
        decoding_pos: int,
        max_new_tokens: int,
        temperature: float,
        enforcer,
    ) -> str:
        tokenizer = self.tokenizer
        vocab = self.model_args[0].vocab_size
        allowed = enforcer.get_allowed_tokens([]) if enforcer else None
        first = masked_sample(logits, allowed, vocab, temperature, self.debug_sample)
        generated = [] if first in tokenizer.stop_tokens else [first]
        self._print_token(first, bool(generated))
        current_pos = torch.tensor([decoding_pos])
        output_token = torch.tensor([[first]])

        for _ in range(max_new_tokens - 1):
            decode_logits = self.generator.decode_forward_text(
                output_token,
                current_pos,
                enable_trace=True,
                page_table=self.page_table,
                kv_cache=self.tt_kv_cache,
                sampling_params=None,
            )
            allowed = enforcer.get_allowed_tokens(generated) if enforcer else None
            next_token = masked_sample(
                decode_logits,
                allowed,
                vocab,
                temperature,
                self.debug_sample,
            )
            current_pos = current_pos + 1
            if next_token in tokenizer.stop_tokens:
                break
            generated.append(next_token)
            self._print_token(next_token, True)
            output_token = torch.tensor([[next_token]])

        ttnn.synchronize_device(self.mesh_device)
        return self._finish_decode(generated)

    def _print_token(self, token: int, enabled: bool) -> None:
        if self.debug_decode and enabled:
            print(self.tokenizer.decode([token]), end="", flush=True)

    def _finish_decode(self, generated: list[int]) -> str:
        text = self.tokenizer.decode(generated).strip()
        if self.debug_decode:
            print("", flush=True)
        return text
