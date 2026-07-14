# SPDX-License-Identifier: Apache-2.0
"""Unified Tenstorrent NPU inference server for ReMEmbR.

Hosts BOTH models needed by remembr in a SINGLE process that owns the NPU:
  - LLM  : Llama-3.2-3B-Instruct   -> OpenAI-compatible  POST /v1/chat/completions
  - VLM  : Qwen2.5-VL-3B-Instruct  ->                    POST /caption

Why one process:
  * ttnn opens the Blackhole board exclusively per-process, so two separate
    server processes cannot both hold the device.
  * The NPU executes one op-graph at a time; concurrent caption+LLM requests are
    therefore serialized through a single worker thread + queue.

All device work (open, model load, prefill, decode) happens on ONE worker thread.
FastAPI handlers merely enqueue a job and block on its Future, so multiple HTTP
requests can wait concurrently while the device runs them one at a time.

Run (from the tt-metal repo root, inside the tt-metal python_env):
    source env_set.sh
    export HUGGING_FACE_HUB_TOKEN=hf_...            # for gated Llama weights
    export MESH_DEVICE=N150                         # match your board
    # optionally: TT_ENABLE_LLM=1 TT_ENABLE_VLM=1   (both default on)
    python models/bos_model/tt_infer_server.py
"""

import base64
import concurrent.futures
import io
import os
import queue
import threading
import time
import uuid

import torch
import ttnn
from loguru import logger
from PIL import Image

# ---- llama32 (LLM) imports -------------------------------------------------
from models.bos_model.llama32.tt.common import (
    PagedAttentionConfig,
    create_tt_model,
    preprocess_inputs_prefill,
)
from models.bos_model.llama32.tt.generator import Generator as LlamaGenerator
from models.bos_model.llama32.tt.generator import SamplingParams, create_submeshes
from models.bos_model.llama32.tt.model_config import DecodersPrecision
from models.bos_model.llama32.demo_llama32 import create_tt_page_table

# ---- qwen25_vl (VLM) imports ----------------------------------------------
from llama_models.llama3.api.datatypes import ImageMedia, UserMessage
from models.bos_model.qwen25_vl.tt.common import hf_multimodal_encode
from models.bos_model.qwen25_vl.tt.generator import Generator as QwenGenerator
from models.bos_model.qwen25_vl.demo_qwen25_vl import (
    get_batch_sampler,
    prepare_generator_args as qwen_prepare_generator_args,
    warmup_model as qwen_warmup_model,
)

# Optional: JSON-constrained (grammar) decoding, the equivalent of Ollama's
# format="json". Guarded so a VLM-only server or a missing dep still runs.
try:
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

    _HAS_LMFE = True
except Exception:  # noqa: BLE001
    _HAS_LMFE = False


def _masked_argmax(logits: "torch.Tensor", token_list, vocab: int) -> int:
    """Greedy argmax over ONLY the grammar-allowed token ids (host-side mask).
    Accepts a logits tensor of any rank whose flattened length is the vocab row
    (batch=1)."""
    row = logits.reshape(-1)
    w = min(row.shape[0], vocab)
    allowed = [t for t in token_list.allowed_tokens if 0 <= t < w]
    if not allowed:  # grammar dead-end (shouldn't happen) -> fall back to plain argmax
        return int(torch.argmax(row[:w]).item())
    idx = torch.tensor(allowed, dtype=torch.long)
    sub = row.index_select(0, idx)
    return int(idx[int(torch.argmax(sub).item())].item())


# Model repos (override via env)
LLM_MODEL = os.environ.get("TT_LLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
VLM_MODEL = os.environ.get("TT_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")

ENABLE_LLM = os.environ.get("TT_ENABLE_LLM", "1") == "1"
ENABLE_VLM = os.environ.get("TT_ENABLE_VLM", "1") == "1"

LLM_MAX_SEQ_LEN = int(os.environ.get("TT_LLM_MAX_SEQ_LEN", "4096"))
VLM_MAX_SEQ_LEN = int(os.environ.get("TT_VLM_MAX_SEQ_LEN", "2048"))

DEFAULT_VLM_PROMPT = (
    "<video> Describe in detail what you see: people, objects, environment, and "
    "activities. Be specific."
)


def _set_hf_model(name: str):
    """ModelArgs picks the checkpoint from os.environ['HF_MODEL']; make sure only
    HF_MODEL (not LLAMA_DIR) is set when we construct a model."""
    os.environ.pop("LLAMA_DIR", None)
    os.environ["HF_MODEL"] = name


# ===========================================================================
# LLM engine (Llama-3.2-3B) — lifts the per-prompt loop from
# demo_llama32.llama_inference (lines ~292-453), returning a string.
# ===========================================================================
class LlamaEngine:
    def __init__(self, mesh_device, max_seq_len=LLM_MAX_SEQ_LEN):
        self.mesh_device = mesh_device
        self.max_seq_len = max_seq_len
        self.page_params = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}
        self._enforcer_data = None  # lazily built token-enforcer data (grammar decoding)

    def _ensure_enforcer_data(self):
        """Build the lm-format-enforcer tokenizer data once (iterates the full
        vocab, ~a few seconds), on first JSON request."""
        if not _HAS_LMFE:
            raise RuntimeError("lm-format-enforcer not installed; `pip install lm-format-enforcer`")
        if self._enforcer_data is None:
            logger.info("[LLM] building JSON grammar tokenizer data ...")
            self._enforcer_data = build_token_enforcer_tokenizer_data(
                self.tokenizer, vocab_size=self.model_args[0].vocab_size
            )
        return self._enforcer_data

    def load(self):
        _set_hf_model(LLM_MODEL)
        logger.info(f"[LLM] loading {LLM_MODEL} ...")
        paged_attention_config = PagedAttentionConfig(
            block_size=self.page_params["page_block_size"],
            max_num_blocks=self.page_params["page_max_num_blocks_per_dp"],
        )
        submesh_devices = create_submeshes(self.mesh_device, 1)
        model_args, model, tt_kv_cache = [], [], []
        state_dict = None
        for submesh in submesh_devices:
            a, m, kv, state_dict = create_tt_model(
                mesh_device=submesh,
                instruct=True,
                max_batch_size=1,
                optimizations=lambda margs: DecodersPrecision.performance(margs.n_layers, margs.model_name),
                max_seq_len=self.max_seq_len,
                paged_attention_config=paged_attention_config,
                dtype=ttnn.bfloat8_b,
                state_dict=state_dict,
            )
            model_args.append(a)
            model.append(m)
            tt_kv_cache.append(kv)
        self.model_args = model_args
        self.model = model
        self.tt_kv_cache = tt_kv_cache
        self.page_table = create_tt_page_table(
            global_batch_size=1, data_parallel=1, paged_attention_config=paged_attention_config
        )
        self.tokenizer = model_args[0].tokenizer
        self.generator = LlamaGenerator(model, model_args, self.mesh_device, tokenizer=self.tokenizer)
        logger.info("[LLM] ready.")

    def generate(self, prompt: str, max_new_tokens: int = 256, force_json: bool = False, json_schema=None) -> str:
        model_args, model, tokenizer = self.model_args, self.model, self.tokenizer
        generator, page_table, tt_kv_cache = self.generator, self.page_table, self.tt_kv_cache
        mesh_device = self.mesh_device

        # Guard: prefill + decode must fit max_seq_len. The chat template adds a
        # handful of tokens, so budget conservatively and truncate an over-long
        # prompt (keeps the head/instructions) instead of crashing in preprocess.
        max_new_tokens = max(1, min(max_new_tokens, self.max_seq_len // 2))
        reserve = max_new_tokens + 48  # gen tokens + chat-template overhead margin
        budget = self.max_seq_len - reserve
        ids = tokenizer.encode(prompt)
        if len(ids) > budget:
            logger.warning(f"[LLM] prompt {len(ids)} tok > budget {budget}; truncating")
            prompt = tokenizer.decode(ids[:budget])

        (input_tokens_prefill_pt, encoded_prompts, decoding_pos, prefill_lens) = preprocess_inputs_prefill(
            [prompt], tokenizer, model_args, True, max_new_tokens, max_prefill_len=self.max_seq_len
        )
        max_prompt_len = max(len(p) for p in encoded_prompts)
        max_new_tokens = max(1, min(max_new_tokens, self.max_seq_len - max_prompt_len))

        # zero the KV cache for this fresh request
        for i in range(len(model)):
            for layer in model[i].layers:
                k_cache, v_cache = layer.attention.layer_past
                ttnn.mul(k_cache, 0, output_tensor=k_cache)
                ttnn.mul(v_cache, 0, output_tensor=v_cache)

        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(1, -1)

        generator.prefill_forward_text(
            input_tokens_prefill_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos
        )
        logits = generator.prefill_forward_text(
            input_tokens_prefill_pt, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos
        )
        # ---- JSON-constrained (grammar) decoding: the format="json" equivalent ----
        # Host-side masking: run decode with sampling_params=None so decode_forward_text
        # returns raw host logits [1,1,vocab]; mask to grammar-allowed ids; argmax.
        if force_json:
            vocab = model_args[0].vocab_size
            enforcer = TokenEnforcer(self._ensure_enforcer_data(), JsonSchemaParser(json_schema))
            first = _masked_argmax(logits, enforcer.get_allowed_tokens([]), vocab)
            generated = [] if first in tokenizer.stop_tokens else [first]
            current_pos = torch.tensor([decoding_pos[0]])
            out_tok = torch.tensor([[first]])
            for _ in range(max_new_tokens - 1):
                dlogits = generator.decode_forward_text(
                    out_tok, current_pos, enable_trace=True,
                    page_table=page_table, kv_cache=tt_kv_cache, sampling_params=None,
                )
                nxt = _masked_argmax(dlogits, enforcer.get_allowed_tokens(generated), vocab)
                current_pos = current_pos + 1
                if nxt in tokenizer.stop_tokens:
                    break
                generated.append(nxt)
                out_tok = torch.tensor([[nxt]])
            ttnn.synchronize_device(mesh_device)
            return tokenizer.decode(generated).strip()

        # ---- unconstrained device-greedy path ----
        prefilled_token = torch.argmax(logits, dim=-1)

        device_sampling_params = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0)  # greedy on device

        first = int(prefilled_token[0].item())
        generated = []
        if first not in tokenizer.stop_tokens:
            generated.append(first)

        current_pos = torch.tensor([decoding_pos[0]])
        out_tok = prefilled_token
        op_event = ttnn.record_event(mesh_device, 0)
        for _ in range(max_new_tokens - 1):
            ttnn.wait_for_event(1, op_event)
            logits = generator.decode_forward_text(
                out_tok,
                current_pos,
                enable_trace=True,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                sampling_params=device_sampling_params,
            )
            out_tok = logits.unsqueeze(1)
            current_pos = current_pos + 1
            write_event = ttnn.record_event(mesh_device, 1)
            ttnn.wait_for_event(0, write_event)

            tok = int(out_tok[0].item())
            if tok in tokenizer.stop_tokens:
                break
            generated.append(tok)
            op_event = ttnn.record_event(mesh_device, 0)

        ttnn.synchronize_device(mesh_device)
        return tokenizer.decode(generated).strip()


# ===========================================================================
# VLM engine (Qwen2.5-VL-3B) — lifts the per-prompt loop from
# demo_qwen25_vl.qwen_inference (HF path, batch 1), returning a caption string.
# ===========================================================================
class QwenVLEngine:
    def __init__(self, mesh_device, max_seq_len=VLM_MAX_SEQ_LEN):
        self.mesh_device = mesh_device
        self.max_seq_len = max_seq_len

    def load(self):
        from transformers import AutoProcessor

        _set_hf_model(VLM_MODEL)
        logger.info(f"[VLM] loading {VLM_MODEL} ...")
        num_devices = self.mesh_device.get_num_devices() if isinstance(self.mesh_device, ttnn.MeshDevice) else 1
        model_args, model = qwen_prepare_generator_args(
            num_devices=num_devices,
            data_parallel=1,
            mesh_device=self.mesh_device,
            max_batch_size=1,
            max_seq_len=self.max_seq_len,
        )
        self.model_args = model_args
        self.processor = AutoProcessor.from_pretrained(model_args[0].CKPT_DIR)
        self.tokenizer = self.processor.tokenizer
        self.generator = QwenGenerator(model, model_args, self.mesh_device)
        self.xattn_caches = [None for _ in self.generator.model]  # HF path uses no cross-attn cache
        qwen_warmup_model(self.generator, model_args, self.xattn_caches, HF_MODEL=True, processor=self.processor)
        logger.info("[VLM] ready.")

    def describe(self, images, prompt: str, max_gen_len: int = 128) -> str:
        processor, tokenizer, generator = self.processor, self.tokenizer, self.generator
        model_args, xattn_caches = self.model_args, self.xattn_caches
        sampler = get_batch_sampler(0, 0.9, model_args[0].tokenizer)  # greedy

        content = [ImageMedia(image=im) for im in images] + [prompt]
        dialog = [UserMessage(content=content)]
        model_input = hf_multimodal_encode(dialog, processor)

        image_grid_thw = [model_input.image_grid_thw]
        vision_images = [model_input.vision.images]
        vision_mask = [model_input.vision.mask]
        prompt_tokens = model_input.tokens
        prefill_lens = torch.tensor([len(prompt_tokens)], dtype=torch.long)
        total_lens = prefill_lens + max_gen_len

        pad_id = tokenizer.pad_token_id
        tokens = torch.full((1, int(max(total_lens))), pad_id, dtype=torch.long)
        tokens[0, : len(prompt_tokens)] = prompt_tokens.detach().clone().to(torch.long)

        # prefill (warmup + real, mirroring the demo)
        masks = generator.prefill_forward(
            vision_images, vision_mask, tokens, xattn_caches, total_lens, prefill_lens, image_grid_thw=image_grid_thw
        )
        batch_logits, pxm, ptm, dxm, dtm = generator.prefill_forward(
            vision_images, vision_mask, tokens, xattn_caches, total_lens, prefill_lens, image_grid_thw=image_grid_thw
        )

        next_tokens, _ = sampler(batch_logits)
        tokens[0, prefill_lens[0]] = next_tokens[0]
        all_outputs = [*tokens[0][: prefill_lens[0]].tolist(), next_tokens[0].item()]

        skip_tokens = {"<|eot_id|>", "<|eom_id|>", "<|im_end|>", "<|im_start|>", "addCriterion"}
        users_decoding = True
        gen_idx = 0
        while users_decoding and gen_idx < max_gen_len - 1:
            position_id = prefill_lens + gen_idx
            next_token_tensor = next_tokens.reshape(1, 1)
            logits = generator.decode_forward(
                position_id, next_token_tensor, pxm, ptm, dxm, dtm, xattn_caches, enable_trace=True
            )
            next_tokens, _ = sampler(logits)
            tokens[torch.arange(1), position_id + 1] = next_tokens
            tok = int(next_tokens[0].item())
            all_outputs.append(tok)
            if tokenizer.decode([tok]) in skip_tokens:
                users_decoding = False
            gen_idx += 1

        # Free this request's device tensors so they do NOT accumulate in device
        # DRAM across captions — that accumulation is what makes each successive
        # caption slower (per-request latency creeps up over a long run). These
        # are all created fresh per request (the next request re-allocates them),
        # so releasing them here is safe. Best-effort + guarded: anything that is
        # not a live ttnn device tensor is simply ignored.
        for _t in (batch_logits, pxm, ptm, dxm, dtm, masks):
            try:
                ttnn.deallocate(_t)
            except Exception:  # noqa: BLE001 - non-ttnn / already-freed
                pass
        try:
            ttnn.deallocate(logits)  # last decode logits (undefined-safe via except)
        except Exception:  # noqa: BLE001
            pass

        tokens_out = [t for t in all_outputs if t != tokenizer.pad_token_id]
        full = tokenizer.decode(tokens_out)
        prompt_str = tokenizer.decode(model_input.tokens.tolist())
        answer = full.replace(prompt_str, "", 1)
        for s in skip_tokens:
            answer = answer.replace(s, "")
        return answer.strip()


# ===========================================================================
# Single-threaded device worker + job queue
# ===========================================================================
_jobs: "queue.Queue" = queue.Queue()
_ready = threading.Event()
_engines = {"llm": None, "vlm": None}


def _worker():
    mesh_device = _open_device()
    try:
        if ENABLE_LLM:
            eng = LlamaEngine(mesh_device)
            eng.load()
            _engines["llm"] = eng
        if ENABLE_VLM:
            eng = QwenVLEngine(mesh_device)
            eng.load()
            _engines["vlm"] = eng
        # After load, leave HF_MODEL set to a truthy value so Qwen's decode_forward
        # keeps routing to its HF text path (Llama reads model_args, not env).
        if ENABLE_VLM:
            os.environ["HF_MODEL"] = VLM_MODEL
        _ready.set()
        logger.info("Server device worker ready; models loaded.")

        while True:
            fn, args, fut = _jobs.get()
            if fn is None:  # shutdown sentinel
                break
            try:
                fut.set_result(fn(*args))
            except Exception as e:  # noqa: BLE001
                fut.set_exception(e)
    finally:
        ttnn.close_mesh_device(mesh_device)


def _open_device():
    mapping = {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}
    param = mapping.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    device_ids = ttnn.get_device_ids()
    if isinstance(param, tuple):
        mesh_shape = ttnn.MeshShape(*param)
    else:
        mesh_shape = ttnn.MeshShape(1, min(param, len(device_ids)))
    trace_region_size = int(os.environ.get("TT_TRACE_REGION_SIZE", str(30_000_000)))
    return ttnn.open_mesh_device(mesh_shape=mesh_shape, trace_region_size=trace_region_size, num_command_queues=2)


def _submit(fn, *args):
    fut: "concurrent.futures.Future" = concurrent.futures.Future()
    _jobs.put((fn, args, fut))
    return fut.result()  # block until the worker finishes this job


# ===========================================================================
# HTTP layer (FastAPI)
# ===========================================================================
from fastapi import FastAPI, HTTPException  # noqa: E402

app = FastAPI(title="tt-infer-server")


def _messages_to_prompt(messages) -> str:
    """Flatten OpenAI chat messages into a single instruction string.
    remembr's NonAgent sends one user message containing the full prompt, so this
    is usually a passthrough."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):  # OpenAI content parts
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        parts.append(content if role == "user" else f"[{role}] {content}")
    return "\n\n".join(parts)


def _decode_image(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))


@app.get("/health")
def health():
    return {"ready": _ready.is_set(), "llm": _engines["llm"] is not None, "vlm": _engines["vlm"] is not None}


@app.post("/v1/chat/completions")
def chat_completions(req: dict):
    if _engines["llm"] is None:
        raise HTTPException(503, "LLM not loaded (TT_ENABLE_LLM=0?)")
    messages = req.get("messages", [])
    max_tokens = int(req.get("max_tokens") or 256)
    prompt = _messages_to_prompt(messages)
    # Honor OpenAI response_format: json_object -> any valid JSON (Ollama format="json"
    # equivalent); json_schema -> constrain to the given JSON schema.
    rf = req.get("response_format") or {}
    force_json = rf.get("type") in ("json_object", "json_schema")
    json_schema = None
    if rf.get("type") == "json_schema":
        js = rf.get("json_schema") or {}
        json_schema = js.get("schema", js)
    text = _submit(_engines["llm"].generate, prompt, max_tokens, force_json, json_schema)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.get("model", LLM_MODEL),
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
    }


@app.post("/caption")
def caption(req: dict):
    if _engines["vlm"] is None:
        raise HTTPException(503, "VLM not loaded (TT_ENABLE_VLM=0?)")
    images = [_decode_image(b) for b in req.get("images", [])]
    if not images:
        raise HTTPException(400, "no images provided")
    prompt = req.get("prompt", DEFAULT_VLM_PROMPT)
    max_gen_len = int(req.get("max_gen_len") or 128)
    text = _submit(_engines["vlm"].describe, images, prompt, max_gen_len)
    return {"caption": text}


def main():
    import uvicorn
    import signal

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    logger.info("Loading models on the NPU (this can take several minutes on first run)...")
    _ready.wait()  # block startup until models are on device
    try:
        uvicorn.run(app, host=os.environ.get("TT_HOST", "0.0.0.0"), port=int(os.environ.get("TT_PORT", "8100")))
    finally:
        # Clean NPU teardown. The worker's `finally: close_mesh_device()` is the
        # ONLY thing that releases the board, but the worker is a daemon thread
        # whose finally is skipped if the interpreter just exits -> the device is
        # left active -> the board wedges. So on shutdown we explicitly signal the
        # worker to stop and JOIN it, letting close_mesh_device() run to completion.
        #
        # We also ignore further SIGINT/SIGTERM here: a second Ctrl-C during this
        # window would otherwise abort the device close mid-way and wedge the board
        # (exactly the "mash Ctrl-C -> board dies" failure). Only SIGKILL (-9) can
        # still interrupt this — which is why -9 must never be used on this server.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        logger.info("Shutting down: releasing the NPU cleanly — do NOT force-kill (-9)...")
        _jobs.put((None, None, None))  # shutdown sentinel -> worker breaks + closes device
        worker.join(timeout=180)
        if worker.is_alive():
            logger.warning("Device worker still busy after 180s (op in flight?); "
                           "device may not have closed cleanly.")
        else:
            logger.info("NPU released cleanly. Safe to exit.")


if __name__ == "__main__":
    main()
