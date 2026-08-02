def load_vlm_backend(model_name: str):
    """Load the matching tt-metal backend without imposing Qwen3 deps on Qwen2.5."""
    if "Qwen3-VL" in model_name:
        from models.bos_model.qwen3_vl.demo_qwen3_vl import (
            get_batch_sampler,
            prepare_generator_args,
            warmup_model,
        )
        from models.bos_model.qwen3_vl.tt.common import hf_multimodal_encode
        from models.bos_model.qwen3_vl.tt.generator import Generator

        precision = None
    else:
        from models.bos_model.qwen25_vl.demo_qwen25_vl import (
            get_batch_sampler,
            prepare_generator_args,
            warmup_model,
        )
        from models.bos_model.qwen25_vl.tt.common import hf_multimodal_encode
        from models.bos_model.qwen25_vl.tt.generator import Generator
        from models.bos_model.qwen25_vl.tt.model_config import DecodersPrecision

        precision = DecodersPrecision

    return {
        "hf_multimodal_encode": hf_multimodal_encode,
        "Generator": Generator,
        "DecodersPrecision": precision,
        "get_batch_sampler": get_batch_sampler,
        "prepare_generator_args": prepare_generator_args,
        "warmup_model": warmup_model,
    }
