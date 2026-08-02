from dataclasses import dataclass


@dataclass(frozen=True)
class ModeSpec:
    llm: str | None
    vlm: str | None


MODES = {
    "3b+3b": ModeSpec(
        llm="meta-llama/Llama-3.2-3B-Instruct",
        vlm="Qwen/Qwen2.5-VL-3B-Instruct",
    ),
    "caption-3b": ModeSpec(llm=None, vlm="Qwen/Qwen2.5-VL-3B-Instruct"),
    "llm-8b": ModeSpec(llm="meta-llama/Llama-3.1-8B-Instruct", vlm=None),
    "caption-8b": ModeSpec(llm=None, vlm="Qwen/Qwen3-VL-8B-Instruct"),
}
