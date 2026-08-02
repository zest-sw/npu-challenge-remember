import os


def select_hf_model(name: str) -> None:
    """ModelArgs reads its checkpoint from process environment."""
    os.environ.pop("LLAMA_DIR", None)
    os.environ["HF_MODEL"] = name
