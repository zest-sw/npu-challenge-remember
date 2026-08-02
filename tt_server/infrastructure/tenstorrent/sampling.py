import torch
from loguru import logger

try:
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

    HAS_LMFE = True
except Exception:  # noqa: BLE001
    HAS_LMFE = False


def masked_sample(
    logits: torch.Tensor,
    token_list,
    vocab: int,
    temperature: float = 0.0,
    debug: bool = False,
) -> int:
    """Select from grammar tokens, or the full vocabulary when token_list is None."""
    row = logits.reshape(-1)
    width = min(row.shape[0], vocab)
    allowed = [token for token in token_list.allowed_tokens if 0 <= token < width] if token_list else []
    if not allowed:
        if temperature > 0:
            probabilities = torch.softmax(row[:width].float() / temperature, dim=-1)
            if debug:
                top_logits = torch.topk(row[:width].float(), 5)
                top_probs = torch.topk(probabilities, 5)
                print(
                    f"[DEBUG_SAMPLE] row.shape={tuple(row.shape)} dtype={row.dtype} "
                    f"width={width} vocab={vocab}\n"
                    f"  raw top5 logits: {list(zip(top_logits.indices.tolist(), top_logits.values.tolist()))}\n"
                    f"  probs top5:      {list(zip(top_probs.indices.tolist(), top_probs.values.tolist()))}\n"
                    f"  probs sum={probabilities.sum().item()} "
                    f"has_nan={torch.isnan(probabilities).any().item()}",
                    flush=True,
                )
            return int(torch.multinomial(probabilities, 1).item())
        return int(torch.argmax(row[:width]).item())

    indices = torch.tensor(allowed, dtype=torch.long)
    selected = row.index_select(0, indices)
    if temperature > 0:
        probabilities = torch.softmax(selected.float() / temperature, dim=-1)
        return int(indices[int(torch.multinomial(probabilities, 1).item())].item())
    return int(indices[int(torch.argmax(selected).item())].item())


class JsonTokenEnforcerFactory:
    def __init__(self, tokenizer, vocab_size: int):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self._tokenizer_data = None

    def create(self, schema):
        if not HAS_LMFE:
            raise RuntimeError("lm-format-enforcer not installed; `pip install lm-format-enforcer`")
        if self._tokenizer_data is None:
            logger.info("[LLM] building JSON grammar tokenizer data ...")
            self._tokenizer_data = build_token_enforcer_tokenizer_data(
                self.tokenizer,
                vocab_size=self.vocab_size,
            )
        return TokenEnforcer(self._tokenizer_data, JsonSchemaParser(schema))
