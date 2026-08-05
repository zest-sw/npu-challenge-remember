"""How much would prompt-lookup decoding buy us, using answers we already recorded?

Why this exists
---------------
Decode on this board sits at 97% of the memory ceiling (issue #30): a token costs one
full sweep of the 8B weights (~8 GB) and the measured peak is 33.0 GB/s, so 250 ms/token
is 3% above the 242 ms floor. Kernel work has nothing left to give. The only lossless
lever is to stop paying one sweep per token (issue #39).

Prompt-lookup decoding does that without a draft model: match the tail of what has been
generated against the prompt, propose the continuation that followed it there, and let
the target model verify the whole run in ONE sweep. Wrong guesses cost nothing -- the
verification still confirms one token -- so the only question is the acceptance rate.

That rate is measurable offline from answers already on disk. No board, no server: take
each recorded answer, replay the algorithm against the prompt it was generated from, and
count how many sweeps it would have taken.

Our answers look unusually copy-heavy, which is why this is worth checking before
building anything: asked where a statue was, the model wrote the retrieved memory's
coordinate `[-60.197, 196.541, -1.097]` back out to three decimals, and the JSON keys it
must emit are spelled out in the prompt template.

    python remembr/scripts/estimate_prompt_lookup_gain.py \
        --result out/0/human_qa_SUBSET/optimal+tt_Llama-3.1-8B-Instruct__captions_tt-qwen25-vl-3b_3_secs_typed2.json

Reported speedup is tokens per target sweep, which maps straight onto latency: at 250 ms
per sweep, 2x means 125 ms/token.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.utils.util import file_to_string  # noqa: E402

PROMPT_FILE = os.path.join(os.path.dirname(__file__), "..", "prompts", "non_agent_system_prompt.txt")


def build_prompt(context, question):
    """The prompt NonAgent sends, so the lookup searches what the model actually saw."""
    template = file_to_string(PROMPT_FILE)
    return (template
            .replace("{context}", context)
            .replace("{question}", question)
            .replace("{{", "{").replace("}}", "}"))


def find_last(haystack, needle):
    """Index of the last occurrence of needle in haystack, or -1."""
    for start in range(len(haystack) - len(needle), -1, -1):
        if haystack[start:start + len(needle)] == needle:
            return start
    return -1


def simulate(prompt_ids, answer_ids, ngram, max_proposal):
    """Target sweeps prompt-lookup would need, and how many tokens each one confirmed.

    One sweep always confirms the next token; a matching proposal confirms extra ones
    for free. This is the standard algorithm: no draft model, guesses come from the
    prompt itself.
    """
    index, sweeps, accepted_total = 0, 0, 0
    while index < len(answer_ids):
        sweeps += 1
        index += 1  # the sweep itself confirms one token
        if index < len(answer_ids) and index >= ngram:
            tail = answer_ids[index - ngram:index]
            at = find_last(prompt_ids, tail)
            if at != -1:
                proposal = prompt_ids[at + ngram:at + ngram + max_proposal]
                accepted = 0
                for offset, token in enumerate(proposal):
                    if index + offset >= len(answer_ids) or answer_ids[index + offset] != token:
                        break
                    accepted += 1
                index += accepted
                accepted_total += accepted
    return sweeps, accepted_total


def load_tokenizer(name):
    from transformers import AutoTokenizer
    os.environ.setdefault("HF_HOME", "/home/user/tt-metal/huggingface")
    return AutoTokenizer.from_pretrained(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, help="an eval result JSON from out/")
    parser.add_argument("--qa_file", default="remembr/data/questions/0/human_qa_SUBSET.json")
    parser.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--ngram", type=int, default=3, help="tail length used to look up the prompt")
    parser.add_argument("--max_proposal", type=int, default=10, help="tokens proposed per hit")
    parser.add_argument("--ms_per_sweep", type=float, default=250.0)
    args = parser.parse_args()

    questions = json.load(open(args.qa_file))["data"]
    responses = json.load(open(args.result))["responses"]
    tokenizer = load_tokenizer(args.tokenizer)

    print(f"{'question':42s} {'tokens':>7s} {'sweeps':>7s} {'free':>6s} {'speedup':>8s}")
    print("-" * 76)
    total_tokens = total_sweeps = 0
    for question, response in zip(questions, responses):
        # The answer the model actually generated is the JSON object, not just its text
        # field -- the keys and punctuation are tokens it had to emit too, and the
        # prompt template spells them out, so they are exactly what lookup catches.
        generated = response.get("response")
        if generated is None:
            continue
        if not isinstance(generated, str):
            generated = json.dumps(generated, ensure_ascii=False)
        prompt_ids = tokenizer.encode(build_prompt(question["context"], question["question"]))
        answer_ids = tokenizer.encode(generated, add_special_tokens=False)
        sweeps, free = simulate(prompt_ids, answer_ids, args.ngram, args.max_proposal)
        total_tokens += len(answer_ids)
        total_sweeps += sweeps
        asked = [line.strip() for line in question["question"].strip().splitlines() if line.strip()][-1]
        print(f"{asked[:42]:42s} {len(answer_ids):>7d} {sweeps:>7d} {free:>6d} "
              f"{len(answer_ids)/sweeps:>7.2f}x")

    if not total_sweeps:
        print("\nno usable responses in that result file")
        return 1
    speedup = total_tokens / total_sweeps
    print(f"\ntokens {total_tokens}  sweeps {total_sweeps}  "
          f"free tokens {total_tokens - total_sweeps} "
          f"({100*(total_tokens-total_sweeps)/total_tokens:.0f}% of output)")
    print(f"overall speedup      : {speedup:.2f}x  (tokens per target sweep)")
    print(f"decode {args.ms_per_sweep:.0f} ms/sweep -> {args.ms_per_sweep/speedup:.0f} ms/token "
          f"({1e3/(args.ms_per_sweep/speedup):.1f} tok/s, was {1e3/args.ms_per_sweep:.1f})")
    print("\nThis is an upper bound on the win from lookup alone: it replays answers the\n"
          "greedy path already produced, so it assumes verification reproduces them\n"
          "exactly. At temperature 0 that holds; sampling would diverge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
