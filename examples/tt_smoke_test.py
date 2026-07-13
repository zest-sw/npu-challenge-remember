"""End-to-end smoke test for ReMEmbR on Tenstorrent NPU.

Prereqs (see plan):
  1. tt_infer_server.py running (LLM+VLM) on http://localhost:8100
  2. Milvus running (docker) on 127.0.0.1:19530
  3. This runs in the *remembr* venv (langchain, pymilvus, sentence-transformers).

Flow:  images -> VLM caption -> MemoryItem -> MilvusMemory -> NonAgent(LLM) query

Usage:
  python examples/tt_smoke_test.py img1.jpg img2.jpg ...
  (no args -> a couple of synthetic solid-color frames, just to exercise the path)
"""
import sys
import time

from PIL import Image

from remembr.captioners.tenstorrent_captioner import TenstorrentCaptioner
from remembr.memory.memory import MemoryItem
from remembr.memory.milvus_memory import MilvusMemory
from remembr.agents.non_agent import NonAgent


def load_frames(paths):
    if paths:
        return [Image.open(p).convert("RGB") for p in paths]
    # fallback: synthetic frames so the pipeline still runs without real images
    return [Image.new("RGB", (224, 224), color=c) for c in ((120, 40, 40), (40, 120, 40))]


def main():
    frames = load_frames(sys.argv[1:])

    import os
    question = os.environ.get("TT_SMOKE_QUESTION", "did you see pasta?")

    print("[1/4] Setting up memory (fresh collection)...")
    memory = MilvusMemory("tt_smoke_test", db_ip="127.0.0.1")
    # The collection persists across runs and MilvusMemory never drops it on
    # construction, so stale captions from previous runs would pollute the
    # retrieval (and the LLM would answer from them). Start each smoke run clean.
    memory.reset(drop_collection=True)

    # Caption EACH frame separately and store it as its own timestamped memory.
    # This is how ReMEmbR is meant to work: one caption per frame/window, so the
    # vector search can later retrieve the *relevant* frame for a question.
    # (Lumping unrelated frames into a single caption gives the VLM one merged
    # scene to hallucinate over and leaves nothing meaningful to retrieve.)
    captioner = TenstorrentCaptioner()
    base_t = time.time()
    print(f"[2/4] Captioning {len(frames)} frame(s) on the NPU (Qwen2.5-VL) + inserting...")
    for i, frame in enumerate(frames):
        caption = captioner.caption([frame])          # one frame -> one caption
        print(f"      frame {i}: {caption[:160]}")
        memory.insert(MemoryItem(
            caption=caption,
            time=base_t + i * 10.0,                    # stagger frames 10s apart
            position=[float(i), 0.0, 0.0],             # and along the x-axis
            theta=0.0,
        ))

    print("[3/4] Building NonAgent (Llama-3.2-3B on the NPU)...")
    agent = NonAgent(llm_type="tt/Llama-3.2-3B-Instruct")
    agent.set_memory(memory)

    print(f"[4/4] Querying: {question!r}")
    # NonAgent uses memory.get_working_memory() as context, which is populated by
    # the search_* methods (not by insert). Retrieve relevant memories first.
    ctx = memory.search_by_text(question)
    print("      retrieved context:")
    for line in ctx.splitlines():
        if line.strip():
            print("        " + line)
    out = agent.query(question)
    print("      AgentOutput:", out)
    print("\nOK: end-to-end path completed.")


if __name__ == "__main__":
    main()
