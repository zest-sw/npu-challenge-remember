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

    print(f"[1/4] Captioning {len(frames)} frame(s) on the NPU (Qwen2.5-VL)...")
    captioner = TenstorrentCaptioner()
    caption = captioner.caption(frames)
    print("      caption:", caption)

    print("[2/4] Inserting into Milvus memory...")
    memory = MilvusMemory("tt_smoke_test", db_ip="127.0.0.1")
    memory.insert(MemoryItem(caption=caption, time=time.time(), position=[0.0, 0.0, 0.0], theta=0.0))

    print("[3/4] Building NonAgent (Llama-3.2-3B on the NPU)...")
    agent = NonAgent(llm_type="tt/Llama-3.2-3B-Instruct")
    agent.set_memory(memory)

    print("[4/4] Querying...")
    out = agent.query("What did you see?")
    print("      AgentOutput:", out)
    print("\nOK: end-to-end path completed.")


if __name__ == "__main__":
    main()
