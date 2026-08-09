"""Does a non-square frame work on this stack, and does it caption better?

Qwen's `smart_resize` rounds height and width independently and has no square rule, and
nothing in the tt-metal port requires h == w. But every tt-metal demo, the warmup dummy
and every caption produced here so far went in square, so it has never been exercised.
Forcing a square also stretches a 1224x1024 frame, which is the opposite of what reading
a sign wants.

One run per shape over the same frames, so the shape is the only variable. At 576:

    stretch     576x576   1296 patches   distorted 1.2x taller
    letterbox   576x576   1296 patches   true shape, 16% black
    contain     576x480   1080 patches   true shape, nothing wasted

Patch count matters too: 1200 is the largest that has ever finished here.

The server decides the shape, so this needs a restart between runs:

    TT_VLM_IMAGE_LONG_SIDE=576 TT_VLM_IMAGE_FIT=contain ./tt_server/run_server.sh
    python tt_server/tests/board/probe_aspect_ratio.py --label contain

Also reads GET /memory around each request. Allocation that stays high after a caption
means the request did not give back what it took; allocation flat while the largest free
run shrinks is fragmentation, which refuses an allocation long before the bytes are gone.

Writes out/logs/aspect_<label>.json so runs can be compared without redoing them.
"""
import argparse
import base64
import glob
import io
import json
import os
import pickle
import re
import statistics
import sys
import time

import requests
from PIL import Image

BASE = os.environ.get("TT_SERVER_URL", "http://localhost:8100")
FRAMES = "/home/user/coda_raw/coda_data/0/*.pkl"
OUT = "/home/user/remembr/out/logs"

NAMED_BUILDING = r"\b(Gates|Bill and Melinda|Union|Library|Hall|Center|Centre)\b"

PROMPT = (
    "You are the camera on a robot moving through a university campus. "
    "Write what is in view as one entry in a search index.\n\n"
    "Rules:\n"
    "- Two or three plain sentences. Under 60 words total.\n"
    "- Name concrete things: what they are, what they are made of, where they are.\n"
    "- Copy any words visible on signs, doors or banners exactly, in quotes.\n"
    "- No greeting, no preamble, no markdown, no lists.\n"
    "- Start with the place. First word is a noun."
)


def frames(count):
    count = max(1, count)
    files = sorted(glob.glob(FRAMES))
    return files[:: max(1, len(files) // count)][:count]


def encoded(path):
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    image = Image.fromarray(data["cam0"][:, :, ::-1].astype("uint8"), "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def memory():
    """Device allocator state, or {} if the server is too old to report it."""
    try:
        return requests.get(f"{BASE}/memory", timeout=15).json()
    except Exception:  # noqa: BLE001
        return {}


def megabytes(view, field):
    dram = view.get("dram") or {}
    return dram.get(field, 0) / 1e6


def caption(body, gen):
    start = time.perf_counter()
    reply = requests.post(f"{BASE}/caption",
                          json={"images": [body], "prompt": PROMPT, "max_gen_len": gen},
                          timeout=3600)
    took = time.perf_counter() - start
    if reply.status_code != 200:
        return None, took, reply.text[:140]
    return reply.json()["caption"].strip(), took, None


def main(label, count, gen):
    health = requests.get(f"{BASE}/health", timeout=10).json()
    print(f"서버: mode={health.get('mode')} vlm={health.get('vlm')}")
    print(f"라벨: {label}   (서버의 TT_VLM_IMAGE_SQUARE / _LONG_SIDE 설정이 모양을 정한다)\n",
          flush=True)

    start_mem = memory()
    if start_mem:
        print(f"요청 전 DRAM 할당 {megabytes(start_mem,'allocated'):.0f}MB"
              f"  최대연속여유 {megabytes(start_mem,'largest_free'):.0f}MB\n", flush=True)

    rows = []
    for index, path in enumerate(frames(count), 1):
        text, took, error = caption(encoded(path), gen)
        if error:
            print(f"{index}  실패 {took:5.1f}s  {error}", flush=True)
            rows.append({"ok": False, "seconds": took, "error": error})
            continue
        after = memory()
        note = ""
        if after:
            note = (f"   DRAM 할당 {megabytes(after,'allocated'):.0f}MB"
                    f"  최대연속여유 {megabytes(after,'largest_free'):.0f}MB")
        print(f"{index}  [{took:5.1f}s, {len(text.split())}단어]{note}\n   {text}\n", flush=True)
        rows.append({"ok": True, "seconds": took, "caption": text, "memory": after})

    done = [r for r in rows if r["ok"]]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"aspect_{label}.json")
    with open(path, "w") as handle:
        json.dump({"label": label, "rows": rows}, handle, ensure_ascii=False, indent=1)

    print("=" * 70)
    print(f"{len(done)}/{len(rows)} 성공   → {path}")
    if not done:
        print("한 장도 못 만들었다. 이 모양은 이 스택에서 안 된다는 뜻이다.")
        return 1
    quote = chr(34)
    captions = [r["caption"] for r in done]
    print(f"캡션당 중앙값 {statistics.median(r['seconds'] for r in done):.1f}s")
    print(f"길이 중앙값 {statistics.median(len(c.split()) for c in captions):.0f}단어")
    print(f"표지판 글자 인용 {sum(1 for c in captions if quote in c)}/{len(captions)}")
    named = sum(1 for c in captions if re.search(NAMED_BUILDING, c))
    print(f"건물 고유명 {named}/{len(captions)}")

    end_mem = memory()
    if start_mem and end_mem:
        grew = megabytes(end_mem, "allocated") - megabytes(start_mem, "allocated")
        shrank = megabytes(start_mem, "largest_free") - megabytes(end_mem, "largest_free")
        print(f"\nDRAM 할당 변화 {grew:+.0f}MB   최대연속여유 변화 {-shrank:+.0f}MB")
        print("  할당이 늘어난 채면 요청이 가져간 것을 다 안 돌려준 것이고,")
        print("  할당은 그대로인데 연속 여유만 줄면 조각화다.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="square 또는 aspect")
    parser.add_argument("--frames", type=int, default=6)  # 1 이상
    parser.add_argument("--gen", type=int, default=128)
    args = parser.parse_args()
    sys.exit(main(args.label, args.frames, args.gen))
