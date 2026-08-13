"""Caption a spread of CODa frames and count what the questions actually need.

The dataset asks for named things -- "the Bill and Melinda Gates Computer Science
building", "the yellow police call pole". Of the 272 captions on disk, 33 named a
building and 1 mentioned the call pole. Retrieval cannot find what was never written
down, so this counts proper nouns and quoted sign text, not just whether captions look
tidy.

Also watches the vision block loop while it waits. A caption that has not come back is
either progressing slowly or stuck inside a ttnn call holding the GIL, and from outside
those look identical -- `/health` stops answering either way (#62). The loop variable
tells them apart.

Finishes with the determinism check from #68: the sampler is greedy, so the same frame
must give the same sentence. If it does not, buffers allocated beside the live decode
trace are being corrupted.

    python tt_server/tests/board/caption_sample.py
    python tt_server/tests/board/caption_sample.py --frames 4 --gen 96

Read-only against the server: it sends captions and samples stacks, never signals it.
"""
import argparse
import base64
import glob
import io
import os
import pickle
import re
import statistics
import subprocess
import sys
import threading
import time

import requests
from PIL import Image

BASE = os.environ.get("TT_SERVER_URL", "http://localhost:8100")
FRAMES = "/home/user/coda_raw/coda_data/0/*.pkl"
PY_SPY = "/home/user/remembr/.venv/bin/py-spy"

# No worked example: a campus one would name objects that really are in this dataset,
# and this model copies example values verbatim (#27).
PROMPT = (
    "You are the camera on a robot moving through a university campus. "
    "Write what is in view as one entry in a search index.\n\n"
    "Rules:\n"
    "- Two or three plain sentences. Under 60 words total.\n"
    "- Name concrete things: what they are, what they are made of, where they are.\n"
    "- Say how many people and what they are doing, or say there are none.\n"
    "- Copy any words visible on signs, doors or banners exactly, in quotes.\n"
    "- No greeting, no preamble, no markdown, no headings, no lists.\n"
    '- Do not write "the image", "appears to be", "likely", or anything about yourself.\n'
    "- Start with the place. First word is a noun."
)

FORMAT_RULES = {
    "어시스턴트 머리말": r"^\s*(certainly|sure|of course|here|let)",
    "마크다운": r"\*\*|^\s*[-*]\s|^\s*\d+\.\s",
    '"the image" 류': r"\bthe (image|photo|picture)\b",
    "헤지 표현": r"\b(appears? to be|likely|probably)\b",
}
NAMED_BUILDING = r"\b(Gates|Bill and Melinda|Union|Library|Hall|Center|Centre)\b"
SIGN_WORDS = r"\bsigns?\b|\breads?\b|\blabel"


def server_pid():
    found = subprocess.run(["pgrep", "-f", "tt_infer_server"],
                           capture_output=True, text=True).stdout.split()
    return found[0] if found else None


def watch(stop, pid):
    """Print the vision block index while a caption is outstanding."""
    if not pid or not os.path.exists(PY_SPY):
        return
    last, repeats = None, 0
    while not stop.wait(45):
        try:
            dump = subprocess.run([PY_SPY, "dump", "--pid", pid, "--nonblocking", "--locals"],
                                  capture_output=True, text=True, timeout=60).stdout
        except Exception:  # noqa: BLE001
            continue
        loop = next((line.strip() for line in dump.splitlines() if "layer_num:" in line), None)
        if not loop:
            continue
        repeats = repeats + 1 if loop == last else 0
        note = "  <- 그대로, 멈춤 의심" if repeats >= 2 else ""
        print(f"      [{time.strftime('%H:%M:%S')}] {loop}{note}", flush=True)
        last = loop


def encoded(path):
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    image = Image.fromarray(data["cam0"][:, :, ::-1].astype("uint8"), "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode(), image.size


def caption(body, gen):
    start = time.perf_counter()
    reply = requests.post(f"{BASE}/caption",
                          json={"images": [body], "prompt": PROMPT, "max_gen_len": gen},
                          timeout=7200)
    took = time.perf_counter() - start
    if reply.status_code != 200:
        return None, took, reply.text[:140]
    return reply.json()["caption"].strip(), took, None


def report(captions, latencies):
    print("=" * 72)
    median = statistics.median(latencies)
    print(f"캡션 {len(captions)}장   캡션당 중앙값 {median:.1f}s"
          f"   272개 환산 {median * 272 / 3600:.1f}시간")
    print(f"길이 중앙값 {statistics.median(len(c.split()) for c in captions):.0f}단어")

    print("\n형식 위반:")
    for name, pattern in FORMAT_RULES.items():
        hit = sum(1 for c in captions if re.search(pattern, c, re.I | re.M))
        print(f"  {name:18s} {hit}/{len(captions)}")

    quote = chr(34)
    print("\n질문이 실제로 요구하는 것:")
    print(f"  표지판 글자 인용    {sum(1 for c in captions if quote in c)}/{len(captions)}")
    print(f"  건물 고유명        {sum(1 for c in captions if re.search(NAMED_BUILDING, c))}/{len(captions)}")
    print(f"  sign/reads 언급    {sum(1 for c in captions if re.search(SIGN_WORDS, c, re.I))}/{len(captions)}")
    print("  (224x224 에서는 인용 0/8, 고유명 0/8 이었다)")


def main(count, gen):
    health = requests.get(f"{BASE}/health", timeout=10).json()
    print(f"서버: mode={health.get('mode')} vlm={health.get('vlm')}", flush=True)
    if not health.get("vlm"):
        sys.exit("VLM 이 올라와 있지 않다.")

    files = sorted(glob.glob(FRAMES))
    picks = files[:: max(1, len(files) // count)][:count]
    print(f"프레임 {len(files)}개 중 {len(picks)}장을 고르게 뽑는다\n", flush=True)

    stop = threading.Event()
    threading.Thread(target=watch, args=(stop, server_pid()), daemon=True).start()

    captions, latencies, sent_size = [], [], None
    for index, path in enumerate(picks, 1):
        body, sent_size = encoded(path)
        text, took, error = caption(body, gen)
        if error:
            print(f"{index}  실패 {took:.1f}s  {error}", flush=True)
            continue
        captions.append(text)
        latencies.append(took)
        print(f"{index}  [{took:5.1f}s, {len(text.split())}단어]  {text}\n", flush=True)
    stop.set()

    if not captions:
        print("캡션이 한 장도 나오지 않았다.")
        return 1

    print("=" * 72)
    print("결정성 시험 (#68): 1번 프레임을 다시. 샘플러가 greedy 이므로 같아야 한다.\n", flush=True)
    body, _ = encoded(picks[0])
    again, _, error = caption(body, gen)
    if error:
        print(f"  재시도 실패: {error}\n")
    else:
        print(f"  1회차: {captions[0][:110]}")
        print(f"  2회차: {again[:110]}")
        print(f"  판정: {'동일 — 이 경로에서는 오염 없음' if again == captions[0] else '다름 — 오염 의심'}\n")

    print(f"클라이언트가 보낸 크기: {sent_size} (서버가 TT_VLM_IMAGE_LONG_SIDE 로 줄인다)\n")
    report(captions, latencies)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--gen", type=int, default=128)
    args = parser.parse_args()
    sys.exit(main(args.frames, args.gen))
