"""Repeated mode-switch stress test against the running tt_infer_server.

For each mode in a repeated cycle:
  1. POST /mode  -> switch (minutes; close+reopen + reload)
  2. verify the mode's endpoints WORK, and the absent endpoint returns 409
  3. confirm the server + board are still alive

If any switch wedges the board, /mode raises (FW init) and we stop with the cycle.
Goal: many switches, zero server/board death.
"""
import base64
import glob
import json
import pickle
import sys
import time
import urllib.request

import cv2

BASE = "http://localhost:8100"
# modes testable with weights on disk (qwen3-8B not downloaded -> caption-8b skipped)
CYCLE = ["3b+3b", "llm-8b", "caption-3b", "3b+3b", "caption-3b", "llm-8b"]

_f = sorted(glob.glob("/home/user/remembr/remembr/data/samsung_dataset/0/*.pkl"))[0]
_ok, _buf = cv2.imencode(".jpg", pickle.load(open(_f, "rb"))["cam0"])
IMG_B64 = base64.b64encode(_buf).decode()


def call(method, path, obj=None, timeout=1800):
    url = BASE + path
    data = json.dumps(obj).encode() if obj is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def check_mode(mode):
    """Verify endpoints match the mode: present ones work, absent ones give 409."""
    has_llm = mode in ("3b+3b", "llm-8b")
    has_vlm = mode in ("3b+3b", "caption-3b")

    s, r = call("POST", "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "Reply: ok"}], "max_tokens": 8})
    if has_llm:
        assert s == 200, f"LLM expected 200, got {s}: {r}"
        print(f"    LLM  200 -> {r['choices'][0]['message']['content'][:30]!r}")
    else:
        assert s == 409, f"LLM expected 409 (no LLM in {mode}), got {s}"
        print(f"    LLM  409 (correctly rejected)")

    s, r = call("POST", "/caption",
                {"images": [IMG_B64], "prompt": "Describe briefly.", "max_gen_len": 24})
    if has_vlm:
        assert s == 200, f"VLM expected 200, got {s}: {r}"
        print(f"    VLM  200 -> {r['caption'][:30]!r}")
    else:
        assert s == 409, f"VLM expected 409 (no VLM in {mode}), got {s}"
        print(f"    VLM  409 (correctly rejected)")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(CYCLE)
    print(f"=== mode-switch stress: {n} switches ===", flush=True)
    for i in range(n):
        mode = CYCLE[i % len(CYCLE)]
        t0 = time.time()
        s, r = call("POST", "/mode", {"mode": mode})
        if s != 200:
            print(f"[{i+1}/{n}] switch -> {mode} FAILED (HTTP {s}): {r}", flush=True)
            sys.exit(1)
        dt = time.time() - t0
        print(f"[{i+1}/{n}] switched -> {mode}  ({dt:.0f}s, switched={r.get('switched')})", flush=True)
        try:
            check_mode(mode)
        except AssertionError as e:
            print(f"    ENDPOINT CHECK FAILED: {e}", flush=True)
            sys.exit(1)
        # board/server alive?
        s, h = call("GET", "/health")
        assert s == 200 and h["ready"], f"health not OK: {h}"
        print(f"    health OK (mode={h['mode']})", flush=True)

    print(f"\nPASSED -- {n} mode switches, server+board alive, all endpoints correct.", flush=True)


if __name__ == "__main__":
    main()
