"""Build a static HTML viewer that pairs each caption with the frame it describes.

Captioning is the one stage of the pipeline whose output nothing downstream can
check: a wrong caption is a fluent, plausible sentence, and it silently becomes
the memory the agent reasons over. This renders caption and frame side by side
so the failure is visible.

Each caption covers a window of frames (file_start..file_end). The captioner
feeds the VLM `--num-video-frames` frames sampled from that window, resized to
224x224 by the server. This dumps the window's first frame at both display
resolution and at the 224x224 the model actually saw, so downscale-induced
errors can be told apart from model errors.

Usage (remembr venv -- needs cv2 + numpy):

    python scripts/caption_viewer.py \
        --caption_file captions_tt-qwen25-vl-3b_3_secs \
        --coda_dir /home/user/coda_raw/coda_data \
        --out /tmp/caption_viewer

    python -m http.server -d /tmp/caption_viewer 8200

With --inline, the frames are embedded as data URIs and the result is a single
self-contained .html -- no image directory and no local web server, so it can be
opened over a remote session or handed to someone else.
"""

import argparse
import base64
import glob
import html
import json
import os
import pickle as pkl

import cv2
import numpy as np


# The server resizes every image to this before the vision encoder sees it, so
# the "as the VLM saw it" pane must use the same size. Keep in sync with
# tt_server/tt_infer_server.py.
VLM_INPUT_SIZE = (224, 224)


def load_frame_rgb(pkl_path: str) -> np.ndarray:
    """Load a CODa frame as RGB.

    The .pkl holds cam0 in BGR (preprocess_coda_rosfree.py wrote it with
    cv2.imread), and preprocess_captions.py flips it to RGB before sending it to
    the VLM. Do the same flip so this shows what the model was given.
    """
    with open(pkl_path, "rb") as f:
        data = pkl.load(f)
    return data["cam0"][:, :, ::-1]


def write_jpeg(path: str, rgb: np.ndarray, quality: int) -> None:
    cv2.imwrite(path, rgb[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def data_uri(rgb: np.ndarray, quality: int) -> str:
    ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def build(args) -> None:
    captions_path = os.path.join(
        args.data_dir, "captions", str(args.sequence_id), "captions", f"{args.caption_file}.json"
    )
    with open(captions_path, "r") as f:
        captions = json.load(f)

    seq_dir = os.path.join(args.coda_dir, str(args.sequence_id))
    pkl_files = sorted(glob.glob(os.path.join(seq_dir, "*.pkl")), key=lambda x: float(os.path.basename(x)[:-4]))

    if args.inline:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        frames_dir = None
    else:
        frames_dir = os.path.join(args.out, "frames")
        os.makedirs(frames_dir, exist_ok=True)

    idxs = list(range(0, len(captions), args.stride))
    if args.limit:
        idxs = idxs[: args.limit]

    items = []
    for n, i in enumerate(idxs):
        c = captions[i]
        start_path = os.path.join(seq_dir, c["file_start"])
        if not os.path.exists(start_path):
            print(f"[{i}] missing frame {c['file_start']}, skipping")
            continue

        rgb = load_frame_rgb(start_path)

        # Display copy: keep the aspect ratio, cap the width so the page stays light.
        h, w = rgb.shape[:2]
        if w > args.width:
            scale = args.width / w
            disp = cv2.resize(rgb, (args.width, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            disp = rgb
        # Exactly what the vision encoder received.
        vlm = cv2.resize(rgb, VLM_INPUT_SIZE, interpolation=cv2.INTER_AREA)

        if args.inline:
            img_src = data_uri(disp, args.quality)
            vlm_src = data_uri(vlm, args.quality)
        else:
            write_jpeg(os.path.join(frames_dir, f"{i:04d}.jpg"), disp, args.quality)
            write_jpeg(os.path.join(frames_dir, f"{i:04d}_vlm.jpg"), vlm, 92)
            img_src = f"frames/{i:04d}.jpg"
            vlm_src = f"frames/{i:04d}_vlm.jpg"

        # How many raw frames this caption's window spans -- a caption that
        # covers many frames but saw only one of them is a known blind spot.
        try:
            span = pkl_files.index(os.path.join(seq_dir, c["file_end"])) - pkl_files.index(start_path)
        except ValueError:
            span = -1

        items.append(
            {
                "i": i,
                "img": img_src,
                "vlm": vlm_src,
                "caption": c["caption"],
                "position": [round(float(p), 3) for p in c["position"]],
                "time": round(float(c["time"]), 2),
                "theta": c["theta"],
                "file_start": c["file_start"],
                "file_end": c["file_end"],
                "span": span,
                "res": f"{w}x{h}",
            }
        )
        if n % 25 == 0:
            print(f"  {n}/{len(idxs)} frames written")

    t0 = min(it["time"] for it in items) if items else 0
    for it in items:
        it["t_rel"] = round(it["time"] - t0, 1)

    out_html = args.out if args.inline else os.path.join(args.out, "index.html")
    with open(out_html, "w") as f:
        f.write(PAGE.replace("__DATA__", json.dumps(items)).replace("__TITLE__", html.escape(args.caption_file)))

    size_mb = os.path.getsize(out_html) / 1e6
    print(f"\n{len(items)} captions -> {out_html} ({size_mb:.1f} MB)")
    if not args.inline:
        print(f"View with:  python -m http.server -d {args.out} 8200")


PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Caption check - __TITLE__</title>
<style>
  /* Deliberately dark-only: this is an image-judging tool, and a bright ground
     skews how exposure and detail in the frames read. Same reason photo/video
     review UIs commit to dark. */
  :root {
    --bg: #0e1013; --panel: #171a20; --edge: #262b34;
    --fg: #e8ebf0; --dim: #8b93a3;
    --accent: #6ea8fe; --warn: #ffb454; --good: #5ec98a;
    --mono: ui-monospace, SFMono-Regular, Menlo, "Cascadia Mono", monospace;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif;
         background: var(--bg); color: var(--fg); }
  header { padding: 10px 16px; background: var(--panel); border-bottom: 1px solid var(--edge);
           display: flex; gap: 14px; align-items: center; flex-wrap: wrap;
           position: sticky; top: 0; z-index: 10; }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; }
  header .sub { color: var(--dim); font-size: 12px; }
  /* Findings before detail: the truncation rate is a property of the whole run,
     not something you should have to page through 272 frames to notice. */
  .stats { display: flex; gap: 0; background: var(--panel);
           border-bottom: 1px solid var(--edge); padding: 0 16px; flex-wrap: wrap; }
  .stat { padding: 9px 20px 9px 0; margin-right: 20px; border-right: 1px solid var(--edge);
          display: flex; align-items: baseline; gap: 7px; }
  .stat:last-child { border-right: 0; }
  .stat .v { font: 600 15px/1 var(--mono); font-variant-numeric: tabular-nums; }
  .stat .l { font-size: 11px; color: var(--dim); text-transform: uppercase; letter-spacing: .05em; }
  .stat.bad .v { color: var(--warn); }
  .stat.ok .v { color: var(--good); }
  .chip { font: 600 9px/1 var(--mono); padding: 2px 4px; border-radius: 3px;
          background: var(--warn); color: #0e1013; letter-spacing: .03em; }
  input[type=search] { background: var(--bg); border: 1px solid var(--edge); color: var(--fg);
                       padding: 6px 10px; border-radius: 6px; width: 260px; font-size: 13px; }
  button { background: var(--bg); border: 1px solid var(--edge); color: var(--fg);
           padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 13px; }
  button:hover { border-color: var(--accent); }
  .wrap { display: grid; grid-template-columns: 190px 1fr; height: calc(100vh - 53px); }
  .strip { overflow-y: auto; border-right: 1px solid var(--edge); background: var(--panel); }
  .thumb { padding: 6px; border-bottom: 1px solid var(--edge); cursor: pointer; display: flex; gap: 8px; }
  .thumb:hover { background: #232833; }
  .thumb.on { background: #2b3648; border-left: 3px solid var(--accent); padding-left: 3px; }
  .thumb img { width: 74px; height: 56px; object-fit: cover; border-radius: 3px; flex-shrink: 0; }
  .thumb .m { font-size: 11px; color: var(--dim); overflow: hidden; font-family: var(--mono);
              font-variant-numeric: tabular-nums; }
  .thumb .m b { color: var(--fg); font-size: 12px; }
  .detail { overflow-y: auto; padding: 18px 22px; }
  .panes { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; margin-bottom: 16px; }
  .pane { background: var(--panel); border: 1px solid var(--edge); border-radius: 8px; padding: 10px; }
  .pane h3 { margin: 0 0 8px; font-size: 12px; color: var(--dim); font-weight: 600;
             text-transform: uppercase; letter-spacing: .04em; }
  .pane img { display: block; border-radius: 4px; max-width: 100%; }
  .pane.big img { max-width: min(640px, 60vw); }
  .pane.vlm img { width: 224px; height: 224px; image-rendering: pixelated; }
  .meta { display: flex; gap: 18px; flex-wrap: wrap; font-size: 12px; color: var(--dim);
          margin-bottom: 14px; align-items: center; }
  .meta b { color: var(--fg); font-weight: 500; font-family: var(--mono);
            font-variant-numeric: tabular-nums; }
  .cap { background: var(--panel); border: 1px solid var(--edge); border-radius: 8px;
         padding: 16px 18px; white-space: pre-wrap; max-width: 900px; }
  .cap mark { background: var(--warn); color: #14161a; border-radius: 2px; padding: 0 2px; }
  .trunc { color: var(--warn); font-size: 12px; margin-top: 10px; }
  kbd { background: var(--bg); border: 1px solid var(--edge); border-radius: 3px;
        padding: 1px 5px; font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>Caption check</h1>
  <span class="sub" id="sub"></span>
  <input type="search" id="q" placeholder="search captions (e.g. bench)">
  <span class="sub" id="hits"></span>
  <button id="prev">&larr; Prev</button>
  <button id="next">Next &rarr;</button>
  <span class="sub"><kbd>j</kbd>/<kbd>k</kbd> or arrows</span>
</header>
<div class="stats" id="stats"></div>
<div class="wrap">
  <div class="strip" id="strip"></div>
  <div class="detail" id="detail"></div>
</div>
<script>
const DATA = __DATA__;
let view = DATA.slice(), cur = 0, term = "";

const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const hl = s => term ? esc(s).replace(new RegExp("("+term.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")+")","gi"), "<mark>$1</mark>") : esc(s);

// A caption that does not end on terminal punctuation ran into the generation
// cap. It still gets embedded and stored as memory, so it matters as much as a
// wrong caption -- flag it everywhere rather than leaving it to be noticed.
const isCut = d => !/[.!?"]\s*$/.test(d.caption.trim());

function stats() {
  const cut = DATA.filter(isCut).length;
  const pct = Math.round(100 * cut / DATA.length);
  const chars = Math.round(DATA.reduce((a, d) => a + d.caption.length, 0) / DATA.length);
  const cells = [
    {v: DATA.length, l: "captions"},
    {v: pct + "%", l: "cut off mid-sentence", cls: pct > 20 ? "bad" : "ok"},
    {v: chars, l: "avg chars"},
    {v: "+" + Math.round(Math.max(...DATA.map(d => d.t_rel))) + "s", l: "span"},
  ];
  document.getElementById("stats").innerHTML = cells.map(c =>
    `<div class="stat ${c.cls || ''}"><span class="v">${c.v}</span><span class="l">${c.l}</span></div>`).join("");
}

function strip() {
  document.getElementById("strip").innerHTML = view.map((d, n) => `
    <div class="thumb ${n===cur?'on':''}" data-n="${n}">
      <img src="${d.img}" loading="lazy">
      <div class="m"><b>#${d.i}</b> ${isCut(d) ? '<span class="chip">CUT</span>' : ''}<br>t+${d.t_rel}s<br>${d.position[0]}, ${d.position[1]}</div>
    </div>`).join("");
  document.querySelectorAll(".thumb").forEach(el =>
    el.onclick = () => { cur = +el.dataset.n; render(); });
}

function render() {
  const d = view[cur];
  if (!d) { document.getElementById("detail").innerHTML = "<p>No captions match.</p>"; return; }
  const cut = isCut(d);
  document.getElementById("detail").innerHTML = `
    <div class="meta">
      <span>caption <b>#${d.i}</b> of ${DATA.length}</span>
      <span>t <b>+${d.t_rel}s</b></span>
      <span>position <b>[${d.position.join(", ")}]</b></span>
      <span>theta <b>${d.theta}</b></span>
      <span>window <b>${d.span} frames</b></span>
      <span>source <b>${d.file_start}</b></span>
      ${cut ? '<span class="chip">CUT OFF</span>' : ''}
    </div>
    <div class="panes">
      <div class="pane big"><h3>Frame (${d.res}, first of window)</h3><img src="${d.img}"></div>
      <div class="pane vlm"><h3>What the VLM saw (224&times;224)</h3><img src="${d.vlm}"></div>
    </div>
    <div class="cap">${hl(d.caption)}${cut ? '<div class="trunc">&#9888; ends mid-sentence &mdash; hit the generation cap</div>' : ''}</div>`;
  strip();
  const on = document.querySelector(".thumb.on");
  if (on) on.scrollIntoView({block: "nearest"});
  document.getElementById("sub").textContent = `__TITLE__ — ${DATA.length} captions`;
}

function filter() {
  term = document.getElementById("q").value.trim();
  view = term ? DATA.filter(d => d.caption.toLowerCase().includes(term.toLowerCase())) : DATA.slice();
  cur = 0;
  document.getElementById("hits").textContent = term ? `${view.length} / ${DATA.length} match` : "";
  render();
}

document.getElementById("q").oninput = filter;
document.getElementById("prev").onclick = () => { cur = Math.max(0, cur-1); render(); };
document.getElementById("next").onclick = () => { cur = Math.min(view.length-1, cur+1); render(); };
document.onkeydown = e => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "j" || e.key === "ArrowDown" || e.key === "ArrowRight") { cur = Math.min(view.length-1, cur+1); render(); }
  if (e.key === "k" || e.key === "ArrowUp" || e.key === "ArrowLeft") { cur = Math.max(0, cur-1); render(); }
};
stats();
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="caption_viewer",
        description="Render captions next to the frames they describe, as a static HTML page.",
    )
    parser.add_argument("--sequence_id", type=int, default=0)
    parser.add_argument("--caption_file", type=str, default="captions_tt-qwen25-vl-3b_3_secs")
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--coda_dir", type=str, default="./coda_data/")
    parser.add_argument("--out", type=str, default="/tmp/caption_viewer",
                        help="output dir, or the .html file itself when --inline")
    parser.add_argument("--inline", action="store_true",
                        help="embed frames as data URIs -> one self-contained .html (no server needed)")
    parser.add_argument("--width", type=int, default=640, help="display JPEG width")
    parser.add_argument("--quality", type=int, default=80, help="display JPEG quality")
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth caption")
    parser.add_argument("--limit", type=int, default=0, help="cap the number of captions (0 = all)")
    build(parser.parse_args())
