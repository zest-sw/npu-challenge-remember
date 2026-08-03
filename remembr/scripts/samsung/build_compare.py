import json

frames = json.load(open('/home/user/remembr/out/caption_compare_4/compare_data.json'))
avg3 = round(sum(f['t3'] for f in frames)/len(frames), 1)
avg8 = round(sum(f['t8'] for f in frames)/len(frames), 1)
ratio = round(avg8/avg3, 1)
halluc = sum(1 for f in frames if f['tag3'] == 'hallucination' or f['tag8'] == 'hallucination')

TAG_LABEL = {"accurate": "정확", "partial": "일부 오판", "hallucination": "환각"}
TAG_CLASS = {"accurate": "ok", "partial": "warn", "hallucination": "bad"}

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def card(f):
    return f'''
<section class="frame">
  <div class="frame-head">
    <span class="fnum">frame #{f['idx']}</span>
  </div>
  <div class="frame-body">
    <img class="shot" src="{f['img']}" alt="frame {f['idx']}">
    <div class="caps">
      <div class="cap cap-3b">
        <div class="cap-head">
          <span class="model-chip m3">3B</span>
          <span class="time">{f['t3']}s</span>
          <span class="tag {TAG_CLASS[f['tag3']]}">{TAG_LABEL[f['tag3']]}</span>
        </div>
        <p class="cap-text">{esc(f['cap3'])}</p>
      </div>
      <div class="cap cap-8b">
        <div class="cap-head">
          <span class="model-chip m8">8B</span>
          <span class="time">{f['t8']}s</span>
          <span class="tag {TAG_CLASS[f['tag8']]}">{TAG_LABEL[f['tag8']]}</span>
        </div>
        <p class="cap-text">{esc(f['cap8'])}</p>
      </div>
    </div>
  </div>
  <div class="note">{esc(f['note'])}</div>
</section>'''

cards = "\n".join(card(f) for f in frames)

html = f'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Qwen2.5-VL-3B vs Qwen3-VL-8B — 캡셔닝 비교</title>
<style>
:root {{
  --bg:#0e1013; --panel:#171a20; --edge:#262b34; --fg:#e8ebf0; --dim:#8b93a3;
  --m3:#6ea8fe; --m8:#ffb454; --ok:#5ec98a; --warn:#e8b339; --bad:#e06c75;
  --mono: ui-monospace, SFMono-Regular, Menlo, "Cascadia Mono", monospace;
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
       font: 14px/1.6 -apple-system, "Segoe UI", Roboto, sans-serif; }}
header {{ padding: 22px 28px 18px; background: var(--panel); border-bottom: 1px solid var(--edge); }}
h1 {{ margin: 0 0 4px; font-size: 19px; font-weight: 650; text-wrap: balance; }}
.sub {{ color: var(--dim); font-size: 13px; }}
.stats {{ display: flex; gap: 0; margin-top: 16px; flex-wrap: wrap; }}
.stat {{ padding: 0 22px 0 0; margin-right: 22px; border-right: 1px solid var(--edge); }}
.stat:last-child {{ border-right: none; }}
.stat .v {{ font: 650 20px/1 var(--mono); font-variant-numeric: tabular-nums; }}
.stat .l {{ font-size: 11px; color: var(--dim); text-transform: uppercase; letter-spacing: .04em; margin-top: 3px; }}
.stat.m3 .v {{ color: var(--m3); }}
.stat.m8 .v {{ color: var(--m8); }}
.stat.bad .v {{ color: var(--bad); }}
main {{ max-width: 1180px; margin: 0 auto; padding: 22px 28px 60px; display: flex; flex-direction: column; gap: 22px; }}
.frame {{ background: var(--panel); border: 1px solid var(--edge); border-radius: 10px; overflow: hidden; }}
.frame-head {{ padding: 10px 16px; border-bottom: 1px solid var(--edge); }}
.fnum {{ font: 650 12px var(--mono); color: var(--dim); letter-spacing: .03em; }}
.frame-body {{ display: grid; grid-template-columns: 300px 1fr; gap: 0; }}
.shot {{ width: 100%; height: 100%; min-height: 220px; object-fit: cover; display: block; }}
.caps {{ display: grid; grid-template-columns: 1fr 1fr; }}
.cap {{ padding: 14px 16px; border-left: 1px solid var(--edge); display: flex; flex-direction: column; gap: 8px; }}
.cap:first-child {{ border-left: none; }}
.cap-head {{ display: flex; align-items: center; gap: 8px; }}
.model-chip {{ font: 700 11px var(--mono); padding: 2px 7px; border-radius: 4px; letter-spacing: .02em; }}
.m3 {{ background: rgba(110,168,254,.15); color: var(--m3); }}
.m8 {{ background: rgba(255,180,84,.15); color: var(--m8); }}
.time {{ font: 500 12px var(--mono); color: var(--dim); font-variant-numeric: tabular-nums; }}
.tag {{ margin-left: auto; font: 650 10px var(--mono); padding: 2px 7px; border-radius: 999px; letter-spacing: .02em; }}
.tag.ok {{ background: rgba(94,201,138,.15); color: var(--ok); }}
.tag.warn {{ background: rgba(232,179,57,.15); color: var(--warn); }}
.tag.bad {{ background: rgba(224,108,117,.15); color: var(--bad); }}
.cap-text {{ margin: 0; font-size: 13px; line-height: 1.65; color: var(--fg); white-space: pre-wrap; }}
.note {{ padding: 10px 16px; border-top: 1px solid var(--edge); font-size: 12px; color: var(--dim); background: rgba(255,255,255,.02); }}
@media (max-width: 720px) {{
  .frame-body {{ grid-template-columns: 1fr; }}
  .caps {{ grid-template-columns: 1fr; }}
  .cap {{ border-left: none; border-top: 1px solid var(--edge); }}
  .cap:first-child {{ border-top: none; }}
}}
</style>
</head>
<body>
<header>
  <h1>Qwen2.5-VL-3B vs Qwen3-VL-8B — 캡셔닝 비교</h1>
  <div class="sub">samsung_dataset 시퀀스 0에서 4프레임, 동일 프롬프트·동일 max_gen_len=120으로 직접 대조</div>
  <div class="stats">
    <div class="stat m3"><div class="v">{avg3}s</div><div class="l">3B 평균</div></div>
    <div class="stat m8"><div class="v">{avg8}s</div><div class="l">8B 평균</div></div>
    <div class="stat"><div class="v">{ratio}&times;</div><div class="l">8B가 더 느림</div></div>
    <div class="stat bad"><div class="v">{halluc}/4</div><div class="l">환각 발생 프레임</div></div>
  </div>
</header>
<main>
{cards}
</main>
</body>
</html>'''

open('/home/user/remembr/out/caption_compare_4/compare.html', 'w').write(html)
print("written /home/user/remembr/out/caption_compare_4/compare.html", len(html), "bytes")
