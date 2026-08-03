import json, base64

r3 = json.load(open('/home/user/remembr/out/caption_compare_15/3b_results.json'))
r8 = json.load(open('/home/user/remembr/out/caption_compare_15/8b_results.json'))
judgments = json.load(open('/home/user/remembr/out/caption_compare_15/judgments.json'))

keys = sorted(r3, key=int)
avg3 = round(sum(r3[k]['elapsed'] for k in keys)/len(keys), 1)
avg8 = round(sum(r8[k]['elapsed'] for k in keys)/len(keys), 1)
ratio = round(avg8/avg3, 1)

tally3 = {"accurate":0,"partial":0,"hallucination":0}
tally8 = {"accurate":0,"partial":0,"hallucination":0}
for k in keys:
    tally3[judgments[k]['t3']] += 1
    tally8[judgments[k]['t8']] += 1

TAG_LABEL = {"accurate": "정확", "partial": "일부 오판", "hallucination": "환각"}
TAG_CLASS = {"accurate": "ok", "partial": "warn", "hallucination": "bad"}

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def b64img(path):
    return "data:image/jpeg;base64," + base64.b64encode(open(path, 'rb').read()).decode()

def card(n, k):
    j = judgments[k]
    img = b64img(f"/home/user/remembr/out/caption_compare_15/frame_{k}.jpg")
    return f'''
<section class="frame">
  <div class="frame-head"><span class="fnum">#{n+1} · frame {r3[k]['i']}</span></div>
  <div class="frame-body">
    <img class="shot" src="{img}" alt="frame {r3[k]['i']}">
    <div class="caps">
      <div class="cap cap-3b">
        <div class="cap-head">
          <span class="model-chip m3">3B</span>
          <span class="time">{round(r3[k]['elapsed'],1)}s</span>
          <span class="tag {TAG_CLASS[j['t3']]}">{TAG_LABEL[j['t3']]}</span>
        </div>
        <p class="cap-text">{esc(r3[k]['caption'])}</p>
      </div>
      <div class="cap cap-8b">
        <div class="cap-head">
          <span class="model-chip m8">8B</span>
          <span class="time">{round(r8[k]['elapsed'],1)}s</span>
          <span class="tag {TAG_CLASS[j['t8']]}">{TAG_LABEL[j['t8']]}</span>
        </div>
        <p class="cap-text">{esc(r8[k]['caption'])}</p>
      </div>
    </div>
  </div>
  <div class="note">{esc(j['note'])}</div>
</section>'''

cards = "\n".join(card(n, k) for n, k in enumerate(keys))

html = f'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>3B vs 8B 캡셔닝 — 어려운 사진 14장 비교</title>
<style>
:root {{
  --bg:#0e1013; --panel:#171a20; --edge:#262b34; --fg:#e8ebf0; --dim:#8b93a3;
  --m3:#6ea8fe; --m8:#ffb454; --ok:#5ec98a; --warn:#e8b339; --bad:#e06c75;
  --mono: ui-monospace, SFMono-Regular, Menlo, "Cascadia Mono", monospace;
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
       font: 14px/1.6 -apple-system, "Segoe UI", Roboto, sans-serif; }}
header {{ padding: 22px 28px 18px; background: var(--panel); border-bottom: 1px solid var(--edge);
         position: sticky; top: 0; z-index: 5; }}
h1 {{ margin: 0 0 4px; font-size: 19px; font-weight: 650; text-wrap: balance; }}
.sub {{ color: var(--dim); font-size: 13px; }}
.stats {{ display: flex; gap: 0; margin-top: 16px; flex-wrap: wrap; }}
.stat {{ padding: 0 22px 0 0; margin-right: 22px; border-right: 1px solid var(--edge); }}
.stat:last-child {{ border-right: none; margin-right: 0; }}
.stat .v {{ font: 650 20px/1 var(--mono); font-variant-numeric: tabular-nums; }}
.stat .l {{ font-size: 11px; color: var(--dim); text-transform: uppercase; letter-spacing: .04em; margin-top: 3px; }}
.stat.m3 .v {{ color: var(--m3); }} .stat.m8 .v {{ color: var(--m8); }} .stat.bad .v {{ color: var(--bad); }}
.stat.ok .v {{ color: var(--ok); }}
.breakdown {{ display: flex; gap: 18px; margin-top: 14px; font-size: 12px; color: var(--dim); flex-wrap: wrap; }}
.breakdown b {{ color: var(--fg); font-family: var(--mono); }}
.bar {{ display: inline-flex; height: 8px; width: 90px; border-radius: 4px; overflow: hidden; vertical-align: middle; margin: 0 6px; }}
.bar span {{ display: block; height: 100%; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 22px 28px 60px; display: flex; flex-direction: column; gap: 22px; }}
.frame {{ background: var(--panel); border: 1px solid var(--edge); border-radius: 10px; overflow: hidden; }}
.frame-head {{ padding: 10px 16px; border-bottom: 1px solid var(--edge); }}
.fnum {{ font: 650 12px var(--mono); color: var(--dim); letter-spacing: .03em; }}
.frame-body {{ display: grid; grid-template-columns: 280px 1fr; gap: 0; }}
.shot {{ width: 100%; height: 100%; min-height: 200px; object-fit: cover; display: block; }}
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
  <h1>Qwen2.5-VL-3B vs Qwen3-VL-8B — 어려운 사진 14장 비교</h1>
  <div class="sub">samsung_dataset 1473프레임 중 블러(Laplacian variance)·저조도·과다노출 상위 프레임만 선별. 동일 프롬프트·max_gen_len=120.</div>
  <div class="stats">
    <div class="stat m3"><div class="v">{avg3}s</div><div class="l">3B 평균</div></div>
    <div class="stat m8"><div class="v">{avg8}s</div><div class="l">8B 평균</div></div>
    <div class="stat"><div class="v">{ratio}&times;</div><div class="l">8B가 더 느림</div></div>
    <div class="stat bad"><div class="v">{tally3['hallucination']}/14</div><div class="l">3B 환각</div></div>
    <div class="stat ok"><div class="v">{tally8['hallucination']}/14</div><div class="l">8B 환각</div></div>
  </div>
  <div class="breakdown">
    <span>3B &nbsp;정확 <b>{tally3['accurate']}</b> · 일부오판 <b>{tally3['partial']}</b> · 환각 <b>{tally3['hallucination']}</b></span>
    <span>8B &nbsp;정확 <b>{tally8['accurate']}</b> · 일부오판 <b>{tally8['partial']}</b> · 환각 <b>{tally8['hallucination']}</b></span>
  </div>
</header>
<main>
{cards}
</main>
</body>
</html>'''

open('/home/user/remembr/out/caption_compare_15/compare15.html', 'w').write(html)
print("written", len(html)/1024, "KB")
