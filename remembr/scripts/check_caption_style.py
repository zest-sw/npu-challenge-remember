"""Does a caption read like a search-index entry, or like a chat reply?

The paper's prompt produced 271/272 captions opening "Certainly! Let's break down the
image description step by step:" at a median 1182 characters of markdown lists (#50).
Every caption shared that opening, so it was dead weight in the embedding and dead
tokens in the answer context. `--caption_style terse` replaces the prompt; this measures
whether the replacement held, before a full run spends ~7 hours of board time.

    python remembr/scripts/check_caption_style.py <captions.json | *.partial.jsonl>
    python remembr/scripts/check_caption_style.py            # the 3B captions on disk

Exit code 1 if any caption breaks a rule, so a run script can gate on it.
"""
import json
import os
import re
import statistics
import sys

DEFAULT = 'remembr/data/captions/0/captions/captions_tt-qwen25-vl-3b_3_secs.json'

# Each rule is (name, regex). A caption matching any of them is a violation.
RULES = (
    ('어시스턴트 머리말', re.compile(r'^\s*(certainly|sure|of course|here(\'s| is)|let me|'
                                r'let\'s|okay|absolutely|i\'ll|i will|as (a|i|the))\b', re.I)),
    ('마크다운', re.compile(r'\*\*|^\s*#{1,6}\s|^\s*[-*]\s|^\s*\d+\.\s', re.M)),
    ('"the image" 류', re.compile(r'\bthe (image|photo|picture|frame|video)\b', re.I)),
    ('헤지 표현', re.compile(r'\b(appears? to be|seems? to be|likely|probably|possibly)\b', re.I)),
    ('과제 언급', re.compile(r'\bstep by step\b|\bbreak (this|it) down\b|\bdescription\b', re.I)),
)

WORD_LIMIT = 60


def load(path):
    """Both shapes the pipeline writes: the final JSON list and the .partial.jsonl."""
    if path.endswith('.jsonl'):
        with open(path) as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with open(path) as handle:
        return json.load(handle)


def violations(caption):
    broken = [name for name, pattern in RULES if pattern.search(caption)]
    words = len(caption.split())
    if words > WORD_LIMIT:
        broken.append(f'{words}단어 (상한 {WORD_LIMIT})')
    return broken


def report(path, show=6):
    rows = load(path)
    captions = [row['caption'] for row in rows]
    print(f"{os.path.basename(path)}   캡션 {len(captions)}개\n")

    for index, caption in enumerate(captions[:show], 1):
        broken = violations(caption)
        mark = '어긋남: ' + ', '.join(broken) if broken else '규칙 통과'
        print(f"--- {index}  {len(caption)}자 / {len(caption.split())}단어  [{mark}]")
        print(caption.strip()[:600] + ('…' if len(caption) > 600 else ''))
        print()
    if len(captions) > show:
        print(f"(위 {show}개만 전문 출력. 아래 통계는 {len(captions)}개 전부)\n")

    chars = [len(c) for c in captions]
    words = [len(c.split()) for c in captions]
    print(f"길이   중앙값 {statistics.median(chars):.0f}자 / {statistics.median(words):.0f}단어"
          f"   최대 {max(chars)}자")
    latencies = [row['caption_latency_s'] for row in rows if row.get('caption_latency_s')]
    if latencies:
        print(f"생성   캡션당 중앙값 {statistics.median(latencies):.1f}s"
              f"   272개 환산 {statistics.median(latencies) * 272 / 3600:.1f}시간")

    print("\n규칙별 어긋난 캡션 수:")
    failed = 0
    for name, pattern in RULES:
        hit = sum(1 for c in captions if pattern.search(c))
        failed += bool(hit)
        print(f"  {name:18s} {hit:4d}/{len(captions)}")
    long = sum(1 for w in words if w > WORD_LIMIT)
    print(f"  {WORD_LIMIT}단어 초과       {long:4d}/{len(captions)}")

    clean = sum(1 for c in captions if not violations(c))
    print(f"\n전부 통과: {clean}/{len(captions)}")
    return clean == len(captions)


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    sys.exit(0 if report(target) else 1)
