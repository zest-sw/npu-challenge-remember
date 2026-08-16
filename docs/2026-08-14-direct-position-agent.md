# DirectPositionAgent (`direct+`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 검색어 생성 콜 없이 검색하고, 문법으로 답변을 한 갈래만 허용하는 최소 파이프라인 — LLM 콜 1회, 리트라이 없음.
(원안은 "질문 전문을 검색어로, 답은 `[x, y, z]`"였다. 둘 다 실측으로 반증돼 바뀌었다 — 위 "실행 결과" 참고.)

**Architecture:** `DirectPositionAgent(Agent)` 독립 클래스 하나. 검색어 생성 콜 없음, 답변 유형 콜 없음(항상 position), 답변은 기억 번호 하나(문법이 `{'type':'integer','enum':[1..k]}`만 허용). eval.py에 `direct+` 분기 하나 추가.

**Tech Stack:** 기존 것만 — langchain_openai `ChatOpenAI`(tt 서버의 `/v1/chat/completions` + json_schema 문법 강제), Milvus 메모리. 새 의존성 없음, 새 프롬프트 파일 없음.

**Spec:** 별도 스펙 파일 없음(bounded). 채팅에서 승인된 설계(2026-08-14)를 아래 "설계 요약"에 전부 담았다 — 이 문서만 읽으면 된다.

---

## 실행 결과 (2026-08-15) — 먼저 읽을 것

구현·측정 완료. **한 줄 요약: 병목은 검색도 식별도 아니라 "좌표를 주고 거리를 암산시킨 것"이었다. 계산을 코드로 옮겨 42.0 m → 21.6 m.**

계획서의 두 결정이 실측으로 반증돼 바뀌었다 — 아래 "설계 요약"의 preamble 항목과 답변 계약 항목을 그 순서로 읽으면 된다.

### 최종 성적

position 12문제(`human_qa.json`에서 추출), `captions_tt-qwen3-vl-8b-terse-200words-512-1f_3_secs`, `llm-8b` 모드, **`--temperature 0`**(아래 "측정 방법" 참고).

| | 평균 오차 | 5m 이내 | 20m 이내 | 질문당 |
|---|---|---|---|---|
| 좌표를 보여주고 모델이 거리 비교 | 42.0 m | 2/12 | 5/12 | 5.4s |
| **거리를 계산해 숫자로 제시 (채택)** | **21.6 m** | 4/12 | 8/12 | **4.9s** |
| 검색 5개 중 최선을 고른다면 | 7.1 m | 8/12 | 11/12 | |
| 물체를 이름 부른 캡션 중 최선 (**진짜 목표**) | 3.3 m | | | |

**7.1 m를 "상한"으로 읽지 말 것** — 아래 "7.1 m는 오라클 수치다" 참고.

두 번 돌려 결과가 동일함을 확인했다(결정론).

### ⚠️ 측정 방법 — `--temperature 0`으로 잴 것

eval 기본값은 0.7이고, 그 값으로는 **같은 설정을 3회 돌린 평균이 27.0 / 39.0 / 39.6 m로 12.6 m 흔들린다.** 이 폭이 위 20.4 m 차이를 덮어서, 처음 단일 실행으로 비교했을 때는 두 방식 차이가 -1.9 m로 뒤집혀 보였다.

`direct+`에는 0.7을 쓸 이유가 없다. `evaluate_output`은 `position`이 None일 때만 미답변으로 보는데 이 에이전트는 항상 채우므로 **미답변이 될 수 없고, eval의 재시도가 발동하지 않는다**(모든 실행에서 12/12가 attempts=1). AGENTS.md가 기록한 temp 0.7의 이득(미답변 4 → 0)은 전적으로 재시도가 다른 샘플을 뽑아서 생긴 것이므로, 재시도가 없는 이 에이전트에는 이득 없이 잡음만 남는다.

### 원인 분류

| 원인 | 문제 수 | 해당 |
|---|---|---|
| 잘 됨 (≤15m) | 8/12 | Q1, Q3, Q5, Q6, Q8, Q9, Q10, Q12 |
| **물어본 물체가 검색된 5개 어디에도 없음** | 4/12 | Q2, Q4, Q7, Q11 |

실패한 4문제의 캡션을 직접 읽어보면 원인이 하나다. 소화전·경찰 비상연락 기둥·동상을
**이름으로 적은 캡션이 상위 5개에 없다.** 그 캡션들은 DB에 있지만 유사도 순위 11·33·48위에
있어서 `k=5`로는 닿지 않는다.

    Q2  "노란 경찰 비상연락 기둥"  ->  후보 5개: 비상구 표지판 / 가로등 / 주황 셔츠 입은 사람
                                       / 정지 표지판 / 흰 구형 가로등    (경찰 기둥 없음)
                                       모델은 "주황 셔츠"를 골랐다 -- 색만 근사한 단서
    Q11 "빨간 소화전"              ->  후보 5개 전부 벽돌 건물·초록 컨테이너·경사로 (소화전 없음)
    Q7  "Gates 건물 앞 동상"       ->  후보 5개에 동상 없음. 모델은 질문에 나온 건물 이름을
                                       그대로 담은 캡션을 골랐다 -- 합리적이지만 정답은 동상 위치

즉 모델이 잘못 고른 것이 아니라 **고를 후보가 안 왔다.** 없는 상황에서 부분 단서(색, 건물 이름)로
처신한 것이고, 통제 실험에서 확인한 "정답이 없으면 아무거나 고른다"와 같은 상황이다.

### ⚠️ 7.1 m는 오라클 수치다 -- 도달 가능한 목표가 아니다

"검색 5개 중 최선을 고르면 7.1 m"를 개선 목표로 읽으면 안 된다. 그 최선이 **텍스트로는 알 수
없는 값**이기 때문이다.

Q2에서 정답에 가장 가까운 후보(8.1 m)의 캡션은 "금속 가로등이 있는 보도"다. 경찰 기둥 얘기가
한 글자도 없다. 그게 정답 근처라는 걸 아는 방법은 **정답 좌표를 미리 아는 것뿐**이다. 어떤
추론으로도 닿을 수 없다.

도달 가능한 목표는 **3.3 m** -- 물체를 이름으로 부른 캡션 중 최선이다. 그건 검색이 그 캡션을
상위로 올리기만 하면 텍스트만 보고도 고를 수 있다.

### 검색에 대해 알게 된 것

- **유사도 1위가 최선인 경우가 0/12.** 최선은 3위에 6번, 2위·4위에 각 2번, 6위에 1번. 임베딩 유사도 순위와 공간 정확도 순위가 일치하지 않는다.
- **정답 캡션이 깊이 묻히는 경우가 있다** — 동상을 이름으로 묘사한 캡션이 48위·33위(Q3·Q7), 쓰레기통이 11위(Q4). `k=5`로는 닿지 않는다.
- k별 상한: `k=1` 28.5 m, `k=2` 15.4 m, `k=3` 8.0 m, `k=5` 7.1 m, `k=10` 5.0 m.
- **"창 안 아무 캡션 중 최선"은 항상 0.0 m이지만 의미 없는 지표다.** 정답 좌표는 로봇이 지나간 지점이라 경로가 되돌아 통과하고, 그 0m 캡션들은 물어본 물체를 묘사하지 않는다(atrium·pavement·hallway). 의미 있는 목표치는 "물체를 이름 부른 캡션 중 최선" = 3.3 m.

### 검색 개선 (2026-08-16): 문장 단위 인덱싱

캡션 1개 = 임베딩 1개인데 캡션은 중앙값 93단어·7문장이다. 벤치를 묻는 질문이 매칭해야 하는
것은 보도·담장·벤치·배전함·건물·사람을 평균 낸 벡터라, 벤치 신호가 1/7로 희석된다. 문장마다
따로 임베딩하고 검색 결과를 부모 캡션으로 되돌리면 그 희석이 사라진다.

12문제 오라클(검색 5개 중 최선):

| 방식 | 오라클 | 물체 언급 캡션이 top5에 |
|---|---|---|
| caption (기존, L2·정규화 없음) | 7.1 m | 11/12 |
| + 정규화만 (코사인) | 5.0 m | 11/12 |
| **sentence (정규화 포함)** | **1.6 m** | **12/12** |

Q7은 물체를 이름 부른 캡션이 **33위 -> 5위**로 올라왔다. Q4는 오라클 52.4 -> 2.6 m. Q4를
빼도 3.0 -> 1.55 m라 한 문제짜리 착시가 아니다. 손해도 있다 -- Q1·Q5·Q8이 0.0 -> 2.9 m,
Q10이 0.0 -> 5.6 m. 물체를 언급하는 더 먼 캡션이 가까운 것을 밀어낸 경우다.

넘파이로 잰 값과 실제 `MilvusMemory`를 돌린 값이 질문별로 완전히 일치한다.

**이것은 오라클이지 성적이 아니다.** 기존 오라클 7.1 m일 때 실제는 21.6 m였다. 천장이 내려간
것이고, 실제 A/B는 `--index_unit sentence`로 재야 한다(NPU 필요).

### 반증된 것

- **"프롬프트에 선택 기준을 명시하면 낫다"**, **"문법에 커밋 자리가 없어서 틀린다"** — 둘 다 "모델이 고르는 법을 모른다"는 가정이었고, 그 가정이 틀렸다(식별 64/64). 반증의 근거는 두 변경의 A/B 숫자가 아니다 -- 그건 temp 0.7 단일 실행이라 잡음 안이다 -- `probe_memory_selection.py`의 통제 실험이다.
- **"모델의 선택 능력이 부족하다"** — `probe_memory_selection.py`가 합성 입력으로 분리해 재보니 **식별은 64/64**이다: 정답이 1~5번 어느 자리든, 후보가 20개여도, 방해자가 같은 장면 유형이어도, 5문장에 파묻혀도, 질문의 명사가 한 번도 안 나오는 의역이어도 찾아낸다. 못 하는 것은 산수뿐이다.
- **"위치 편향(뒷부분 선호)"** — 편향 자체가 원인이 아니라 **증상**이다. 거리 판단을 포기할 때 마지막 항목으로 떨어진다: 실제 기하(97.1/104.5/122.2/123.0 m)에서 정답을 매 자리로 옮겨도 좌표를 주면 1/5(대부분 마지막을 답함), 거리를 숫자로 주면 4/5.
- **"검색어 생성 콜을 없애서 인덱싱을 앞질렀다"** — 아래 flush 버그의 원인 가설이었으나 틀렸다(같은 질의를 0.3~40초에 반복해도 결과 동일).

### 부수 발견: Milvus flush 버그 (PR #73, 머지됨)

이 브랜치를 측정하다 찾았고, 이 작업 전체에서 가장 값진 산출물이다. `milvus_memory.py`에 `flush()` 호출이 없어서, insert 직후 검색하는 eval이 **일부만 보이는 메모리**를 검색하고 있었다(`k=5` 요청에 1개나 0개 반환, `num_entities`가 0). `MilvusMemory`를 쓰는 모든 에이전트에 해당하며, 이 수정 전 측정치는 전부 이 영향을 받았다.

### 남은 과제

1. **검색 순위 개선 — 문장 단위 인덱싱으로 해결됐다**(위 절). 남은 일은 `--index_unit sentence`
   실제 A/B뿐이다.
   - **어휘 재순위는 반증됐다.** 후보 20개를 IDF 가중 어휘 겹침으로 재정렬해도 **12문제 중 10개는
     상위 5개 구성이 그대로**였고, 새로 들어온 Q9·Q12는 6위->5위 한 칸이라 오라클이 안 변했다.
     11·33·48위에 묻힌 캡션은 하나도 못 끌어올렸다. 21.6 -> 16.1 m는 같은 5개의 **순서가 바뀌어
     모델이 다른 걸 고른** 결과이고, 그 개선의 92%가 Q2 한 문제다. 캡션에 `recycling`·`police`·
     `statue` 같은 단어가 아예 없어서 매칭할 대상이 없다.
   - **LLM 쿼리 생성도 반증됐다**: 원문 질문 21.6 m vs 원본 프롬프트 47.2 m, 최선의 개선 프롬프트
     31.4 m. LLM 콜도 하나 더 든다.
   - **후보 수를 줄이는 쪽도 반증됐다**: `k`=1/2/3이 28.5·30.5·34.6 m, `k`=10/20이 36.3·41.5 m로
     `k`=5(21.6 m)보다 나쁘다. top-20은 top-5를 포함하므로 오라클은 나빠질 수 없는데 성적이 두 배
     나빠졌다 -- 검색이 아니라 **고르기**가 무너진다.
2. **모델 쪽은 더 짜낼 게 별로 없다** — `probe_memory_selection.py`가 식별 64/64을 보이고, 남은 실패는 후보가 안 온 경우다.

   프롬프트·출력형식을 네 번 손댔지만 **전부 없는 병을 고치려던 것이었다.** 각각 무엇을
   의심했는지 남겨둔다:

   | 시도 | 의심한 것 | 한 것 |
   |---|---|---|
   | 선택 기준 명시 | 뭘 기준으로 고를지 안 알려줬다 | 프롬프트에 "물체를 언급한 기억을 골라라" 추가 |
   | 인덱스 출력 | 좌표를 받아적다 딴 기억 것을 베낀다 | 좌표 대신 번호로 답하게 함 |
   | 캡션을 앞으로 | 근거가 긴 줄 맨 뒤라 안 읽힌다 | 캡션을 줄 맨 앞으로 |
   | 캡션 첫 문장만 | 92단어에 핵심 명사가 묻힌다 | 첫 문장만 제시 |

   앞의 둘은 "고르는 법을 모른다", 뒤의 둘은 "제대로 못 읽는다"는 가정이었는데 **둘 다
   사실이 아니었다**(식별 64/64). 실제 원인은 좌표로 거리를 암산시킨 것과 후보 부재였다.

   **이 네 개의 A/B 숫자는 근거로 쓰지 말 것** — 전부 `--temperature 0.7` 단일 실행이라
   실행 간 변동폭(12.6 m) 안에 들어간다. 프롬프트로 안 된다는 근거는 그 숫자가 아니라
   식별 64/64과 후보 부재 쪽이다.
3. **현재 위치를 문자열이 아니라 인자로 받기** — 지금 `current_position()`이 질문 문자열에서
   정규식으로 좌표를 꺼낸다. 동작하지만 왕복이다:

       벤치마크   `form_question_jsons.py`가 캡션에서 좌표를 계산 -> 문장에 글자로 삽입
       실제 로봇  `agent_node.py`가 `/amcl_pose`로 좌표를 받음 -> 문장에 글자로 삽입
       양쪽 다    `direct_position_agent.py`가 정규식으로 되꺼냄

   실제 로봇은 **이미 구조화된 좌표를 손에 들고 있다**(`agent_node.py`가 `/amcl_pose`를 구독한다).
   그걸 글자로 바꿨다가 파싱하는 것은 순수한 손실이고, 문구 변경에 조용히 깨진다 -- 실제로 두
   경로의 문장이 이미 다르다(벤치마크 `you are located at [...]`, 로봇
   `You are currently located at [...] and the time is ...`). 지금 정규식이 둘 다 잡는 것은 우연이다.

   `Agent.query(question, here=None)`처럼 인자로 받으면 정규식이 사라지고 실시간 값을 그대로 쓴다.
   기본값 `None`이면 기존 에이전트는 안 깨지지만 base 클래스와 호출부(eval, `agent_node.py`) 변경이라
   별도 PR 사안이다.

   참고로 문자열에 넣는 설계는 NVIDIA 원본이다(`cf2db07`, John Welsh). 벤치마크만의 편법이 아니라
   실제 배포 경로도 같다. 그 선택의 대가가 위 preamble 표의 209.72 m다.

4. **time/duration 확장 시 답변 계약 재설계** — 인덱스 방식은 "답이 검색된 기억의 좌표 중 하나"라는 전제에 서 있다. position에서는 12/12 성립하지만(소수점 9자리 일치) 계산이 필요한 유형에는 그대로 쓸 수 없다.

---

### 측정 재현 절차

질문 파일과 캡션은 `remembr/data/`(gitignored)에 있어 커밋되지 않는다. 위 숫자를 다시 내려면:

```bash
# position 12문제만 뽑는다 (human_qa.json 30문제 중)
python - <<'EOF'
import json
src = '<데이터 경로>/questions/0/human_qa.json'
d = json.load(open(src))
pos = [x for x in d['data'] if x['type'] == 'position']
json.dump({'version': d.get('version', 0.1), 'data': pos},
          open('remembr/data/questions/0/human_qa_pos12.json', 'w'), indent=2)
EOF

# 캡션은 읽기 전용으로 연결
ln -s <데이터 경로>/captions remembr/data/captions

# temperature 0 필수 -- 기본값 0.7에서는 실행 간 12.6 m 흔들린다
REMEMBR_TRACE=1 PYTHONPATH=$(pwd) python remembr/scripts/eval.py \
  --model direct+tt/Llama-3.1-8B-Instruct --sequence_id 0 \
  --qa_file human_qa_pos12 \
  --caption_file captions_tt-qwen3-vl-8b-terse-200words-512-1f_3_secs \
  --data_dir remembr/data --out_dir out --temperature 0 --postfix _repro

# 문장 단위 인덱싱과 비교할 때는 --index_unit sentence 하나만 바꾼다.
# 캡션 파일의 임베딩을 못 쓰고 문장마다 새로 임베딩하므로 적재가 느려진다.
```

모델 능력 쪽 숫자(식별 64/64, 좌표 1/5 vs 거리 4/5)는 데이터가 필요 없다 --
`python remembr/scripts/probe_memory_selection.py`가 합성 입력으로 102콜에 재현한다.

## 설계 요약 (승인된 결정들)

- ~~**검색어**: 질문 **전문 그대로** `memory.search_by_text()`에 넣는다. preamble 잘라내기 없음 (`RetrieveAgent.asked_line` 안 씀).~~
  **→ 보드 실측으로 반증됨 (2026-08-15). 아래 "preamble 처리" 참고.**
- **preamble 처리** (실측 후 확정): 질문은 항상 `<preamble>\n<asked>` 두 줄이다 (`form_question_jsons.py:214`). `split_question()`으로 나눠서 **검색에는 asked만, LLM 프롬프트에는 preamble을 라벨 붙인 별도 블록으로** 넣는다. 같은 컨텍스트·질문·greedy에서 측정한 값 (정답 `[-60.197, 196.541, -1.097]`):

  | 프롬프트 모양 | 현재 위치 정보 | 생성된 좌표 | 오차 |
  |---|---|---|---|
  | preamble을 질문 문장에 그대로 (원래 계획) | 있음 | `[0.097, -4.321, 0.003]` | 209.72 m |
  | + "현재 위치는 답이 아니다" 명시 | 있음 | `[0.0, 0.0, 0.0]` | 205.56 m |
  | preamble 삭제 (`asked_line`만) | **잃음** | `[-60.197, 196.541, -1.097]` | 0.00 m |
  | **preamble을 라벨 블록으로 hoist (채택)** | **있음** | `[-60.197, 196.541, -1.097]` | **0.00 m** |

  좌표를 요구하는 문장 안에 좌표가 있으면 모델이 그걸 답으로 낸다 — 자기 현재 위치인데도. 프롬프트로 "그건 답이 아니다"라고 못 박아도 안 고쳐진다. 그런데 preamble을 그냥 버리면 데이터셋 절반이 못 풀린다: "Where is the **nearest** recycling bin?"은 현재 위치 없이 어느 게 가장 가까운지 고를 수 없고, "**How long have you been** outside?"는 현재 시각 없이 계산이 안 된다. 그래서 버리지 않고 문장 밖으로 옮긴다. 검색어에서는 그냥 버린다 — 로봇 자기 위치·시각은 retriever에 쓸모없고 임베딩만 희석시킨다 (`RetrieveAgent`가 `asked_line`을 쓰는 것과 같은 이유).

  **미검증**: 이 측정들은 현재 위치가 **필요 없는** 질문("동상 어디 있어?") 하나로 한 것이다. 라벨을 아예 뺀 조건도 똑같이 0.00 m였으므로, "모델이 라벨을 실제로 활용한다"가 아니라 "라벨로 올려두면 방해가 안 된다"까지만 증명됐다. `nearest`/`how long ago` 질문으로 활용 여부를 따로 재야 한다. 그리고 전부 n=1이다.
- **비위치 질문 처리**: 없음. 모든 질문에 좌표로 답한다. time/binary/text 질문은 알고 틀리는 position 전용 프로브다.
- **리트라이**: **없음.** LLM 콜은 질문당 정확히 1회. 파싱 실패는 즉시 raise — `eval.answer_squad_question`의 MAX_ATTEMPTS=5가 받아서 무응답 처리한다. greedy(temperature=0)에서는 재시도가 같은 바이트를 되풀이하므로 내부 리트라이는 시간 낭비다.
- **백엔드**: `tt/` 전용. 문법 강제(`response_format: json_schema`)가 요지라 Ollama 경로는 만들지 않고 `__init__`에서 거부한다.
- ~~**문법**: top-level 배열 `{'type':'array','items':{'type':'number'},'minItems':3,'maxItems':3}`.~~
  **→ 기억 번호로 변경 (2026-08-15).** top-level 배열은 보드에서 정상 동작함을 확인했다(계획의 최대 리스크였고 해소됨). 그런데 실패 3건이 전부 "다른 기억의 좌표를 정확히 베낀" 형태여서, 좌표를 타이핑하는 일을 모델에서 떼어냈다. 문법은 `{'type':'integer','enum':[1..k]}`이고 코드가 그 번호의 좌표를 꺼낸다. `enum`을 쓴 이유는 agent.py가 기록한 대로 lm-format-enforcer가 형제 키워드를 버리는 형태가 있어 `minimum/maximum`보다 안전하기 때문이다(보드에서 두 형태 다 동작 확인). 정답을 잃지 않는 근거는 위 "실행 결과"에 있다.
- **향후 확장 (이번 범위 아님)**: 맨 앞에 답변 유형 확정 단계를 붙여 binary/text/time도 각 유형의 bare 출력만 뱉게 한다. 그 seam은 유형별 `*_SCHEMA` 상수 + 유형별 `*_output()` 매퍼가 `POSITION_SCHEMA`/`position_output`을 미러링하는 것 — 지금은 position 것만 만든다.

## Global Constraints

- LLM 콜: 질문당 정확히 1회. 리트라이 루프 금지.
- 새 pip 의존성 금지, 새 프롬프트 txt 파일 금지 (시스템 프롬프트는 인라인 상수).
- eval 모델 문자열: `direct+tt/<model>` (예: `direct+tt/Llama-3.1-8B-Instruct`).
- 테스트는 보드·서버·Milvus 없이 돌아야 한다 (기존 `test_non_agent_retry.py` 패턴: `__new__` + Fake 객체 + `__main__` 러너).
- 커밋 메시지 끝에 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## File Structure

- Create: `remembr/agents/direct_position_agent.py` — 스키마 상수, `position_output()` 매퍼, `DirectPositionAgent` 클래스 (전부 한 파일, ~80줄)
- Create: `remembr/tests/test_direct_position_agent.py` — 오프라인 자가 테스트
- Modify: `remembr/scripts/eval.py` — import 1줄 + `elif 'direct' in args.model:` 분기 1개

---

### Task 1: 문법 스키마 + `position_output()` 매퍼

**Files:**
- Create: `remembr/agents/direct_position_agent.py` (모듈 하단의 클래스는 Task 2에서)
- Test: `remembr/tests/test_direct_position_agent.py`

**Interfaces:**
- Consumes: `remembr.agents.agent.AgentOutput` (기존)
- Produces: `POSITION_SCHEMA: dict`, `position_output(parsed) -> AgentOutput` — Task 2와 3이 이 이름 그대로 쓴다.

- [x] **Step 1: 실패하는 테스트 작성**

`remembr/tests/test_direct_position_agent.py` 생성:

```python
"""Self-check for the direct+ pipeline: question -> search -> a bare [x, y, z].

No NPU, no Milvus, no server:

    python remembr/tests/test_direct_position_agent.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from remembr.agents.agent import AgentOutput  # noqa: E402
from remembr.agents.direct_position_agent import (  # noqa: E402
    POSITION_SCHEMA,
    position_output,
)


def test_schema_is_exactly_three_numbers():
    """The grammar is the whole contract: three numbers, nothing else generable."""
    assert POSITION_SCHEMA == {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
    }, POSITION_SCHEMA


def test_a_bare_array_becomes_a_position_answer():
    out = position_output([1, 2.5, -3])
    assert isinstance(out, AgentOutput)
    assert out.type == "position"
    assert out.position == [1.0, 2.5, -3.0]
    assert out.time is None and out.duration is None and out.binary is None


def test_wrong_shapes_are_rejected():
    # Shapes the old loose-string field let through and eval then mis-scored.
    for bad in ([1, 2], [[1, 2, 3]], "at [1,2,3]", [1, 2, "a"], [True, 1, 2], None):
        try:
            position_output(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  ok   {name}")
        except AssertionError as error:
            failures += 1
            print(f"  FAIL {name}: {error}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"  ERROR {name}: {type(error).__name__}: {error}")
    print(f"\n{'FAILED' if failures else 'all passed'} ({failures} failure(s))")
    sys.exit(1 if failures else 0)
```

- [x] **Step 2: 실패 확인**

Run: `python remembr/tests/test_direct_position_agent.py`
Expected: `ModuleNotFoundError: No module named 'remembr.agents.direct_position_agent'`

- [x] **Step 3: 최소 구현**

`remembr/agents/direct_position_agent.py` 생성:

```python
"""Question -> vector search with the question itself -> a bare [x, y, z].

No query-generation call, no answer-type call, no JSON object: the grammar
allows only a three-number array, so every generated token is answer, none is
punctuation. A position-only probe -- every question gets a coordinate, and
non-position questions are knowingly wrong.

No retries. Greedy decoding repeats itself byte-for-byte, so a second try buys
nothing; a parse failure raises and eval's own attempt cap records it.

    python remembr/scripts/eval.py --model direct+tt/Llama-3.1-8B-Instruct ...
"""
import json
import os

from remembr.agents.agent import Agent, AgentOutput
from remembr.memory.memory import Memory
from remembr.utils import trace

# The whole answer contract. Number grammar is finite, so no maxLength is
# needed, and with no string branch the anyOf/null rules in agent.py do not
# apply here.
POSITION_SCHEMA = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 3,
    "maxItems": 3,
}

SYSTEM_PROMPT = (
    "You are a robot looking back over your memory of what you saw. "
    "From the memory context, answer where the thing the question asks about "
    "is. Respond with only a JSON array of three numbers: [x, y, z]."
)


def position_output(parsed) -> AgentOutput:
    """A bare [x, y, z] as the AgentOutput eval scores.

    bool is an int subclass, so `[True, 1, 2]` must be rejected explicitly.
    """
    if (
        not isinstance(parsed, list)
        or len(parsed) != 3
        or not all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in parsed)
    ):
        raise ValueError(f"expected [x, y, z], got: {parsed!r}")
    return AgentOutput(type="position", text="", binary=None,
                       position=[float(n) for n in parsed],
                       orientation=None, duration=None, time=None)
```

- [x] **Step 4: 통과 확인**

Run: `python remembr/tests/test_direct_position_agent.py`
Expected: `all passed (0 failure(s))` — 테스트 3개 ok

- [x] **Step 5: Commit**

```bash
git add remembr/agents/direct_position_agent.py remembr/tests/test_direct_position_agent.py
git commit -m "feat(agents): direct+ 파이프라인의 [x,y,z] 문법과 출력 매퍼

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `DirectPositionAgent` — 검색 1회 + LLM 콜 1회, 리트라이 없음

**Files:**
- Modify: `remembr/agents/direct_position_agent.py` (Task 1 파일에 클래스 추가)
- Test: `remembr/tests/test_direct_position_agent.py` (테스트 추가)

**Interfaces:**
- Consumes: Task 1의 `POSITION_SCHEMA`, `position_output`; `Memory.search_by_text(str)`, `Memory.get_working_memory()`, `Memory.memory_to_string(list)` (기존)
- Produces: `DirectPositionAgent(llm_type: str, num_ctx: int = 8192, temperature: float = 0)`, `.set_memory(memory)`, `.query(question: str) -> AgentOutput` — Task 3의 eval 분기가 이 시그니처를 쓴다. `num_ctx`는 eval 호출 규격 맞춤용으로 받고 무시한다.

- [x] **Step 1: 실패하는 테스트 추가**

`test_direct_position_agent.py`의 `if __name__` 위에 추가:

```python
from remembr.agents.direct_position_agent import DirectPositionAgent  # noqa: E402


class FakeReply:
    def __init__(self, content):
        self.content = content


class FakeChain:
    """Stands in for ChatOpenAI. Counts calls so the no-retry rule is testable."""

    def __init__(self, content):
        self.content = content
        self.calls = 0

    def invoke(self, _inputs):
        self.calls += 1
        return FakeReply(self.content)


class FakeMemory:
    """Records what was searched for, so the question-as-query rule is testable."""

    def __init__(self):
        self.searched = []

    def search_by_text(self, query):
        self.searched.append(query)
        return []

    def get_working_memory(self):
        return []

    def memory_to_string(self, _memory):
        return "At time=2023-01-16 07:58:58, the robot was at [1.0, 2.0, 0.0]."


def make_agent(reply):
    """Build the agent without __init__ (which would build an LLM client)."""
    agent = DirectPositionAgent.__new__(DirectPositionAgent)
    agent.llm_type = "tt/fake"
    agent.chain = FakeChain(reply)
    agent.memory = FakeMemory()
    return agent


PREAMBLE_QUESTION = (
    "The current time is 08:10:00 and you are at [0.0, 0.0, 0.0].\n"
    "Where is the yellow police call pole?"
)


def test_query_searches_with_the_whole_question_and_calls_once():
    agent = make_agent("[1.0, 2.0, 3.0]")
    out = agent.query(PREAMBLE_QUESTION)
    # The question itself, preamble included, is the search query -- no trimming.
    assert agent.memory.searched == [PREAMBLE_QUESTION], agent.memory.searched
    assert agent.chain.calls == 1, f"expected 1 call, made {agent.chain.calls}"
    assert out.position == [1.0, 2.0, 3.0]


def test_a_parse_failure_raises_immediately_no_retry():
    agent = make_agent("[1.0, 2.0")  # truncated -- json.loads fails
    try:
        agent.query(PREAMBLE_QUESTION)
    except Exception:
        pass
    else:
        raise AssertionError("expected a raise on unparseable output")
    assert agent.chain.calls == 1, f"no retries allowed, made {agent.chain.calls}"


def test_a_wrong_shape_also_raises_immediately():
    agent = make_agent("null")  # valid JSON, wrong shape
    try:
        agent.query(PREAMBLE_QUESTION)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError for a non-array answer")
    assert agent.chain.calls == 1


def test_non_tt_backend_is_refused():
    try:
        DirectPositionAgent(llm_type="llama3")
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError for a non-tt/ backend")
```

- [x] **Step 2: 실패 확인**

Run: `python remembr/tests/test_direct_position_agent.py`
Expected: `ImportError: cannot import name 'DirectPositionAgent'`

- [x] **Step 3: 최소 구현**

`direct_position_agent.py`의 `position_output` 아래에 추가:

```python
class DirectPositionAgent(Agent):
    def __init__(self, llm_type="tt/Llama-3.1-8B-Instruct", num_ctx=8192, temperature=0):
        # num_ctx: accepted so eval.py can construct every agent the same way; the
        # tt server owns its own context budget, so it is unused here.
        if not llm_type.startswith("tt/"):
            raise ValueError(
                "DirectPositionAgent needs the tt/ server's grammar-constrained "
                f"decoding; got llm_type={llm_type!r}"
            )
        from langchain_openai import ChatOpenAI  # lazy: optional dep

        self.llm_type = llm_type
        self.chain = ChatOpenAI(
            model=llm_type[len("tt/"):],
            base_url=os.environ.get("TT_LLM_BASE_URL", "http://localhost:8100/v1"),
            api_key="local",
            temperature=temperature,
            # Three numbers and two brackets; the grammar cannot ramble past that.
            max_tokens=48,
            model_kwargs={"response_format": {
                "type": "json_schema",
                "json_schema": {"name": "position", "schema": POSITION_SCHEMA},
            }},
        )

    def set_memory(self, memory: Memory):
        self.memory = memory

    def query(self, question: str) -> AgentOutput:
        # The question itself is the search query -- no trimming, no query call.
        trace.step("검색어", repr(question))
        self.memory.search_by_text(question)
        context = self.memory.memory_to_string(self.memory.get_working_memory())

        # Exactly one LLM call. No retry: greedy decoding would repeat the same
        # bytes, and eval.answer_squad_question already caps attempts at 5.
        reply = self.chain.invoke(f"{SYSTEM_PROMPT}\n\n{context}\n\nQuestion: {question}")
        output = position_output(json.loads(reply.content))
        trace.step("답변", f"position={output.position!r}")
        return output
```

- [x] **Step 4: 통과 확인**

Run: `python remembr/tests/test_direct_position_agent.py`
Expected: `all passed (0 failure(s))` — 테스트 7개 ok

- [x] **Step 5: 기존 테스트 회귀 확인**

Run: `python remembr/tests/test_non_agent_retry.py && python remembr/tests/test_retrieve_agent.py`
Expected: 둘 다 `all passed`

- [x] **Step 6: Commit**

```bash
git add remembr/agents/direct_position_agent.py remembr/tests/test_direct_position_agent.py
git commit -m "feat(agents): DirectPositionAgent -- 질문 그대로 검색, 콜 1회, 좌표만 생성

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: eval.py에 `direct+` 분기

**Files:**
- Modify: `remembr/scripts/eval.py` — import 블록(±30행 부근)과 모델 선택 블록(±339행 `elif 'retrieve'` 분기 뒤)

**Interfaces:**
- Consumes: Task 2의 `DirectPositionAgent(llm_type, num_ctx=..., temperature=...)`
- Produces: `--model direct+tt/<name>` 실행 경로. 모델 문자열에 `direct`가 있으면 이 분기를 탄다 (`remembr`/`retrieve`/`optimal`/`vlm` 검사보다 뒤여도 충돌 없음 — 어느 키워드도 `direct+tt/...`에 안 들어 있다).

- [x] **Step 1: import 추가**

`from agents.retrieve_agent import RetrieveAgent` 아래에:

```python
from agents.direct_position_agent import DirectPositionAgent
```

- [x] **Step 2: 분기 추가**

`elif 'retrieve' in args.model:` 분기의 `use_milvus = True` 다음, `elif 'optimal'` 앞에:

```python
    elif 'direct' in args.model:
        # The question itself is the Milvus query and the answer grammar is a
        # bare [x, y, z]: no query call, no answer-type call, no JSON-object
        # tokens. Position-only probe -- non-position questions are knowingly
        # wrong. See remembr/agents/direct_position_agent.py.
        base_llm = args.model.split('+')[-1]
        agent = DirectPositionAgent(llm_type=base_llm, num_ctx=args.num_ctx,
                                    temperature=args.temperature)
        use_milvus = True
```

- [x] **Step 3: import 검증 (보드 없이)** — 부분 검증만 됨

Run: `python -c "import sys; sys.path.insert(0, 'remembr'); import scripts.eval" 2>&1 | tail -1` 또는 `python remembr/scripts/eval.py --help`
Expected: import 에러 없이 argparse 도움말 출력

실제: **이 환경에선 불가.** `/tt-metal/python_env`의 `langgraph`가 깨진 namespace
package(`langgraph.graph` 없음)라 eval.py 27행 `from agents.remembr_agent import
ReMEmbRAgent`에서 실패한다. 이 변경 전(git stash)에도 동일하게 실패 — 선행 조건이다.
대신 검증한 것:
  * `python -m py_compile remembr/scripts/eval.py` → 문법 OK
  * `sys.path.insert(0,'remembr'); from agents.direct_position_agent import
    DirectPositionAgent` → import OK (eval.py가 쓰는 것과 같은 경로 형태)
  * 분기 디스패치: `direct+tt/...`가 `remembr`/`retrieve`/`optimal`/`vlm` 어느
    키워드에도 안 걸리고 `direct` 분기를 탄다
`langgraph`가 성한 환경에서 `--help`를 한 번 돌려 마무리할 것.

- [x] **Step 4: 전체 오프라인 테스트 일괄 확인**

Run: `for t in remembr/tests/test_*.py; do python "$t" || break; done`
Expected: 전부 `all passed`

실제: 5/7 `all passed`. 실패한 `test_answer_vocabulary.py`,
`test_schema_constraints.py`는 위와 같은 `langgraph` 문제로 `remembr_agent`를
import하지 못해서이며 이번 변경과 무관하다(새 모듈은 `langgraph`를 안 쓴다).
`test_direct_position_agent.py`(7개), `test_non_agent_retry.py`,
`test_retrieve_agent.py`는 전부 통과 — 회귀 없음.

- [x] **Step 5: Commit**

```bash
git add remembr/scripts/eval.py
git commit -m "feat(eval): --model direct+tt/... 분기 추가

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## 보드에서의 검증 (계획 밖, 참고)

오프라인 테스트가 전부 통과해도 실제 효과는 보드에서만 보인다:

```bash
python remembr/scripts/eval.py --model direct+tt/Llama-3.1-8B-Instruct --caption_file <기존 캡션> ...
```

비교 대상은 `retrieve+tt/...`(검색어 생성 콜 있음, JSON 오브젝트 답변)와 position 질문 정확도·질문당 소요 시간. JSON 오브젝트 토큰 절약이 실제 몇 초인지 여기서 측정된다.

**먼저 할 것 — 질문 1개짜리 스모크 체크.** 오프라인 테스트가 `FakeChain`을 쓰므로 이
계획의 유일한 미검증 리스크는 서버에 남아 있다: `POSITION_SCHEMA`는 이 repo에서 처음으로
**top-level이 object가 아닌 array**인 스키마다. 지금까지의 모든 사용례(`ANSWER_SCHEMA`,
`ANSWER_TYPE_SCHEMA`, `probe_query_format.py`의 `SCHEMA_*`)는 top-level object였다.
`presentation/http/app.py`는 `wrapper.get("schema", wrapper)`로 스키마를 그대로 넘기고
`schemas.py`도 `dict[str, Any]`라 서버가 막는 부분은 없지만, 실제 문법을 만드는 건
`lmformatenforcer.JsonSchemaParser`이고 top-level 배열에서의 동작은 아직 실측되지 않았다.
전체 eval을 돌리기 전에 질문 하나로 `[x, y, z]`가 실제로 나오는지 확인할 것 — 여기서 막히면
이후 비교 실행이 전부 헛수고가 된다.

또 `--model direct+tt/<name>`의 `<name>`은 서버가 **현재 올라와 있는 모드**의 모델이어야
한다(`POST /mode`가 자동 전환되지 않고 409를 낸다). 8B를 쓰려면 서버가 `llm-8b` 모드여야
하고, 그 모드에선 VLM이 함께 상주하지 못한다 — 기본 `3b+3b`라면
`direct+tt/Llama-3.2-3B-Instruct`를 쓸 것.
