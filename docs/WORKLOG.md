# 작업 로그 — NPU eval 파이프라인 디버깅 & agent/3B/8B/perf/acc 8-way 비교

NPU 포트(Qwen2.5-VL 캡셔너 + Llama LLM)의 NaVQA eval을 **처음으로 유효하게 완주**시키고,
Llama를 **Optimal+/ReMEmbR × 3B/8B × performance/accuracy 8개 구성**으로 비교. eval이 결과를 못 남기던 원인 3종
(정답 타임존·저장경로·무한 재시도)을 모두 고쳤고, 8개 구성 결과를 확보했다(Optimal+ 3B 각 5회·8B 각 3회,
ReMEmbR 3B-perf 1회·3B-acc 2회·8B 각 1회).

---

## 1. eval 파이프라인 버그 3종 ★

| # | 버그 | 증상 | 수정 | 커밋 |
|---|---|---|---|---|
| ① | **GT 타임존** (`form_question_jsons.py`) | CSV 정답(US/Pacific)을 머신 로컬(KST)로 파싱 → 캡션 창 밖 → 컨텍스트·정답이 항상 마지막 캡션(≈원점) | `TZ=America/Los_Angeles` 고정 + `searchsorted`+clamp | `9c49297` |
| ② | **저장 경로 `/`** (`eval.py`) | 모델명 `tt/Llama...`의 `/`가 경로로 해석 → 완주해도 저장 크래시(결과 유실) | 파일명 `/`→`_` sanitize | `9c49297` |
| ③ | **무한 재시도 / None 크래시** (`eval.py`) | `while True`가 3B의 null 답에 무한 재시도, `evaluate_output`이 None에 TypeError 폭주 | 재시도 캡(5회) + None→`unanswered` 처리 | `09fe687` |

①은 정답뿐 아니라 **모델이 보는 컨텍스트도** 오염(둘 다 같은 인덱스) → 수정 후 position/time GT 정상화.
②가 그동안 `out/`이 비어 있던 진짜 원인. ③으로 무한루프 없이 완주(perf ~26분).

---

## 2. 통합 8-way 비교 ★

`jung_results`의 모든 결과 파일을 하나의 채점 기준으로 재집계. 캡션·질문·eval 코드는 동일하고, 차이는
**답변 경로(Optimal+ working memory / ReMEmbR agent) + 모델 크기 + Llama 정밀도**다. 반복 실행은 구성별로 합쳐
문항 단위 평균을 냈으며, loss는 해당 필드에 유효하게 답한 문항만의 평균이다.
지표: **응답률**(유효 필드 낸 비율) / **정답률**(score_navqa 임계: position≤15m, time·duration≤2min, binary 정확일치,
미응답=오답 — NaVQA 논문 정의) / **loss**(답한 것만의 평균 오차; eval.py는 임계 판정 없이 raw 오차만 기록).

| 구성 | O-3B-perf | O-3B-acc | O-8B-perf | O-8B-acc | R-3B-perf | R-3B-acc | R-8B-perf | R-8B-acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **실행 수** | 5 | 5 | 3 | 3 | 1 | 2 | 1 | 1 |

`O` = Optimal+, `R` = ReMEmbR agent. 실행마다 non-text 25문항(position 12, binary 4, time 7, duration 2)을 평가했다.

### Overall (non-text 25문항)

| 지표 | O-3B-perf | O-3B-acc | O-8B-perf | O-8B-acc | R-3B-perf | R-3B-acc | R-8B-perf | R-8B-acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **응답률** | 70% | 57% | 85% | **92%** | 56% | 38% | 84% | 52% |
| **정답률**(임계) | 13% | 21% | **47%** | 37% | 0% | 2% | 28% | 8% |

### 유형별 (응답률 / loss / 정답률)

| 유형 | 지표 | O-3B-perf | O-3B-acc | O-8B-perf | O-8B-acc | R-3B-perf | R-3B-acc | R-8B-perf | R-8B-acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **position** (12/run) | 응답률 | 45% | 40% | 69% | 83% | **100%** | 67% | **100%** | 92% |
| | loss(m)↓ | 128.7 | 110.0 | **44.2** | 60.2 | 73.4 | 75.9 | 72.9 | 91.4 |
| | 정답률 | 0% | 3% | **33%** | 22% | 0% | 4% | 8% | 8% |
| **binary** (4/run) | 응답률 | 90% | 85% | **100%** | **100%** | 0% | 38% | 75% | 25% |
| | 정답률 | 60% | 65% | 67% | 58% | 0% | 0% | **75%** | 0% |
| **time** (7/run) | 응답률 | 91% | 69% | **100%** | **100%** | 14% | 0% | 86% | 14% |
| | loss(min)↓ | 13.5 | 6.3 | 4.7 | 3.9 | 3.7 | — | 1.9 | **0.2** |
| | 정답률 | 11% | 29% | **57%** | 48% | 0% | 0% | 43% | 14% |
| **duration** (2/run) | 응답률 | **100%** | 60% | **100%** | **100%** | 50% | 0% | 0% | 0% |
| | loss(min)↓ | 21.3 | 9.7 | 4.9 | **4.6** | 9.7 | — | — | — |
| | 정답률 | 0% | 10% | **50%** | **50%** | 0% | 0% | 0% | 0% |

`—`는 해당 유형에 유효 응답이 없어 loss를 계산할 수 없다는 뜻이다. 굵은 값은 행별 최고(오차는 최저)이며,
표시 반올림 때문에 원자료의 근소한 차이가 보이지 않을 수 있다.

### 세 축별 비교 요약

아래 변화량은 같은 agent·모델 크기·정밀도 조건을 한 쌍씩 맞춘 **Overall 기준 %p 차이**다. 실행 수가 다른 8개 구성을
한꺼번에 평균내지 않고, 비교하려는 축 하나만 바꿔 해석했다.

#### Performance → Accuracy

| 고정 조건 | 응답률 변화 | 정답률 변화 | 요약 |
|---|---:|---:|---|
| Optimal+ 3B | 70% → 57% (**-13%p**) | 13% → 21% (**+8%p**) | acc가 덜 답하지만 더 정확 |
| Optimal+ 8B | 85% → 92% (**+7%p**) | 47% → 37% (**-10%p**) | acc가 더 답하지만 임계 정답은 perf 우세 |
| ReMEmbR 3B | 56% → 38% (**-18%p**) | 0% → 2% (**+2%p**) | acc의 미세한 정답 이득보다 응답 손실이 큼 |
| ReMEmbR 8B | 84% → 52% (**-32%p**) | 28% → 8% (**-20%p**) | perf가 응답률·정답률 모두 우세 |

- **정밀도 모드의 효과는 일관되지 않다.** Optimal+ 3B에서만 acc가 품질 우세이고, Optimal+ 8B는 커버리지와 정확도가
  엇갈린다. ReMEmbR, 특히 8B에서는 perf가 명확히 유리하다.
- acc가 더 낮은 raw loss를 보인 유형도 있지만, 유효 응답 자체가 적으면 쉬운 일부 문항만 남은 선택 편향일 수 있다.
  따라서 모드 선택은 loss 단독보다 응답률과 임계 정답률을 함께 봐야 한다.

#### 3B → 8B

| 고정 조건 | 응답률 변화 | 정답률 변화 | 요약 |
|---|---:|---:|---|
| Optimal+ perf | 70% → 85% (**+15%p**) | 13% → 47% (**+34%p**) | 정답률 상승 폭이 전체 비교 중 최대 |
| Optimal+ acc | 57% → 92% (**+35%p**) | 21% → 37% (**+16%p**) | 응답률 상승 폭이 전체 비교 중 최대 |
| ReMEmbR perf | 56% → 84% (**+28%p**) | 0% → 28% (**+28%p**) | agent loop도 8B에서 처음 유효한 성능 |
| ReMEmbR acc | 38% → 52% (**+14%p**) | 2% → 8% (**+6%p**) | 개선되지만 절대 성능은 여전히 낮음 |

- **3B→8B는 네 쌍 모두 응답률과 정답률이 함께 상승한 유일한 축**이다. 모드 변경이나 agent 변경보다 방향이 가장
  일관적이며, 현재 결과에서 모델 용량이 가장 강한 성능 레버다.
- 특히 Optimal+ perf는 position 정답률 0%→33%, time 정답률 11%→57%로 올라 전체 정답률 개선을 주도했다.

#### ReMEmbR → Optimal+

| 고정 조건 | 응답률 변화 | 정답률 변화 | 요약 |
|---|---:|---:|---|
| 3B perf | 56% → 70% (**+14%p**) | 0% → 13% (**+13%p**) | Optimal+ 우세 |
| 3B acc | 38% → 57% (**+19%p**) | 2% → 21% (**+19%p**) | Optimal+ 우세 |
| 8B perf | 84% → 85% (**+1%p**) | 28% → 47% (**+19%p**) | 응답률은 대등, 정확도는 Optimal+ 우세 |
| 8B acc | 52% → 92% (**+40%p**) | 8% → 37% (**+29%p**) | Optimal+가 커버리지·정확도 모두 크게 우세 |

- **Overall에서는 네 쌍 모두 Optimal+가 우세**하다. ReMEmbR 8B-perf만 응답률이 84%로 Optimal+ 85%에 근접하지만,
  정답률은 19%p 낮다.
- ReMEmbR의 retrieval이 모든 유형에서 나쁜 것은 아니다. R-8B-perf는 position 응답률 100%, binary 정답률 75%로
  O-8B-perf(69%, 67%)보다 높다. 다만 time·duration 등 유형 간 출력 안정성이 부족해 Overall 이득으로 이어지지 못했다.

### 종합 정리

- **Optimal+ 3B — accuracy가 정확도 우세, performance가 커버리지 우세.** 정답률 acc 21%>perf 13%, loss도 acc가 전 유형에서
  낮음(pos 110<129, time 6.3<13.5, dur 9.7<21.3). 대신 응답률은 perf 70%>acc 57%. → **품질(정답률·loss)로는 acc가 낫다.**
- **Optimal+ 8B — perf/acc 대체로 대등(혼조), 확정적 승자 없음 + 분산 큼.**
  - 정답률(임계)은 **오히려 perf가 약간 앞섬(47% vs 37%)** → "acc 우세"는 성립 안 함.
  - loss는 갈림: position·binary는 **perf**(44<60, 67%>58%), time·duration은 **acc**(3.9<4.7, 4.6<4.9).
  - 응답률은 **8B에선 acc가 더 높음(92% vs 85%)** — 3B(perf 우위)와 반대. 8B는 둘 다 형식은 거의 다 채우고,
    차이는 position 응답률(acc 83%>perf 69%)에서 남. 단 acc는 많이 답해도 임계 통과율은 perf보다 낮음.
  - §4의 동일 설정 재실행이 **32%↔48%(±16%p)** 로 튐 → 3회로는 우열 확정 불가, **추가 테스트 필요.**
- **ReMEmbR는 8B-perf만 경쟁력 있는 신호를 보였다.** R-8B-perf는 응답률 84%, 정답률 28%로 ReMEmbR 네 구성 중
  가장 높지만 O-8B-perf(85%, 47%)에는 못 미친다. 반면 R-3B 두 구성은 정답률 0~2%, R-8B-acc는 8%로 낮다.
- **agent 종류와 무관하게 3B→8B 용량 증가가 가장 큰 레버다.** Optimal+ 정답률은 13~21%→37~47%, ReMEmbR는
  0~2%→8~28%로 올랐다. 다만 ReMEmbR는 실행 수가 대부분 1회라 run-to-run 분산을 분리할 수 없다.
- **ReMEmbR의 낮은 overall은 유형별 출력 불균형에서 온다.** position 응답률은 67~100%지만 time/binary/duration에서
  유효 필드를 자주 못 냈다. 특히 네 구성 모두 duration 정답이 없고, R-3B-acc는 time·duration 유효 응답이 0개다.

> ⚠️ Optimal+에는 3~5회 평균이 있지만 ReMEmbR는 R-3B-acc를 제외하면 각 1회뿐이다. 따라서 agent 간 격차는
> 현재 파일에서 관측된 결과이며, 확정적 우열로 일반화하려면 ReMEmbR 반복 실행이 더 필요하다.

---

## 3. Optimal+ 8B 심화 관찰 (position 도약이 종합점수 견인)

- **position에서 질적 도약.** 3B는 질문에 주어진 현재 좌표를 복창(오차 수십~수백 m)해 사실상 검색 실패였으나,
  8B-perf는 캡션 메모리에서 **과거 좌표를 실제로 꺼내옴**(노란 비상전화 0.003m, Gates 동상 0.0007m 등).
  position 임계 정답 3B ≈0 → 8B-perf 33%. 이게 종합 정답률 2배의 유일한 원동력.
- **binary는 오히려 퇴보.** 8B가 캡션에 과도하게 문자적으로 충실 → **부정 편향**(GT=yes를 "위치 안 맞음"으로
  과잉추론해 no). 3B-perf는 단순히 표지판 보고 yes로 맞힘. (단 binary 4문항 극소표본.)
- **time은 bimodal.** 일부는 거의 정확(0.28·1.09min), 동시에 세션 길이 초과하는 붕괴도 냄. 임계 통과는 늘지만
  평균은 여전히 불안정.
- **비용: 8B latency ~2배** (33~37s vs 15s).

**결론 — "병목은 정밀도가 아니라 용량" 가설 검증.** 3B→8B로 응답률·position 검색·종합정답이 뚜렷이 향상.
단 binary 부정편향·time 산술 붕괴는 크기만으로 안 풀림. ⚠️ seq0 25문항·유형당 2~12문항이라 방향성 신호로만.

---

## 4. 동시구동 & run-to-run 분산

- **동시구동(VLM+LLM)**: seq 2048(LLM)/1024(VLM), 12GB DRAM 기준 **3B+VLM은 정상 구동**(3B 측정은 VLM 켠 채),
  **8B+VLM은 서버 미기동**(12GB 초과 추정) → **8B는 VLM 끈 채로만 측정**. eval은 캡션 파일만 읽고 VLM을 호출
  안 하므로 정확도 비교엔 VLM 상태 무관.
- **run-to-run 분산**: 동일 8B-perf 설정 재실행이 Overall 정답 **32%↔48%** 로 튐(temperature 0.7 + 재시도
  무작위성). §2의 "소표본 → 강한 결론 금지"를 정면 실증.

---

## 5. ReMEmbR Agent 3B perf vs Optimal+ 3B perf (단일 run 사례)

`remembr+tt_Llama-3.2-3B-Instruct__captions_tt-qwen25-vl-3b_3_secs_remembr_perf.json`과
`optimal+tt_Llama-3.2-3B-Instruct__captions_tt-qwen25-vl-3b_3_secs_perf.json`을 직접 비교.
둘 다 같은 3B TT 모델 + 같은 캡션 파일을 썼지만, **optimal+는 working-memory 직접답변**이고
**ReMEmbR agent는 tool-call/retrieve/generate 루프**를 거친다.

### 핵심 수치

| 지표 | optimal+ 3B perf | remembr+ 3B perf |
|---|---|---|
| 전체 문항 수 | 30 | 30 |
| **응답 완료** | **25** | 19 |
| **미응답** | 5 | **11** |
| **응답률** | **83.3%** | 63.3% |
| **평균 elapsed** | **14.94s** | 100.71s |
| median elapsed | 15.58s | 72.28s |
| 평균 시도 횟수 | - | 3.04 |
| 최대 시도 횟수 | - | 5 |
| 평균 DB search latency | - | 0.0196s |
| 평균 LLM latency | - | 100.69s |

### loss 비교 (오차 계산 가능한 응답만 기준)

| 지표 | optimal+ 3B perf | remembr+ 3B perf |
|---|---|---|
| position loss | 144.15m (`n=7`) | **73.43m (`n=12`)** |
| time loss | 7.58min (`n=7`) | **3.67min (`n=1`)** |
| duration loss | 45.35min (`n=2`) | **9.7min (`n=1`)** |

> ⚠️ 이 loss는 **전체 30문항 평균이 아니라, 해당 오차를 실제로 계산할 수 있었던 응답만의 평균**이다.
> 따라서 응답률이 더 낮은 모델이 더 쉬운/적은 샘플에 대해서만 loss를 남겼다면 수치가 더 좋아 보일 수 있다.
> 실제로 remembr+는 응답률에서 뒤지지만, 남긴 응답들의 raw loss는 더 낮게 나타났다.

### 해석

- **현재 3B에서는 optimal+가 ReMEmbR agent보다 훨씬 안정적이고 빠르다.**
  응답률이 83.3% vs 63.3%로 **20%p 높고**, 평균 지연은 14.94s vs 100.71s로
  **약 6.7배 빠르다.**
- **병목은 retrieval이 아니라 LLM 루프**다. ReMEmbR agent의 DB 검색 시간은 평균 0.0196s에 불과하고,
  거의 모든 시간이 tool-call / generate / retry에 들어간다.
- **ReMEmbR agent는 출력 형식 안정성이 아직 낮다.**
  `optimal+`의 `type`은 주로 `text/position/time`에 머무르지만, ReMEmbR agent 결과에는
  `location`, `geolocation`, `minutes-ago`, `image description`, `answer` 등
  평가 스키마와 직접 대응되지 않는 type 라벨이 다수 섞였다.
- **loss만 보면 remembr+가 더 낮지만, 이는 응답률과 분리해서 보면 안 된다.**
  position은 remembr+가 12개 답변에 대해 평균 73.4m로 optimal+(7개, 144.1m)보다 낮았지만,
  time/duration은 각각 1개 샘플뿐이라 일반화하기 어렵다. 즉 **raw loss는 remembr+ 우세, 완주성·응답률은 optimal+ 우세**다.
- 첫 질문("When did you leave the building?")에서도 차이가 명확하다.
  `optimal+`는 `type='text', time=1.0`으로 scoring 가능한 답을 냈지만,
  ReMEmbR agent는 `type='geolocation', time='7 minutes-ago'`로 기록되어
  해당 run에서는 `unanswered` 처리됐다.

### 중간 결론

- **동일 3B TT 모델 + 동일 캡션 조건에서는 agent loop가 아직 이득을 못 만들었다.**
  retrieval을 여러 번 돌려도 응답률과 지연시간 모두 악화됐다.
- 지금 단계의 ReMEmbR agent 평가는 "완주 가능성 확인"에는 성공했지만,
  **성능상으로는 optimal+ 대비 열세**다.
- 따라서 이후 개선 우선순위는 retrieval 품질보다 **최종 출력 스키마 일관성, tool-call 안정성,
  재시도 축소** 쪽에 둬야 한다.

---

## 부록

### 참고 커밋
- `9c49297` fix(eval): 정답 타임존 + 저장 파일명 sanitize (버그 ①②)
- `09fe687` fix(eval): 재시도 캡 + unanswered 처리 (버그 ③)
- `1200fa0` test: Llama accuracy 모드 실행 (이후 performance로 되돌림)
- `0029849` qwen accuracy → performance / `b2dab02` 서버 temperature 존중(무한 재시도 1차 원인 해소)

### 참고 파일
- 수정: `scripts/question_scripts/form_question_jsons.py`, `scripts/eval.py`, `tt_server/tt_infer_server.py`
- 결과: `out/0/human_qa/jung_results/{optimal+,remembr+}eval/llm{3b,8b}models/*.json`
  (Optimal+ 3B 각 5회·8B 각 3회, ReMEmbR 3B-perf 1회·3B-acc 2회·8B 각 1회). 파일명 라벨 3.1/3.2 혼재 —
  실제 구분은 **agent+크기(3B/8B)+모드(acc/perf)**, 로드 모델은 서버 `TT_LLM_MODEL`이 결정하므로 결과엔 무관.
- 데이터: `data/questions/0/human_qa.json`(수정본), `human_qa.BROKEN-TZ.json`(옛 백업)
