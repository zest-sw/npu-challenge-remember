# 작업 로그 — 2026-07-17~18: NPU eval 파이프라인 디버깅 & 3B/8B × perf/acc 4-way 비교

NPU 포트(Qwen2.5-VL 캡셔너 + Llama LLM)의 NaVQA eval을 **처음으로 유효하게 완주**시키고,
Llama를 **3B/8B × performance/accuracy 4개 구성**으로 공정 비교. eval이 그동안 결과를 못 남긴
원인들(정답 타임존, 저장 경로, 무한 재시도)을 모두 찾아 고쳤고, 네 구성의 결과를 확보.
7/18에 **8B(Llama-3.1-8B) 추가 측정** → "병목은 수치 정밀도가 아니라 모델 용량" 가설을 직접 검증
(결과: 가설 지지 — §3.5).

---

## 현재 상태 한눈에

- **4-way 비교 완료** — 3B/8B × perf/acc 네 결과 모두 동일 채점 방식(캡 eval)으로 확보 (§3, §3.5).
- **8B(Llama-3.1-8B) 측정 완료 (7/18)** — `TT_ENABLE_VLM=0 TT_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct`로
  서버 재기동해 perf/acc 측정. seq_len은 3B와 동일(1024) 유지 → 비교 공정.
- **핵심 결과**: 8B-perf가 종합정답 2배(4→8/25), **position 검색이 질적으로 살아남**(3B 0개→8B 4개 정답).
  단 binary는 퇴보(75%→25%), time은 여전히 불안정. "용량 병목" 가설 지지.
- **동시구동 (7/18)**: **3B+VLM은 seq 2048/1024로 12GB에 fit해 정상 구동**(3B 측정은 VLM 켠 채 수행).
  반면 **8B+VLM은 서버가 기동 못 함**(로딩 한참 무응답, 2회 시도) → 8B는 VLM 끈 채로만 측정 (§3.6).
  perf_test는 8B-perf 재실행으로 Overall 32%↔48% **run-to-run 분산**을 보여줌 (소표본 불안정 실증).
- eval 파이프라인 버그 3종 모두 처리 (타임존·저장경로·무한재시도).
- 서버: 3B는 VLM 켠 채, 8B는 VLM 끈 채 measured. 8B는 **performance 모드**(line 163). seq_len 2048/1024.
- 결과 파일: `out/0/human_qa/..._{perf,acc}.json` 4개 (라벨 3.1/3.2 혼재 — §3.5 주의).
- 커밋: `9c49297`(버그수정 2종), `1200fa0`(Llama accuracy 실험), `09fe687`(재시도 캡).

---

## 1. 완료된 작업

### 1-1. Qwen VLM performance 전환 + seq0 재캡셔닝 (7/16)
- 272 캡션(2h21m), `captions_tt-qwen25-vl-3b_3_secs.json`. 백업은 `.../json_folder/`.

### 1-2. eval 파이프라인 버그 3종 ★ 이번 세션 핵심

| # | 버그 | 증상 | 수정 | 상태 |
|---|---|---|---|---|
| ① | **GT 타임존** (`form_question_jsons.py`) | CSV 정답 timestamp(US/Pacific)를 머신 로컬(KST)로 파싱 → 캡션 창 밖 → 컨텍스트·정답이 항상 마지막 캡션(≈원점) | `TZ=America/Los_Angeles` 고정 + `searchsorted`+clamp | ✅ 커밋 `9c49297` |
| ② | **저장 경로 `/`** (`eval.py`) | 모델명 `tt/Llama...`의 `/`가 경로로 해석 → 완주해도 저장 크래시(결과 유실) | 파일명 `/`→`_` sanitize | ✅ 커밋 `9c49297` |
| ③ | **무한 재시도 / None 크래시** (`eval.py`) | `while True`가 3B의 null 답에 무한 재시도, `evaluate_output`이 None에 TypeError/NameError 폭주 | 재시도 캡(5회) + None을 `unanswered`로 처리 | ✅ (이 커밋에 포함) |

- ①은 정답뿐 아니라 **모델이 보는 컨텍스트도** 오염시켜서(둘 다 같은 인덱스 사용), 수정 후 재시도가
  크게 줄고 position/time GT가 실제 좌표로 정상화됨.
- ②는 **지금까지 `out/`이 비어 있던 진짜 원인** (temperature 버그와 별개).
- ③으로 eval이 무한루프 없이 완주(perf 1h10m → 캡 적용 후 ~26분), None 문항은 `unanswered`로
  분리 집계 가능.

### 1-3. Llama accuracy 실험 준비 (커밋 `1200fa0`, 이후 perf 재실행 위해 되돌림)
- `tt_infer_server.py:163`을 `optimizations=None`(→accuracy)로 두고 서버 재시작 → accuracy로 `_acc` 실행.
- 이후 line 163을 performance 람다로 복원 + 서버 재시작 → 같은 캡 eval로 `_perf` 실행.

---

## 2. §3 비교의 채점 방식 (캡 eval)

두 실행 모두 **재시도 캡 + `unanswered` 분리** 방식으로 채점:
- 유형별로 **응답률**(모델이 유효 필드를 낸 비율)과 **답한 것의 오차**를 분리 보고.
- None/`'null'`/산문 답은 `unanswered`로 오차 평균에서 제외 (옛 no-cap 방식은 재시도 끝에 쓰레기
  좌표로 채점해 오차를 왜곡했음).
- 유일한 차이는 **Llama 정밀도(performance vs accuracy)** — 캡션·질문·서버구성·eval코드 동일.

---

## 3. performance vs accuracy 비교 결과 ★

| 유형 (문항수) | performance | accuracy |
|---|---|---|
| **position** (12) | 응답 **58%**, 오차 144.1 | 응답 25%, 오차 116.0 |
| **binary** (4) | 응답 100%, **정답률 75%** | 응답 100%, 정답률 50% |
| **time** (7) | 응답 **100%**, 오차 7.6 | 응답 57%, 오차 5.4 |
| **duration** (2) | 응답 100%, 오차 45.4 | 응답 50%, 오차 0.3 |
| text (5) | 스킵 (자동 채점 불가) | 스킵 |

### 결론 — accuracy 정밀도는 명확한 이득 없음 (오히려 응답률 하락)
- **응답률은 performance가 전반적으로 높음** (position 58%↔25%, time 100%↔57%). accuracy가 유효 답을
  더 적게 냄.
- **binary(가장 깨끗한 이산 지표)는 performance 우세** (75%↔50%).
- accuracy의 "오차"가 낮아 보이는 건 **함정** — 답한 문항이 적어(더 쉬운 부분집합) 생긴 것으로,
  performance와 **답한 문항이 달라 오차 직접 비교 불가**.

### ⚠️ 통계적 한계 (강한 결론 금지)
- **표본 극소**: 유형당 2~12문항 (duration은 1~2개). 노이즈 수준.
- **확률적 요소**: temperature 0.7 + 5회 캡 → "답했나/못했나"가 부분적으로 운. 차이의 상당수가 무작위.
- **seq0 25문항만**: 신뢰할 결론엔 시퀀스·문항이 훨씬 많이 필요.
→ 정직한 결론은 **"accuracy가 확실히 나쁘다"가 아니라 "명확한 이득을 못 줬다"**.

### 시사점 — 병목은 "정밀도"가 아니라 "용량(3B)"
- 정밀도(bfp4→bf16)를 올려도 안 나아짐 → Llama의 **수치 정밀도가 아니라 근본 능력(3B 크기)** 이 병목일
  가능성. 다음 레버는 accuracy가 아니라 **8B**(용량↑).
- 3B의 주된 position 실패는 **"좌표를 아예 안 냄"(형식, 응답률 25~58%)** 이 1차, "낸 것도 부정확"이 2차.

---

## 3.5. ★ 8B 추가 측정 — 3B vs 8B (4-way, 7/18)

8B(Llama-3.1-8B)를 `TT_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct`로 서버 재기동해 같은 캡션·질문·
eval 코드·**seq_len 2048(LLM)/1024(VLM)**로 perf/acc 측정. 3B와의 차이는 모델 크기 + **VLM 상태**:
**3B는 VLM 켠 채, 8B는 VLM 끈 채**(8B+VLM은 12GB에 안 들어가 동시구동 불가 — §3.6). 단 eval은 캡션
파일만 읽고 VLM을 호출하지 않으므로 **정확도 비교엔 VLM 상태가 무관**(VLM이 LLM 답변에 관여 안 함).

### 4-way 정량 (seq0, non-text 25문항, 캡 eval)

| 지표 | 3B-acc | 3B-perf | 8B-acc | 8B-perf |
|---|---|---|---|---|
| **응답률** | 48% | 80% | 92% | **92%** |
| **Overall 정답**(15m/2min) | 5/25 (20%) | 4/25 (16%) | 5/25 (20%) | **8/25 (32%)** |
| position 오차(m) / 응답수 | 116.0 / 3 | 144.1 / 7 | 71.5 / 10 | **68.4 / 10** |
| time 오차(min) / 응답수 | 5.42 / 4 | 7.58 / 7 | **3.02** / 7 | 7.92 / 7 |
| duration 오차(min) / 응답수 | **0.30** / 1 | 45.35 / 2 | 4.64 / 2 | 6.63 / 2 |
| binary 정답률 | 50% (2/4) | **75% (3/4)** | 25% (1/4) | 25% (1/4) |
| 평균 latency | 15.3s | 14.9s | 37.0s | 33.3s |

**Overall 정답의 유형별 내역** (무엇을 맞혀서 점수가 났나 — 이게 핵심):
- 3B-perf 4점 = binary 3 + time 1  (**position 0**)
- 8B-perf 8점 = **position 4** + time 3 + binary 1

### 핵심 발견

1. **position에서 질적 도약 — 8B-perf가 "진짜 기억 검색"을 함.** 3B는 거의 전부 질문에 주어진
   현재 좌표를 복창(오차 수십~수백 m)해 사실상 검색 실패였는데, 8B-perf는 캡션 메모리에서 과거
   좌표를 실제로 꺼내옴:
   - 노란 비상전화 기둥: 오차 **0.003m** (3B는 현재좌표 복창으로 77m)
   - Gates 동상(MEDIUM): 오차 **0.0007m**
   - 앉을 곳(MEDIUM): 오차 **0.0004m**
   - 안 빈 자전거 거치대: 오차 **2.1m**
   position Overall 정답이 **3B 0개 → 8B-perf 4개**. 이게 종합점수 2배(4→8)의 유일한 원동력.

2. **binary는 오히려 퇴보 (perf 75%→25%).** 8B가 캡션에 과도하게 문자 그대로 충실해져 **부정 편향**
   발생 — GT가 yes인 3문항 중 2개(자전거탄사람·로딩존)를 "현재 위치가 표지판 위치와 안 맞는다"는
   식으로 과잉추론해 no로 틀림. 3B-perf는 단순하게 표지판 보고 yes로 맞혔음. (단 binary 4문항 극소표본.)

3. **time은 bimodal — 평균은 안 좋아짐.** 8B-perf는 일부를 거의 정확히 맞히면서(빨간소파 0.28,
   벽돌길 1.09) 동시에 세션 길이를 넘는 붕괴도 냄(건물 첫진입 "44분 전"=오차 40, 체류 "11.35분"인데
   세션은 11분). "현재시각 − 과거시각" 산술이 크기를 키워도 여전히 불안정. threshold 통과 수는 늘었지만
   (1→3) 평균 오차는 3B와 비슷.

4. **acc vs perf: 3B든 8B든 perf 우위 재확인.** 8B에서도 Overall perf 32% > acc 20%. 특히 position
   검색이 perf(4개 정답)에서 acc(0개)보다 크게 나음. "정밀도(acc)는 이득 없다"가 8B로도 유지 — 8B-acc는
   position에서 8B-perf만큼 좌표를 꺼내오지 못하고 복창/무응답으로 회귀.

5. **비용: 8B latency ~2배** (33~37s vs 15s). 정확도 향상의 대가.

### 결론 — "용량 병목" 가설 검증됨 (단, 크기가 만병통치는 아님)
- 3B→8B로 **응답률(80→92%)·position 검색·종합정답(4→8)이 뚜렷이 향상** → WORKLOG의 "병목은 수치
  정밀도가 아니라 모델 용량"이 **직접 확인**됨. 특히 죽어 있던 공간 기억 검색(position)이 살아난 게 결정적.
- 다만 **binary 부정편향·time 산술 붕괴**는 8B에서도 남거나 악화 → 크기만으로 다 해결되진 않음.
  다음 레버는 (a) 시간 산술을 프롬프트/후처리로 보조, (b) binary 프롬프트 재설계.
- ⚠️ **표본 한계 그대로**: seq0 25문항, 유형당 2~10개(binary·duration은 2~4). 방향성 신호일 뿐 강한 결론 금지.

### 파일명 라벨 주의
`out/`의 파일명이 `Llama-3.1-3B`·`Llama-3.2-8B`로 뒤섞여 있음(3.2엔 8B 없고 3.1엔 3B 없음 → `--model`
라벨 오타). 실제 구분은 **크기(3B/8B) + 모드(acc/perf)**이며, 실제 로드 모델은 서버의 `TT_LLM_MODEL`이
결정하므로 결과 자체엔 영향 없음. 다음에 재실행 시 `--model` 라벨을 3B는 3.2, 8B는 3.1로 맞출 것.

---

## 3.6. ★ 동시구동(VLM+LLM) 가능 여부 & perf 재실행 (7/18)

### 동시구동 — 3B는 되고, 8B는 안 됨 (메모리)
서버는 LLM+VLM을 한 프로세스에 함께 올림(TT_ENABLE_VLM=1 기본). seq 2048(LLM)/1024(VLM), 12GB DRAM 기준:
- **3B + VLM: 정상 구동** ✅ — 둘이 함께 올라가 캡셔닝·eval 완주. (7/17 3B 측정이 실제로 이 구성 = VLM 켠 채.)
- **8B + VLM: 서버 기동 실패** ❌ — 더 작은 3B였다면 진작 떴을 시간을 **한참 기다려도 `/health` 무응답**
  (2회 시도, 프로세스는 살아있으나 로딩 단계에서 6분+ 진행 안 됨 = crash가 아닌 hang/극저속).
  OOM 로그를 직접 캡처하진 못했으나, **3B는 되고 8B만 안 되는** 정황상 **8B+VLM이 seq 2048/1024로는
  12GB DRAM에 안 들어가는 것**으로 판단.
- → 그래서 **8B는 VLM을 끈 채로만 측정 가능**했음 (§3.5의 8B perf/acc/perf_test 전부 VLM off).
  8B에서 VLM 켠 동시구동을 하려면 seq_len을 낮춰야 하는데, 그러면 8B-perf(seq 2048)와 컨텍스트 조건이
  달라져 정확도 비교가 오염됨 → 별도 설계 필요(미결).

### perf vs perf_test — 동일 8B/perf 설정 재실행 (run-to-run 분산)
`perf_test`는 8B-performance를 한 번 더 돌린 것 (perf와 같은 8B/perf/VLM off/seq2048).

| 지표 | perf | perf_test |
|---|---|---|
| 총 LLM 추론 | 833s | 829s (**0.99x** — 동일 설정이라 당연) |
| **Overall 정답/25** | 8 (32%) | **12 (48%)** |
| position 오차(m)/n | 68.4 / 10 | 26.7 / 9 |
| binary | 1/4 | 2/4 |
| latency | 33.3s | 33.2s |

- **정확도 32%↔48%는 순수 run-to-run 분산** — 사실상 동일 설정을 두 번 돌렸는데 ±16%p 튐 (temperature
  0.7 + 재시도 무작위성). → §3.5의 "표본 극소 → 강한 결론 금지"를 **정면 실증**.
- 8B-perf 중심 추정은 두 run 평균 (8+12)/2 ≈ **40%**. 3B→8B 비교(4→8)도 이 분산폭 감안 시 "방향은 맞지만
  크기 과신 금물"(단 두 8B run 모두 3B의 4점보다 높아 용량 이득 자체는 유지).

---

## 4. 미결 / 보류 항목

| 항목 | 상태 | 비고 |
|---|---|---|
| **git push** | ⬜ 막힘 | remote가 HTTPS인데 `credential.helper` 없음 → GitHub 인증 필요 (gh/PAT/SSH) |
| **tt-metal fork 커밋** | ⬜ | `demo_qwen25_vl.py`(Qwen optimizations 파라미터) 미커밋. 서버가 의존하는 load-bearing 변경이라 로컬 커밋 권장 |
| **line 163 상태** | ℹ️ | 현재 performance로 복원됨. accuracy 실험 커밋(`1200fa0`)과 현재 코드가 다름(의도된 되돌림) |
| **응답률 별도 지표화** | 🅿️ | eval.py 콘솔 러닝 지표엔 unanswered 카운트 미출력. 최종 분석은 저장 JSON의 `unanswered` 플래그로 산출 중 |
| **theta = 3.14 더미** | ⬜ | 회전 미구현(placeholder). eval엔 영향 없음 |

---

## 5. 앞으로 할 일 (순서)

1. **재시도 캡 + WORKLOG 커밋** (이 문서 작성 직후).
2. **git push 인증 설정** (gh auth login / PAT / SSH 중 택) → 대기 커밋 push.
3. ✅ **8B 측정 완료 (7/18)** — §3.5. "용량 병목" 가설 검증됨. 다음 레버는 (a) **time 산술 보조**
   (프롬프트에 "현재시각−과거시각" 계산을 명시/후처리), (b) **binary 프롬프트 재설계**(8B 부정편향 완화).
3b. ⬜ **8B+VLM 동시구동** — seq 2048/1024로는 12GB 초과로 서버 미기동(§3.6). 시도하려면 `TT_LLM_MAX_SEQ_LEN`을
   낮춰 fit시켜야 하나, 그러면 8B-perf(seq 2048)와 컨텍스트가 달라져 비교 오염 → seq 맞춘 대조 설계 필요.
   (당장은 캡셔닝→eval 분리 워크플로라 8B 동시구동이 꼭 필요하진 않음.)
4. **(선택) 더 많은 시퀀스/문항** — seq0 25문항은 통계적으로 약함. 8B의 position 도약·binary 퇴보가
   진짜인지 확인하려면 다른 시퀀스 캡션·평가 필요 (특히 binary·duration은 유형당 2~4문항이라 노이즈).
5. **(향후) 라이브 파이프라인 배선** — chat_demo/nova_carter 백엔드를 `tt/`로 (현재 VILA·GPU).

---

## 부록

### 참고 커밋
- `9c49297` fix(eval): correct answer-timestamp timezone and sanitize result filename (버그 ①②)
- `1200fa0` test: run Llama in accuracy mode (perf/acc comparison) — 이후 performance로 되돌림
- (이 커밋) fix(eval): cap retries and mark unanswered (버그 ③)
- `0029849` qwen accuracy → performance (7/16)
- `b2dab02` 서버 temperature 존중 → eval 무한 재시도(1차 원인) 해소

### 참고 파일
- 수정: `scripts/question_scripts/form_question_jsons.py`, `scripts/eval.py`, `tt_server/tt_infer_server.py`
- 결과(5개): `out/0/human_qa/optimal+tt_Llama-3.{1,2}-{3B,8B}-Instruct__captions_tt-qwen25-vl-3b_3_secs_{acc,perf,perf_test}.json`
  - 3B-acc / 3B-perf (7/17, **VLM on**), 8B-acc / 8B-perf / 8B-perf_test (7/18, **VLM off**)
  - 라벨은 3B=3.2, 8B=3.1로 정리 완료. `perf_test`는 8B-perf 재실행(§3.6, run-to-run 분산 확인용).
- 데이터: `data/questions/0/human_qa.json`(수정본), `human_qa.BROKEN-TZ.json`(옛 백업)
