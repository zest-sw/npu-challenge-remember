# 작업 로그 — 2026-07-17: NPU eval 파이프라인 디버깅 & performance/accuracy 비교

NPU 포트(Qwen2.5-VL 캡셔너 + Llama-3.2-3B LLM)의 NaVQA eval을 **처음으로 유효하게 완주**시키고,
Llama 정밀도(performance vs accuracy)를 공정 비교. eval이 그동안 결과를 못 남긴 원인들(정답
타임존, 저장 경로, 무한 재시도)을 모두 찾아 고쳤고, 두 정밀도 모드의 결과를 확보.

---

## 현재 상태 한눈에

- **perf/acc 비교 완료** — 두 결과 모두 동일 채점 방식(캡 eval)으로 확보 (아래 §3).
- eval 파이프라인 버그 3종 모두 처리 (타임존·저장경로·무한재시도).
- 서버: **performance 모드**로 up (perf 재실행 위해 line 163을 performance로 복원).
- 결과 파일: `out/0/human_qa/..._perf.json`, `..._acc.json`.
- 커밋: `9c49297`(버그수정 2종), `1200fa0`(Llama accuracy 실험). 재시도 캡은 이 커밋과 함께 추가.

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
3. **(선택) 8B 참고 측정** — "용량 병목" 가설 검증. `TT_ENABLE_VLM=0` + `TT_LLM_MODEL=<8B>`로 서버
   재시작(배포 불가·eval 전용 구성). 8B 가중치 로컬 존재 확인 선행.
4. **(선택) 더 많은 시퀀스/문항** — seq0 25문항은 통계적으로 약함. 신뢰도 높이려면 다른 시퀀스 캡션·평가.
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
- 결과: `out/0/human_qa/optimal+tt_Llama-3.2-3B-Instruct__captions_tt-qwen25-vl-3b_3_secs_{perf,acc}.json`
- 데이터: `data/questions/0/human_qa.json`(수정본), `human_qa.BROKEN-TZ.json`(옛 백업)
