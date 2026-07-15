# 개발 진행 상태 (DEV STATUS)

> **이 문서의 역할**: ReMEmbR × NPU 포팅 프로젝트의 **살아있는 추적 문서**. "지금 어디까지 됐고,
> 무엇이 검증됐고, 뭐가 남았고, 왜 그렇게 결정했는지"의 단일 소스.
> 작업할 때마다 아래 **① 상태 보드**와 **⑥ 변경 로그**를 갱신한다.
> 상세 절차/설계는 [문서 지도](#7-문서-지도)의 각 문서를 참조.

_최종 갱신: 2026-07-15 · 브랜치 `feat/tenstorrent-npu-backend`_

---

## 1. 상태 보드 (한눈에)

**목표**: ReMEmbR 파이프라인(캡션 → 벡터 메모리 → 질의응답)을 **BOS NPU** 위 3B 모델
(Qwen2.5-VL 캡셔너 + Llama-3.2-3B LLM)로 end-to-end 구동하고, 논문 NaVQA/CODa로 평가.

| 마일스톤 | 상태 |
|---|---|
| NPU 통합 추론 서버 (LLM+VLM, HTTP) | ✅ 완료 |
| NonAgent `tt/` 백엔드 + JSON 제약 디코딩 | ✅ 완료 |
| NPU 캡셔너 (Qwen2.5-VL) | ✅ 완료 |
| end-to-end 스모크 테스트 | ✅ 완료 |
| 원 논문 데이터 확보 (data.csv + CODa seq0) | ✅ 완료 |
| ROS-free CODa→.pkl 추출 | ✅ 완료 (8213 프레임) |
| 캡셔닝 파이프라인 (체크포인트/재개) | ✅ 완료 (full run에서 resume 동작 확인) |
| 서버 안정화 (clean shutdown + slowdown) | ✅ 완료 (slowdown 원인=Inspector, 실측 해결) |
| **seq0 캡션 생성 완료** | ✅ **272개 생성** (`captions_tt-qwen25-vl-3b_3_secs.json`, 6.1MB) |
| 질문 JSON 생성 (form_question_jsons) | ⬜ 다음 단계 |
| 평가 실행 (eval.py, optimal+tt/) | ⬜ 미착수 |
| 다중 프레임(논문 충실도) 캡션 | ⬜ 선택 과제 |

---

## 2. 지금 동작하는 것 (검증됨)

- **NPU 서버**: Llama-3.2-3B + Qwen2.5-VL-3B co-resident, 단일 워커+큐 직렬화, `/health` `/caption` `/v1/chat/completions`.
- **JSON 제약 디코딩**: NonAgent가 100% 파싱 가능한 JSON 응답, 재시도 루프 없음(~10–12s/질의).
- **캡셔닝**: 단일 CODa 프레임 → 정상 캡션(거부 없음), 3연속 요청 안정 확인.
- **데이터**: `data.csv`(210문항) + CODa seq0 `.pkl` 8213개(`/home/user/coda_raw/coda_data/0/`).
- **보드 복구**: wedged 상태에서 좀비 프로세스 정리 → 재기동으로 복구 성공(재부팅 없이).

## 3. 진행 중 / 다음 단계

1. **seq0 캡셔닝 완주** — 서버 재시작 후 max_gen_len·slowdown 수정 반영해 재실행(체크포인트 resume).
2. **질문 JSON 생성** → `form_question_jsons.py --caption_file captions_tt-qwen25-vl-3b_3_secs`.
3. **평가** → `eval.py --model optimal+tt/Llama-3.2-3B-Instruct --sequence_id 0 ...`.
4. (선택) **다중 프레임 캡션**으로 논문 충실도 향상 — 절차는 [PAPER_EVAL_RUNBOOK](PAPER_EVAL_RUNBOOK.md) 4-1.

## 4. 알려진 이슈 / 주의점 (gotchas)

| 이슈 | 상태 / 대응 |
|---|---|
| ~~**서버 slowdown**~~ (캡션당 21s→80s) | ✅ **해결**. 원인은 device 텐서가 아니라 **tt-metal Inspector**(디버그, 기본 ON)가 workload마다 YAML `flush()` → 캡션당 ~5천회 동기 I/O, YAML 153MB/534만 라인 누적 → flush가 점점 blocking. `run_server.sh`에서 `TT_METAL_INSPECTOR=0`으로 해결([8be8bc6]). 실측: 비전 19→2.1 it/s 하락이 **~42 it/s 안정**으로. |
| **캡션 문장 잘림** | 서버 기본 max_gen_len=128에서 잘림. 캡셔너가 256 전송하도록 수정([8caa7c9]). 재시작 후 적용. |
| **보드 wedged** | 원인=캡셔닝/컴파일 중 강제종료(특히 Ctrl-C 연타). shutdown 정리 로직 추가([f9bc7bf]). **`-9` 절대 금지.** 복구는 [PAPER_EVAL_RUNBOOK](PAPER_EVAL_RUNBOOK.md) 6번. |
| **재부팅 제약** | 이 PC는 NPU 먼저 켜고 부팅해야 인식됨. wedged 시 재부팅보다 프로세스 정리+재기동을 먼저 시도. |
| **device 노드 = `/dev/bos/0`** | Tenstorrent 아님(`/dev/tenstorrent` 없음). 확인: `fuser -v /dev/bos/0`, `lsmod \| grep bos`. |
| **3B 품질** | JSON 유효성은 보장되나 grounding은 3B 한계. 필요시 8B(DRAM 시간분리 폴백). |
| **충실도 편차** | 캡셔너 VILA→Qwen, LLM 8B→3B(이식 불가피), 프레임 6→1(절충). [PAPER_EVAL_RUNBOOK](PAPER_EVAL_RUNBOOK.md) 충실도 표. |
| **venv 의존성 함정** | 새 패키지가 `langchain-core`를 0.3, `numpy`를 2.x로 끌어올리면 스택이 깨짐. 핀 준수([RUNBOOK](../tt_server/RUNBOOK.md) 1-3). |

## 5. 핵심 설계 결정 (왜 그렇게 했나)

- **두 venv를 HTTP로 분리** — ttnn/transformers(서버) vs langchain/pymilvus(remembr) 버전 충돌 회피.
- **NonAgent 채택** (ReMEmbRAgent 아님) — 3B는 tool-calling 불안정. 단일 JSON 호출이 NPU에 적합.
- **JSON 제약 디코딩** — Ollama `format="json"` 대체. 3B의 무한 재시도/헛소리 방지.
- **캡셔너 = HTTP 클라이언트** — VLM을 remembr 프로세스에 직접 로드하지 않음(의존성/디바이스 소유 충돌).
- **`<video>` 제거, video→image 프롬프트** — VILA 잔재. Qwen 정지프레임에서 거부 유발.
- **ROS-free 추출기** — 원본 `preprocess_coda.py`의 ROS 의존은 시각화용뿐. 평문 pose+PNG만으로 동일 .pkl 생성.
- **체크포인트(JSONL append)** — 긴 캡셔닝(~70–95분)이 중단돼도 손실 없이 resume.
- **clean shutdown + teardown 중 신호 무시** — 워커가 daemon이라 device close가 스킵되던 문제 + Ctrl-C 연타 wedge 방지.

## 6. 변경 로그 (Changelog)

### 2026-07-15
- ✅ **seq0 캡셔닝 완주**: 272 캡션 생성(`captions_tt-qwen25-vl-3b_3_secs.json`). 체크포인트 resume 실동작 확인(중단 후 이어서 → 완료 시 partial 자동 삭제).
- `8be8bc6` **slowdown 원인 규명 & 해결**: tt-metal **Inspector**(디버그, 기본 ON)가 mesh workload마다 YAML `flush()` → 캡션당 ~5천회 동기 디스크 I/O, 한 run에 153MB/534만 라인 누적 → 파일이 커질수록 flush가 blocking → 비전 블록 **19→2.1 it/s 단조 하락**. `run_server.sh`에서 tt-metal 자체 스위치(`TT_METAL_INSPECTOR=0`, `_RPC=0`)로 비활성 — **tt-metal 소스 무수정**. 실측: **~42 it/s 안정 유지**, YAML 미생성. 함께 이전의 오진 수정(`describe()`의 `ttnn.deallocate`)을 되돌림.
- `f9bc7bf` **서버 안정화**: 종료 시 워커 join으로 `close_mesh_device()` 보장 + teardown 중 SIGINT/SIGTERM 무시(Ctrl-C 연타 wedge 방지).
- `804bcf6` **eval import**: `ChatNVIDIA`(NIM) lazy import → NIM 패키지 없이 eval 실행.
- `8caa7c9` **캡셔너**: import fallback(inner/outer 경로 양쪽), `max_gen_len`(기본 256) 전송으로 문장 잘림 해결.
- `16a253e` **캡셔닝 파이프라인**: `--captioner {vila,tenstorrent}` 선택 + VILA lazy import, 프롬프트 video→image 중화, JSONL 체크포인트/resume, early-exit를 "최종 JSON 존재"로 수정.
- `334c0b4` **ROS-free 추출기**: `preprocess_coda_rosfree.py` — pose txt + cam0 PNG → .pkl.
- `c5cc220` **문서**: PAPER_EVAL_RUNBOOK 신설, RUNBOOK에 langgraph/opencv 핀·shutdown 절차.
- (운영) 원 논문 데이터 확보: `data.csv` 다운로드, CODa seq0(42GB) 다운·압축해제·.pkl 8213개 추출.
- (운영) venv 의존성 정비: `langgraph==0.2.40`/`langsmith==0.1.147`, `numpy<2`/`opencv-python-headless<5` 설치(핀 충돌 회피).
- (진단) 보드 wedged 원인 규명: `/dev/tenstorrent` 아닌 **`/dev/bos`** 칩. 캡셔닝 중 강제종료가 원인.

### 2026-07-13
- `05fff77` VILA `<video>` 토큰 제거, 프레임별 캡션. 스모크 테스트 강화.
- `7d96cbc` LLM JSON 제약(grammar) 디코딩.
- `165e846` NonAgent NPU end-to-end 견고화.
- `8f9db90` end-to-end 스모크 테스트.
- `bb169e2` 통합 NPU 추론 서버(Llama+Qwen).
- `d29661c`/`ef0729d` NonAgent `tt/` 백엔드 + NPU 캡셔너.

---

## 7. 문서 지도

| 문서 | 내용 |
|---|---|
| **[DEV_STATUS.md](DEV_STATUS.md)** (이 문서) | 진행 상태·이슈·결정·변경 로그 (시작점) |
| [TENSTORRENT_INTEGRATION.md](TENSTORRENT_INTEGRATION.md) | 아키텍처·설계 결정·디버깅 로그 |
| [PAPER_EVAL_RUNBOOK.md](PAPER_EVAL_RUNBOOK.md) | 논문 NaVQA/CODa 평가 전체 절차 + 복구 |
| [../tt_server/README.md](../tt_server/README.md) | 서버 구조·요구사항·env |
| [../tt_server/RUNBOOK.md](../tt_server/RUNBOOK.md) | 서버 기동·venv 셋업·트러블슈팅·종료 |
