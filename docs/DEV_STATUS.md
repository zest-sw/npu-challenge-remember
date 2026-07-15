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
| 서버 안정화 (clean shutdown) | ✅ 완료 |
| slowdown 해결 ① Inspector | ✅ 완료 (`TT_METAL_INSPECTOR=0`) — 19→2.1 it/s 하락 제거 |
| slowdown 해결 ② Tracy 빌드 | 🔄 tt-metal 재빌드 중 (`ENABLE_TRACY=OFF`) — RSS +33MB/요청 누수 제거 목적 |
| **seq0 캡션 생성 완료** | ✅ **272개 생성** (`captions_tt-qwen25-vl-3b_3_secs.json`, 6.1MB) |
| **질문 JSON 생성** (form_question_jsons) | ✅ **30문항** (`data/questions/0/human_qa.json`) |
| 서버 temperature 지원 | ✅ 완료 (`b2dab02`) — eval 무한 재시도 해소. 실측: temp=0 동일 / 0.7 매번 다름 |
| **캡션 재생성** (품질 일관화) | ⏸️ 51/272에서 대기 — Tracy 재빌드 후 재개(체크포인트 보존) |
| **평가 실행** (eval.py, optimal+tt/) | ⬜ 캡션 재생성 후 |
| 다중 프레임(논문 충실도) 캡션 | ⬜ 선택 과제 |

### 하드웨어 실측 스펙 (`tt_server/npu_info.py`)

| 항목 | 값 |
|---|---|
| arch / devices | `BLACKHOLE` / 1개 (`/dev/bos/0`, 모듈 `bos` 2.3.0) |
| **DRAM** | **12 GiB** = 4 GiB × **3 뱅크** (`dram_grid_size (3,1)`) |
| Tensix 연산 그리드 | **5×4 = 20 코어** (harvesting mask `0x0` = 전 코어 정상) |
| L1 / worker 코어 | 1.5 MB (eth 코어 512 KB) |
| PCIe | `0000:01:00.0` Co-processor (Synopsys DWC) |

> **12 GiB**가 "3B+3B는 co-resident 가능, 8B는 불가"의 근거다. 평가 단계에서 8B를 쓰려면
> 단계별 단일 모델 로딩(`TT_ENABLE_VLM=0`)이 필요하다.

---

## 2. 지금 동작하는 것 (검증됨)

- **NPU 서버**: Llama-3.2-3B + Qwen2.5-VL-3B co-resident, 단일 워커+큐 직렬화, `/health` `/caption` `/v1/chat/completions`.
- **JSON 제약 디코딩**: NonAgent가 100% 파싱 가능한 JSON 응답, 재시도 루프 없음(~10–12s/질의).
- **캡셔닝**: 단일 CODa 프레임 → 정상 캡션(거부 없음), 3연속 요청 안정 확인.
- **데이터**: `data.csv`(210문항) + CODa seq0 `.pkl` 8213개(`/home/user/coda_raw/coda_data/0/`).
- **보드 복구**: wedged 상태에서 좀비 프로세스 정리 → 재기동으로 복구 성공(재부팅 없이).

## 3. 진행 중 / 다음 단계

1. 🔄 **캡션 재생성 완주 대기** — 전부 256토큰으로 통일(~2.9h). 완료 시 최종 JSON 생성 + partial 자동 삭제.
2. **질문 JSON 재생성** — 캡션이 바뀌었으므로 `form_question_jsons.py`를 다시 돌려 optimal 컨텍스트를 갱신해야 한다.
3. **평가 실행** — `eval.py --model optimal+tt/Llama-3.2-3B-Instruct --sequence_id 0 ...` (temperature 기본 0.7).
4. (선택) **다중 프레임 캡션**으로 논문 충실도 향상 — 절차는 [PAPER_EVAL_RUNBOOK](PAPER_EVAL_RUNBOOK.md) 4-1.

## 4. 알려진 이슈 / 주의점 (gotchas)

| 이슈 | 상태 / 대응 |
|---|---|
| **서버 slowdown** (원인 2개) | **① tt-metal Inspector**(디버그, 기본 ON): workload마다 YAML `flush()` → 캡션당 ~5천회 동기 I/O, 153MB/534만 라인 누적 → 19→2.1 it/s. `TT_METAL_INSPECTOR=0`으로 해결([8be8bc6]).<br>**② Tracy 프로파일링 빌드**: Inspector를 끄니 드러남(84→50 it/s). 동일 이미지 12회에 **RSS +33MB/요청, 지연 +1.0초/요청 선형 증가**. `build -> build_Release_tracy`(`ENABLE_TRACY=ON`)라 Tracy가 zone을 계속 누적. **컴파일 타임 스위치라 env로 못 끔** → 기본 빌드(`./build_metal.sh`, TRACY OFF)로 교체 중. 롤백: `ln -nsf build_Release_tracy build`. |
| **캡션 문장 잘림** | 서버 기본 max_gen_len=128에서 잘림. 캡셔너가 256 전송하도록 수정([8caa7c9]). 재시작 후 적용. |
| **보드 wedged** | 원인=캡셔닝/컴파일 중 강제종료(특히 Ctrl-C 연타). shutdown 정리 로직 추가([f9bc7bf]). **`-9` 절대 금지.** 복구는 [PAPER_EVAL_RUNBOOK](PAPER_EVAL_RUNBOOK.md) 6번. |
| **재부팅 제약** | 이 PC는 NPU 먼저 켜고 부팅해야 인식됨. wedged 시 재부팅보다 프로세스 정리+재기동을 먼저 시도. |
| **device 노드 = `/dev/bos/0`** | Tenstorrent 아님(`/dev/tenstorrent` 없음). 확인: `fuser -v /dev/bos/0`, `lsmod \| grep bos`. |
| ~~**eval 무한 재시도**~~ | ✅ **해결**([b2dab02]). 원인=서버가 temperature를 무시하고 greedy → eval의 재시도가 매번 동일한 답. 서버가 OpenAI `temperature`를 존중하도록 수정(0이면 greedy 유지, >0이면 grammar 허용 토큰에서 샘플링). 트리거였던 "3B가 time 질문에 `time=None`" 자체는 남아 있으나, 이제 재시도가 진전한다. |
| **3B 품질** | JSON 유효성은 보장되나 grounding은 3B 한계 — 질문 type에 맞는 필드를 못 채우는 사례 있음(재시도로 흡수). **8B로 올리지 않는 이유는 5번 설계 결정 참고**(로봇 배포 시 co-resident 필요). |
| ~~**캡션 품질 혼합**~~ | 🔄 재생성 중으로 해소 예정. (기존 혼합본은 `*.MIXED-BACKUP.json`으로 보관) |
| **충실도 편차** | 캡셔너 VILA→Qwen, LLM 8B→3B(이식 불가피), 프레임 6→1(절충). [PAPER_EVAL_RUNBOOK](PAPER_EVAL_RUNBOOK.md) 충실도 표. |
| **venv 의존성 함정** | 새 패키지가 `langchain-core`를 0.3, `numpy`를 2.x로 끌어올리면 스택이 깨짐. 핀 준수([RUNBOOK](../tt_server/RUNBOOK.md) 1-3). |

## 5. 핵심 설계 결정 (왜 그렇게 했나)

- **두 venv를 HTTP로 분리** — ttnn/transformers(서버) vs langchain/pymilvus(remembr) 버전 충돌 회피.
- **NonAgent 채택** (ReMEmbRAgent 아님) — 3B는 tool-calling 불안정. 단일 JSON 호출이 NPU에 적합.
- **LLM은 3B 유지 (8B로 안 감)** — 평가만 보면 평가 단계엔 VLM이 필요 없으니 `TT_ENABLE_VLM=0`으로
  8B를 단독 로딩해 품질을 올릴 수 있다(DRAM 12GiB에 8B는 단독으로만 들어감). **그러나 최종 목표는
  로봇 배포**이고, 주행 중에는 캡션 생성과 질의응답이 함께 일어나므로 **두 모델이 co-resident여야 한다.**
  8B+VLM은 12GiB에 안 들어간다 → **3B+3B가 배포 가능한 유일한 구성.** 시간분리(time-separated)는
  평가 전용 최적화일 뿐 실배포에 못 쓰므로, **평가도 배포와 같은 3B 구성으로** 한다(평가의 대표성 확보).
- **서버가 temperature를 존중** — 원본 eval은 `temperature=0.7`(기본값)로 재시도하면 다른 답이 나온다는
  전제로 재시도 루프를 돌린다. 서버가 greedy 고정이면 그 계약이 깨져 무한 루프가 된다. eval 코드를
  고치는 대신 **서버가 계약을 지키게** 했다(temperature=0이면 기존 greedy 그대로).
- **JSON 제약 디코딩** — Ollama `format="json"` 대체. 3B의 무한 재시도/헛소리 방지.
- **캡셔너 = HTTP 클라이언트** — VLM을 remembr 프로세스에 직접 로드하지 않음(의존성/디바이스 소유 충돌).
- **`<video>` 제거, video→image 프롬프트** — VILA 잔재. Qwen 정지프레임에서 거부 유발.
- **ROS-free 추출기** — 원본 `preprocess_coda.py`의 ROS 의존은 시각화용뿐. 평문 pose+PNG만으로 동일 .pkl 생성.
- **체크포인트(JSONL append)** — 긴 캡셔닝(~70–95분)이 중단돼도 손실 없이 resume.
- **clean shutdown + teardown 중 신호 무시** — 워커가 daemon이라 device close가 스킵되던 문제 + Ctrl-C 연타 wedge 방지.

## 6. 변경 로그 (Changelog)

### 2026-07-15
- `b2dab02` **서버가 temperature 존중 → eval 무한 재시도 해소**: `chat_completions`가 `temperature`를
  읽지 않고, JSON 경로는 masked argmax, non-JSON은 `SamplingParams(temperature=0.0)`로 **greedy 고정**이었다.
  원본 eval은 `--temperature 0.7`로 재시도하면 답이 달라진다는 전제라, greedy면 재시도가 무한 반복된다.
  temperature를 `chat_completions → generate() → sampler`로 전달하고 `_masked_argmax`→`_masked_sample`
  (temp≤0=기존 greedy, >0=grammar 허용 토큰에서 샘플링). grammar 제약은 그대로라 JSON 유효성 무영향.
  **실측**: temp=0 → 3회 요청 고유응답 1개, temp=0.7 → 3개(둘 다 schema 유효).
- 🔄 **캡션 재생성 착수**: 혼합 품질(128/256 토큰 섞임) 해소 위해 전량 재생성. 기존본은 `*.MIXED-BACKUP.json` 보관.
  초기 실측: 길이 \~1150자로 일관, 비전 **83\~182 it/s**, 캡션당 VisionBlock 1회 — 그간 수정들이 모두 효과 확인.
- `8aea788` **eval optimal 경로 버그 수정**: `TextMemory.memory_to_string`이 이미 문자열인 optimal
  컨텍스트를 문자 단위로 순회해 `doc.time`에서 터지던 것(→ eval 재시도 무한루프) 수정. 문자열이면 그대로 통과.
- ✅ **질문 JSON 생성**: `human_qa.json` 30문항(position 12 / time 7 / text 5 / binary 4 / duration 2).
  eval은 text를 skip하므로 **25문항 채점** 대상. 나머지 seq는 캡션이 없어 자동 skip.
- ⛔ **평가 착수 → 막힘**: 서버가 temperature를 무시(greedy)해 eval.py 재시도 루프가 무한 반복. 방향 결정 대기.
- `9fe023c` **npu_info 수정**: 추측했던 device 메서드가 실제로 없어 전부 `(n/a)`였던 것 → `open_mesh_device`와
  MeshDevice 실 API만 사용. DRAM 크기는 MeshDevice에 API가 없어 **SoC descriptor YAML**에서 읽음.
  ttnn의 loguru+C++ 로그를 **fd 리다이렉트로 억제**(`-v`로 볼 수 있음). → **실측 스펙 확보(DRAM 12GiB 등)**.
- `4f81c2e` **중복 prefill 제거**: 데모(`vision_demo.py`)가 `compile_prefill` + `inference_prefill`로
  **의도적으로 prefill을 2번** 하는데(컴파일/추론 시간 분리 측정용 = 벤치마크 장치), 서버는 첫 요청 이후
  캐시가 warm이라 두 번째가 순수 중복이었다. 첫 결과는 VLM에선 미사용 `masks`에, LLM에선 **변수에 받지도 않음**.
  VLM·LLM 둘 다 1회로. 실측: 캡션당 VisionBlock 2회→**1회**, ~76s→**~38s**.
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
