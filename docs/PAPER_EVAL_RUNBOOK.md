# 원본 논문 데이터(NaVQA/CODa) 테스트 실행 가이드 — NPU

ReMEmbR 논문의 **NaVQA benchmark**를 우리 Tenstorrent NPU 스택(Qwen2.5-VL captioner + Llama-3.2-3B LLM)에서
실행하는 전체 절차. 데이터 수집 → 전처리 → captioning → 질문 생성 → 평가까지 한 문서에 담았다.

> ⚠️ **먼저 5장 "NPU 안전 수칙"을 읽어라.** captioning 도중 잘못된 방식으로 서버를 죽이면
> 보드가 먹통이 되어 **재부팅이 필요**해진다. 실제로 한 번 우리에게 일어났던 일이다.

---

## 0. 파이프라인 한눈에 보기

```
CODa seq .zip ──(selective unzip)──► 2d_rect/cam0 + poses/dense
      │                                   │
      │                    preprocess_coda_rosfree.py (no ROS needed)
      ▼                                   ▼
   coda_data/{seq}/{ts}.pkl  ──► preprocess_captions.py --captioner tenstorrent
                                         │  (NPU Qwen2.5-VL, server required)
                                         ▼
                    data/captions/{seq}/captions/captions_<name>_<sec>_secs.json
                                         │
        data/navqa/data.csv ──► form_question_jsons.py
                                         ▼
                    data/questions/{seq}/human_qa.json
                                         │
                                         ▼
   eval.py --model optimal+tt/Llama-3.2-3B-Instruct  (NPU LLM, server required)
                                         ▼
                                    out/ (scored results)
```

---

## 1. 현재 수집 현황 (2026-07 기준)

| Asset | Location | Status |
|---|---|---|
| NaVQA 정답지 `data.csv` (210 questions) | `remembr/data/navqa/data.csv` | ✅ 확보 |
| NaVQA question segments | `remembr/data/navqa/{0,3,4,6,16,21,22}/qa_unfilled.json` | ✅ 리포지토리에 포함 |
| CODa seq 0 원본 zip (42GB) | `/home/user/coda_raw/0.zip` | ✅ 다운로드 & 검증 완료 |
| 8213 cam0 images + pose | `/home/user/coda_raw/seq0/` | ✅ 추출 완료 |
| 8213 `.pkl` frames | `/home/user/coda_raw/coda_data/0/` | ✅ 추출 완료 |
| ROS-free extractor | `remembr/scripts/preprocess_coda_rosfree.py` | ✅ |
| **seq0 captions** | `remembr/data/captions/0/captions/captions_tt-qwen25-vl-3b_3_secs.json` | 🔄 재생성 중 (품질 일관성) |
| **seq0 30 questions** | `remembr/data/questions/0/human_qa.json` | ⚠️ captions 재생성 후 **반드시 재생성해야 함** |

**남은 작업**: captions 재생성(품질 일관성) → question JSON 재생성 →
**4-3 평가**. 지원되지 않던 temperature로 인한 blocker는 이제 해결되었다(서버가 OpenAI의
`temperature` 파라미터를 존중한다).

### 논문 대비 충실도 (무엇이 일치하고 무엇이 다른가)

| Item | Paper original | Ours (NPU) | Fidelity |
|---|---|---|---|
| Dataset / answer key | NaVQA · CODa | 동일 (seq 0) | ✅ |
| `seconds_per_caption` | 3 seconds | 3 seconds | ✅ |
| Caption prompt | Campus prompt (video) | 문구 유지, `video→image`만 중립화 | ✅ 거의 충실 |
| Text embedding | mxbai-embed-large-v1 | 동일 | ✅ |
| Evaluation & scoring (`eval.py`) | 동일 | 동일 라우팅 (`tt/` backend) | ✅ |
| **Captioner model** | VILA 1.5 (13B) | Qwen2.5-VL 3B | ⚠️ 다름 (NPU port) |
| **Frames per caption** | ~6 frames (video window) | **1 frame (snapshot)** | ⚠️ 편차 (속도/안정성 trade-off) |
| **Inference LLM** | GPT-4o / llama3.1 8B | Llama-3.2-3B | ⚠️ 다름 (NPU port) |

> captioner와 LLM 모델 차이는 **NPU port에 내재된 것**(작은 로컬 모델)이라 불가피하다. 반면
> **frame 수(6→1)**는 우리가 선택한 trade-off이므로, 논문에 대한 충실도를 높이고 싶다면
> 여기서부터 시작하라(4-1 참고). 프롬프트는 원문 텍스트를 최대한 그대로 유지했다.

---

## 2. 데이터 수집 (다른 sequence를 추가할 때의 재현 절차)

seq 0은 이미 완료되었다. 다른 sequence(3, 4, 6, 16, 21, 22)가 필요하면 아래 단계를 반복하라.

### 2-1. 정답지 (최초 1회만)
```bash
curl -sL -o /home/user/remembr/remembr/data/navqa/data.csv \
  https://raw.githubusercontent.com/NVIDIA-AI-IOT/remembr/main/remembr/data/navqa/data.csv
```

### 2-2. CODa sequence 다운로드 (용량 주의)
필요할 수 있는 sequence별 zip 크기: **0=42GB, 3=72GB, 4=46GB, 6=68GB, 16=92GB, 21=111GB, 22=78GB** (총 ≈509GB).
```bash
SEQ=0
curl -C - -L --retry 5 -o /home/user/coda_raw/${SEQ}.zip \
  https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/sequences/${SEQ}.zip
```

### 2-3. 필요한 것만 선택적으로 추출 (cam0 + pose)
전체 zip은 대부분 LiDAR(3d_raw 등)이지만, **우리는 cam0 + pose만 필요하다**.
```bash
cd /home/user/coda_raw
unzip -q -o ${SEQ}.zip "2d_rect/cam0/${SEQ}/*" "poses/dense/${SEQ}.txt" -d seq${SEQ}/
```

### 2-4. ROS-free `.pkl` 추출
원본 `preprocess_coda.py`는 ROS(rospy/cv_bridge/tf)에 의존해서 무겁다.
우리 extractor는 plain-text pose txt + cam0 PNG만 읽어서 동일한 `.pkl` 파일을 생성한다(ROS 불필요).
```bash
cd /home/user/remembr
source remembr_venv/bin/activate
python remembr/scripts/preprocess_coda_rosfree.py \
  --seq_id ${SEQ} \
  --coda_root /home/user/coda_raw/seq${SEQ} \
  --out_dir  /home/user/coda_raw/coda_data/${SEQ} \
  --frame_skip 1
```
- 출력: `{ts}.pkl` = `{cam0(BGR), position, rotation(xyzw), timestamp}`.
- pose 행 ↔ cam0 frame은 인덱스 기준으로 1:1 대응한다(검증됨).

---

## 3. NPU 서버 시작 (terminal A, tt-metal 환경)

```bash
cd /home/user/tt-metal
TT_LLM_MAX_SEQ_LEN=2048 TT_VLM_MAX_SEQ_LEN=1024 \
  bash /home/user/remembr/tt_server/run_server.sh
```
- `TT_LLM_MAX_SEQ_LEN=2048`은 **필수**다(NonAgent 프롬프트가 2048까지 pad한다).
- `Application startup complete.`가 나타날 때까지 기다려라.
- 확인: `curl -s http://localhost:8100/health` → `{"ready":true,"llm":true,"vlm":true}`
- **서버가 올라오지 않고 `Device 0 init: failed to initialize FW! Try resetting the board.`가 나오면 → 6장의 복구 절차를 참고하라.**

---

## 4. 테스트 실행 (terminal B, remembr_venv, **inner** 디렉터리)

```bash
cd /home/user/remembr
source remembr_venv/bin/activate
cd remembr                              # inner: scripts use ./data relative paths
export OPENAI_API_KEY=local PYTHONPATH=/home/user/remembr
export TT_VLM_TIMEOUT=3600               # headroom for the first compile
```

### 4-1. NPU captioning
```bash
python scripts/preprocess_captions.py \
  --captioner tenstorrent \
  --seq_id 0 --seconds_per_caption 3 \
  --num-video-frames 1 \
  --captioner_name tt-qwen25-vl-3b \
  --data_path /home/user/coda_raw/coda_data \
  --out_path data/captions/0/captions
```
- **`--num-video-frames 1` = 안정성/속도 trade-off (⚠️ 논문 대비 충실도 편차).**
  논문은 각 3초 window를 **~6 frames(video)**로 caption하는 반면, 우리는 각 window를
  **1 frame(snapshot)**만으로 caption한다 → 시간 정보를 줄이는 **명백한 편차**다. 반면 기본값(6 images)은
  **Qwen multi-image shape의 첫 trace compile을 15분 이상** 걸리게 만들어, client 타임아웃과 보드 먹통을
  유발했다(실제로 일어났다). 재현성과 안정성을 위해 1 frame을 쓰지만, **논문 수준의 충실도가 필요하면**
  frame 수를 올리고(예: 3–6) 첫 compile(수십 분)을 끝까지 기다려라 — **절대 도중에 죽이지 마라**.
- caption 프롬프트는 논문 원문을 그대로 사용하며, VILA video 문구만 중립화한다(`<video>` 제거,
  "the few seconds of the video" → "the image"). 나머지("Think step by step", focus list)는 논문과 정확히 동일하다.
- **문장 잘림 방지**: captioner는 `max_gen_len=256`을 보낸다(기본값, `TT_VLM_MAX_GEN_LEN`으로 조정 가능).
  서버 기본값인 128이면 장황한 caption이 중간에 잘린다.
- **Checkpoint/resume**: 모든 caption 하나하나가 즉시
  `..._secs.json.partial.jsonl`에 기록된다. 중단(kill, 서버 문제) 후 **같은 명령을 다시 실행하면 이어서 진행된다**
  (이미 끝난 segment는 건너뛴다). 완전히 끝나면 최종 JSON을 저장하고 partial을 삭제한다. 처음부터 다시 시작하려면
  partial을 삭제하라.
- **점진적 slowdown은 해결되었다(원인이 두 가지였다)**:
  **① tt-metal Inspector**(debug, 기본 ON)가 mesh workload마다 YAML을 flush → caption당 ~5,000회
  동기 disk I/O, YAML이 153MB까지 커지며 → vision이 19에서 2.1 it/s로 떨어졌다.
  `run_server.sh`가 `TT_METAL_INSPECTOR=0`으로 이를 비활성화한다.
  **② Tracy profiler 빌드** — ①을 끄고 나서야 드러났다(84→50 it/s). tt-metal이
  `--enable-profiler`로 빌드되면 Tracy가 zone을 계속 누적한다(같은 이미지 12회 실행 기준 request당 +33MB RSS,
  request당 +1.0s latency).
  **env로는 끌 수 없으므로 기본 빌드로 재빌드해야 한다** → [RUNBOOK 1-0](../tt_server/RUNBOOK.md).
  둘 다 고친 후 측정: **caption당 안정적으로 36초, RSS 증가 0**.
  다시 느려지면 서버 로그의 `32 x Qwen2_5_VLVisionBlock ... it/s` 값이 가장 좋은 지표다.
- 결과: `data/captions/0/captions/captions_tt-qwen25-vl-3b_3_secs.json`.
  각 caption item에는 `caption_latency_s`가 포함된다. 이 값은 해당 segment의 captioner 호출 wall-clock 시간이며,
  text embedding 계산 시간은 포함하지 않는다.
- 첫 caption만 trace compilation 때문에 느리고, 이후에는 캐시되어 빠르다.

### 4-2. question JSON 생성 (정답지 + captions → human_qa.json)
```bash
python scripts/question_scripts/form_question_jsons.py \
  --caption_file captions_tt-qwen25-vl-3b_3_secs
```
- `./data/navqa/data.csv` + 위의 captions → `data/questions/0/human_qa.json`.
- captions가 있는 sequence만 처리되고, 나머지는 자동으로 건너뛴다.
- ⚠️ **captions를 재생성하면 이 단계도 반드시 다시 실행해야 한다.** question JSON은 captions에서
  추출한 "optimal context"(답하는 데 필요한 captions)를 담고 있으므로, captions가 바뀌면 그 context도
  갱신해야 한다.

### 4-3. 평가
**권장 — optimal(oracle) context + NonAgent** (seq_len 안에 맞고, tool-calling 불필요):
```bash
python scripts/eval.py \
  --model optimal+tt/Llama-3.2-3B-Instruct \
  --sequence_id 0 \
  --caption_file captions_tt-qwen25-vl-3b_3_secs \
  --coda_dir /home/user/coda_raw/coda_data \
  --postfix _0
```
**Full RAG (선택) — Milvus retrieval + agent** (3B tool-calling은 불안정):
```bash
python scripts/eval.py \
  --model remembr+tt/Llama-3.2-3B-Instruct \
  --sequence_id 0 --caption_file captions_tt-qwen25-vl-3b_3_secs \
  --coda_dir /home/user/coda_raw/coda_data --postfix _0
```
- 결과는 `./out/`로 간다. binary/position/time/duration은 채점되고, text 질문은 건너뛴다
  (원본 코드 동작). seq0의 경우 30개 중 text 질문 5개가 제외되어 **25개 질문이 채점된다**.
- latency 필드도 함께 저장된다. 최상위 `eval_total_latency_s`는 전체 eval 실행 시간이고, 각 response의
  `eval_latency_s`는 해당 질문 처리 전체 시간이다. `query_latency_s`/`elapsed`는 retry를 포함한
  `model.query()` 누적 시간, `last_attempt_latency_s`는 마지막 attempt 시간, `attempts`는 시도 횟수다.
  `attempt_latencies`에는 성공, unanswered retry, exception attempt가 모두 attempt별로 기록된다. 건너뛴 text
  질문도 `unscored`/`skipped`로 표시하고 `eval_latency_s`는 남긴다.
- ⚠️ **접두사 없는 plain `tt/...`는 피하라** — TextMemory가 전체 window의 captions를 넣어,
  긴 질문에서 seq_len을 초과한다.
- **retry loop 주의**: 3B가 time 질문에 `type='text', time=None`으로 답하면,
  `evaluate_output`이 `abs(answer - None)`에서 crash하고 `answer_squad_question`의 `while True`가
  재시도한다. 이 재시도가 진전을 이루려면 **매번 다른 답이 나와야 하므로**, `--temperature`를 0으로
  설정하지 마라(기본 0.7 권장). 서버는 이제 OpenAI의 `temperature` 파라미터를 존중한다(단, 0은 greedy를
  의미하며, 이 경우 무한 루프가 된다).

### 모델별 memory 모드
| `--model` | Memory | Context | NPU에서 3B 적합성 |
|---|---|---|---|
| `optimal+tt/...` | Oracle context | 작음 (answer captions만) | ✅ 권장 |
| `remembr+tt/...` | Milvus RAG top-k | 제한적 | △ 약한 tool-calling |
| `tt/...` | 전체 TextMemory | **overflow 위험** | ✗ |

---

## 5. NPU 안전 수칙 (필수)

device가 실행 또는 compile 중인데 teardown 없이 프로세스가 죽으면, core·NoC·ARC
상태가 정리되지 않아 **hardware deadlock(먹통)**에 빠진다. 재시작하면
`failed to initialize FW`가 나오고 복구가 번거롭다(이 PC는 재부팅 시 NPU를 먼저 켜야 인식된다는 제약도 있다). 수칙:

1. **서버는 단 한 번의 `Ctrl-C`(또는 SIGTERM)로 종료하고 기다려라.** 서버가 worker를
   join하고 `close_mesh_device()`로 device를 깨끗이 닫는다. 다음 로그가 보이면 완료된 것이다:
   ```
   Shutting down: releasing the NPU cleanly ...
   NPU released cleanly. Safe to exit.
   ```
   **teardown 도중 추가로 누른 Ctrl-C는 무시되므로**, 키를 마구 눌러도 안전하다(예전 버그는 고쳐졌다).
2. **절대 `kill -9`(SIGKILL)를 쓰지 마라.** 이것만이 teardown을 건너뛰어 보드를 먹통으로 만든다.
   유일하게 진짜 위험한 종료 방식이다.
3. **client(captioning)를 죽이는 것은 안전하며** checkpoint에서 이어진다. 다만 client를 죽여도
   서버는 죽지 않으니, 성급하게 서버까지 `-9`하지 마라.
4. **첫 compile(느린 구간)을 죽이지 말고 기다려라.** `--num-video-frames 1`로 이 구간을
   짧게 유지하고, `TT_VLM_TIMEOUT=3600`으로 client에 넉넉한 timeout을 줘라.
5. device 점유 확인: `fuser -v /dev/bos/0` (결과 없음 = free), `lsmod | grep bos` (usage 0 = 미사용).

---

## 6. 먹통 보드 복구 (FW init 실패 시)

증상: 서버 시작 시
```
Device 0: Timeout waiting for physical cores to finish
Device 0 init: failed to initialize FW! Try resetting the board.
```

### 진단 (⚠️ device node는 `/dev/bos` — Tenstorrent가 아니다)
```bash
fuser -v /dev/bos/0            # who is holding the device (empty result = free)
lsmod | grep bos              # bos module usage count (0 = nobody is using it)
pgrep -af tt_infer_server     # is a server process still around
uptime -p                     # did it reboot
# Definitive test for whether the board is actually dead (after bringing the server down):
#   python3 -m ttnn.examples.usage.run_op_on_device
#   -> tensor output means the board is fine; an FW init error means wedged
```

### 복구 순서
1. **먼저 zombie 서버를 확인하고 정리하라.** 멈춘 서버가 **device를 잡고 있지 않으면**
   (`fuser`에 없거나, op가 device를 열 수 있으면) `kill -9 <pid>`로 정리해도 **안전하다**
   (중단시킬 device 작업이 없다).
   → 정리 후 서버만 재시작해도 해결되는 경우가 많다(실제로 그랬다).
2. **보드가 정말 먹통이면(FW init 실패)** → warm reset이 필요하다. 이 환경의 제약:
   `tt-smi`가 설치되어 있지 않고, sudo는 비밀번호가 필요하며, `/sys/bus/pci/rescan`은 root 전용이라
   `sudo <bos reset tool>`은 sudo 계정이 있어야만 가능하다.
3. **sudo가 없으면 → 재부팅.** 다만 **이 PC는 NPU를 먼저 켜야만 NPU를 인식하므로**,
   단순한 `sudo reboot`으로는 부족할 수 있다 — 올바른 전원 순서(전원 차단 → NPU on → PC on)가
   필요할 수 있다. 심하게 먹통이면 warm reboot으로는 해결되지 않고 cold power cycle이 필요하다.
4. 복구 확인: `fuser -v /dev/bos/0` (free) + `python3 -m ttnn.examples.usage.run_op_on_device`
   (op가 정상 실행).

> 이 환경의 제약: **`bos` reset tool이 설치되어 있지 않고, sudo는 비밀번호가 필요하며,
> `/sys/bus/pci/rescan`은 root 전용이다**. 그래서 "명령줄에서 NPU만 reset"은 sudo 계정이 있어야만
> 가능하다. 대부분의 경우 **zombie 서버를 정리하고 재시작하면** 해결된다(보드가 진짜 죽은 게 아니라면).

---

## 7. 관련 파일

| File | Role |
|---|---|
| `remembr/scripts/preprocess_coda_rosfree.py` | ROS-free CODa→.pkl extractor |
| `remembr/scripts/preprocess_captions.py` | Captioning (`--captioner tenstorrent` 지원) |
| `remembr/scripts/question_scripts/form_question_jsons.py` | data.csv + captions → question JSON |
| `remembr/scripts/eval.py` | Evaluation (`tt/` backend routing) |
| `remembr/tt_server/run_server.sh` | NPU 서버 시작 (Inspector 비활성화 포함) |
| `remembr/tt_server/npu_info.py` | NPU spec dump (arch/DRAM/Tensix/L1) |
| `remembr/scripts/viz.py` | (선택) 평가 결과를 driving video에 오버레이해 시각화 — 파이프라인에는 불필요 |
| `docs/TENSTORRENT_INTEGRATION.md` | 아키텍처 & 설계 결정 |
