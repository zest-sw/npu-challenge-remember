# 원 논문 데이터(NaVQA/CODa) 테스트 실행 가이드 — NPU

ReMEmbR 논문의 **NaVQA 벤치마크**를 우리 Tenstorrent NPU 구조(Qwen2.5-VL 캡셔너 +
Llama-3.2-3B LLM)로 돌리는 전체 절차. 데이터 확보 → 전처리 → 캡셔닝 → 질문 생성 →
평가까지를 한 문서에 정리한다.

> ⚠️ **먼저 읽을 것 — 5번 "NPU 안전 수칙".** 캡셔닝 중 서버를 잘못 죽이면 보드가
> wedged 상태가 되어 **재부팅이 필요**해진다. 실제로 한 번 겪었다.

---

## 0. 파이프라인 한눈에

```
CODa seq .zip ──(선택 압축해제)──► 2d_rect/cam0 + poses/dense
      │                                   │
      │                    preprocess_coda_rosfree.py (ROS 불필요)
      ▼                                   ▼
   coda_data/{seq}/{ts}.pkl  ──► preprocess_captions.py --captioner tenstorrent
                                         │  (NPU Qwen2.5-VL, 서버 필요)
                                         ▼
                    data/captions/{seq}/captions/captions_<name>_<sec>_secs.json
                                         │
        data/navqa/data.csv ──► form_question_jsons.py
                                         ▼
                    data/questions/{seq}/human_qa.json
                                         │
                                         ▼
   eval.py --model optimal+tt/Llama-3.2-3B-Instruct  (NPU LLM, 서버 필요)
                                         ▼
                                    out/ (채점 결과)
```

---

## 1. 현재 확보 상태 (2026-07 기준)

| 자산 | 위치 | 상태 |
|---|---|---|
| NaVQA 정답키 `data.csv` (210문항) | `remembr/data/navqa/data.csv` | ✅ 확보 |
| NaVQA 질문 세그먼트 | `remembr/data/navqa/{0,3,4,6,16,21,22}/qa_unfilled.json` | ✅ 리포 포함 |
| CODa seq 0 원본 zip (42GB) | `/home/user/coda_raw/0.zip` | ✅ 다운로드·검증 |
| cam0 8213장 + pose | `/home/user/coda_raw/seq0/` | ✅ 압축해제 |
| `.pkl` 프레임 8213개 | `/home/user/coda_raw/coda_data/0/` | ✅ 추출 |
| ROS-free 추출기 | `remembr/scripts/preprocess_coda_rosfree.py` | ✅ |

**남은 것**: 캡셔닝(2단계)부터. 보드가 정상일 때 재개.

### 논문 대비 충실도 (무엇이 같고 무엇이 다른가)

| 항목 | 논문 원본 | 우리(NPU) | 충실도 |
|---|---|---|---|
| 데이터셋 / 정답키 | NaVQA · CODa | 동일 (seq 0) | ✅ |
| `seconds_per_caption` | 3초 | 3초 | ✅ |
| 캡션 프롬프트 | 캠퍼스 프롬프트(video) | 문구 보존, `video→image`만 중화 | ✅ 거의 충실 |
| 텍스트 임베딩 | mxbai-embed-large-v1 | 동일 | ✅ |
| 평가·채점 (`eval.py`) | 동일 | 동일 라우팅(`tt/` 백엔드) | ✅ |
| **캡셔너 모델** | VILA 1.5 (13B) | Qwen2.5-VL 3B | ⚠️ 다름 (NPU 이식) |
| **프레임/캡션** | ~6프레임(비디오 윈도우) | **1프레임(스냅샷)** | ⚠️ 편차 (속도/안정성 절충) |
| **추론 LLM** | GPT-4o / llama3.1 8B | Llama-3.2-3B | ⚠️ 다름 (NPU 이식) |

> 캡셔너·LLM 모델 차이는 **NPU 이식의 본질**(작은 로컬 모델)이라 불가피하다. 반면
> **프레임 수(6→1)** 는 우리가 고른 절충이므로, 논문 충실도를 높이려면 여기부터
> 올린다(4-1 참고). 프롬프트는 원문을 최대한 보존했다.

---

## 2. 데이터 확보 (다른 시퀀스 추가 시 재현 절차)

seq 0은 이미 완료. 다른 시퀀스(3,4,6,16,21,22)가 필요하면 아래를 반복.

### 2-1. 정답키 (한 번만)
```bash
curl -sL -o /home/user/remembr/remembr/data/navqa/data.csv \
  https://raw.githubusercontent.com/NVIDIA-AI-IOT/remembr/main/remembr/data/navqa/data.csv
```

### 2-2. CODa 시퀀스 다운로드 (용량 주의)
필요 시퀀스별 zip 크기: **0=42GB, 3=72GB, 4=46GB, 6=68GB, 16=92GB, 21=111GB, 22=78GB** (합계 ≈509GB).
```bash
SEQ=0
curl -C - -L --retry 5 -o /home/user/coda_raw/${SEQ}.zip \
  https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/sequences/${SEQ}.zip
```

### 2-3. 필요한 부분만 선택 압축해제 (cam0 + pose)
전체 zip엔 LiDAR(3d_raw 등)가 대부분이지만 **우리는 cam0 + pose만 필요**.
```bash
cd /home/user/coda_raw
unzip -q -o ${SEQ}.zip "2d_rect/cam0/${SEQ}/*" "poses/dense/${SEQ}.txt" -d seq${SEQ}/
```

### 2-4. ROS-free `.pkl` 추출
원본 `preprocess_coda.py`는 ROS(rospy/cv_bridge/tf) 의존이라 무겁다. 우리 추출기는
평문 pose txt + cam0 PNG만 읽어 동일한 `.pkl`을 만든다 (ROS 불필요).
```bash
cd /home/user/remembr
source ~/remembr_venv/bin/activate
python remembr/scripts/preprocess_coda_rosfree.py \
  --seq_id ${SEQ} \
  --coda_root /home/user/coda_raw/seq${SEQ} \
  --out_dir  /home/user/coda_raw/coda_data/${SEQ} \
  --frame_skip 1
```
- 출력: `{ts}.pkl` = `{cam0(BGR), position, rotation(xyzw), timestamp}`.
- pose 행 ↔ cam0 프레임은 index로 1:1 대응(검증됨).

---

## 3. NPU 서버 기동 (터미널 A, tt-metal 환경)

```bash
cd /home/user/tt-metal
TT_LLM_MAX_SEQ_LEN=2048 TT_VLM_MAX_SEQ_LEN=1024 \
  bash /home/user/remembr/tt_server/run_server.sh
```
- `TT_LLM_MAX_SEQ_LEN=2048` **필수**(NonAgent 프롬프트가 2048까지 패딩).
- `Application startup complete.` 뜰 때까지 대기.
- 확인: `curl -s http://localhost:8100/health` → `{"ready":true,"llm":true,"vlm":true}`
- **서버가 뜨지 않고 `Device 0 init: failed to initialize FW! Try resetting the board.`가 나오면 → 6번 복구 절차.**

---

## 4. 테스트 실행 (터미널 B, remembr_venv, **inner** 디렉토리)

```bash
source ~/remembr_venv/bin/activate
cd /home/user/remembr/remembr           # inner: 스크립트가 ./data 상대경로 사용
export OPENAI_API_KEY=local PYTHONPATH=/home/user/remembr
export TT_VLM_TIMEOUT=3600               # 첫 컴파일 여유
```

### 4-1. NPU 캡셔닝
```bash
python scripts/preprocess_captions.py \
  --captioner tenstorrent \
  --seq_id 0 --seconds_per_caption 3 \
  --num-video-frames 1 \
  --captioner_name tt-qwen25-vl-3b \
  --data_path /home/user/coda_raw/coda_data \
  --out_path data/captions/0/captions
```
- **`--num-video-frames 1` = 안정성/속도 절충 (⚠️ 논문 대비 충실도 편차).**
  논문은 3초 윈도우를 **~6프레임(비디오)** 으로 캡션하지만, 우리는 각 윈도우를
  **1프레임(스냅샷)** 으로만 캡션한다 → 시간적 정보가 줄어드는 **명백한 편차**.
  대신 기본값(6장)은 Qwen 멀티이미지 shape의 **첫 트레이스 컴파일이 15분+** 걸려
  클라이언트 타임아웃 + 보드 wedged 사고를 유발했다(실제 발생). 재현 안정성을 위해
  1프레임을 쓰되, **논문 수준 충실도가 필요하면** 프레임 수를 올리고(예: 3~6) 첫
  컴파일(수십 분)을 **절대 중간에 죽이지 말고** 기다린다.
- 캡션 프롬프트는 논문 원문을 그대로 쓰되, VILA 비디오 문구만 중화한다
  (`<video>` 제거, "the few seconds of the video" → "the image"). 나머지("Think
  step by step", focus 리스트)는 논문 그대로.
- **문장 잘림 방지**: 캡셔너가 `max_gen_len=256`(기본, `TT_VLM_MAX_GEN_LEN`으로 조절)을 보낸다.
  서버 기본 128이면 verbose 캡션이 중간에 잘린다.
- **체크포인트/resume**: 캡션 1개마다 `..._secs.json.partial.jsonl`에 즉시 기록된다. 중단(kill,
  서버 문제) 후 **같은 명령을 재실행하면 이어서** 진행(이미 한 세그먼트는 skip). 전체 완료 시
  최종 JSON 저장 + partial 삭제. 처음부터 다시 하려면 partial을 지운다.
- **점점 느려지는 문제는 해결됨**: 원인은 tt-metal **Inspector**(디버그 기능, 기본 ON)가 mesh workload마다
  YAML을 flush해 캡션당 ~5,000회 동기 디스크 I/O를 내던 것. `run_server.sh`가 이제
  `TT_METAL_INSPECTOR=0`/`TT_METAL_INSPECTOR_RPC=0`으로 끈다(tt-metal 소스 무수정).
  실측: 비전 블록 19→2.1 it/s로 하락하던 것이 **~42 it/s 안정 유지**로 개선.
  혹시 다시 느려지면 서버 로그의 `32 x Qwen2_5_VLVisionBlock ... it/s` 값이 최고의 게이지다.
- 결과: `data/captions/0/captions/captions_tt-qwen25-vl-3b_3_secs.json`.
- 첫 캡션만 트레이스 컴파일로 느리고, 이후 캐시되어 빨라진다.

### 4-2. 질문 JSON 생성 (정답키 + 캡션 → human_qa.json)
```bash
python scripts/question_scripts/form_question_jsons.py \
  --caption_file captions_tt-qwen25-vl-3b_3_secs
```
- `./data/navqa/data.csv` + 위 캡션 → `data/questions/0/human_qa.json`.
- 캡션이 있는 seq만 처리, 나머지는 자동 skip.

### 4-3. 평가
**권장 — optimal(오라클) 컨텍스트 + NonAgent** (seq_len에 맞고 tool-calling 불필요):
```bash
python scripts/eval.py \
  --model optimal+tt/Llama-3.2-3B-Instruct \
  --sequence_id 0 \
  --caption_file captions_tt-qwen25-vl-3b_3_secs \
  --coda_dir /home/user/coda_raw/coda_data \
  --postfix _0
```
**전체 RAG(선택) — Milvus 검색 + 에이전트** (3B tool-calling 불안정):
```bash
python scripts/eval.py \
  --model remembr+tt/Llama-3.2-3B-Instruct \
  --sequence_id 0 --caption_file captions_tt-qwen25-vl-3b_3_secs \
  --coda_dir /home/user/coda_raw/coda_data --postfix _0
```
- 결과는 `./out/`. binary/position/time/duration 채점, text 질문은 skip(원 코드 동작).
- ⚠️ **순수 `tt/...`(접두어 없이)는 피할 것** — TextMemory가 윈도우 전체 캡션을 넣어
  긴 질문에서 seq_len 초과.

### 모델별 메모리 방식
| `--model` | 메모리 | 컨텍스트 | NPU 3B 적합성 |
|---|---|---|---|
| `optimal+tt/...` | 오라클 컨텍스트 | 작음(정답 캡션만) | ✅ 권장 |
| `remembr+tt/...` | Milvus RAG top-k | 제한됨 | △ tool-calling 약함 |
| `tt/...` | TextMemory 전체 | **초과 위험** | ✗ |

---

## 5. NPU 안전 수칙 (반드시)

device가 실행/컴파일 중일 때 teardown 없이 죽으면 코어·NoC·ARC 상태가 정리되지 않아
**하드웨어 데드락(wedged)** 에 빠진다. 재기동 시 `failed to initialize FW`가 나고 복구가
번거롭다(이 PC는 재부팅 시 NPU를 먼저 켜야 인식되는 제약도 있음). 규칙:

1. **서버 종료는 `Ctrl-C` 한 번(또는 SIGTERM) 후 기다린다.** 서버가 워커를 join해
   `close_mesh_device()`로 device를 깨끗이 닫는다. 아래 로그가 뜨면 완료:
   ```
   Shutting down: releasing the NPU cleanly ...
   NPU released cleanly. Safe to exit.
   ```
   teardown 중 **추가 Ctrl-C는 무시**되므로 연타해도 안전(예전 버그 수정됨).
2. **`kill -9`(SIGKILL)는 절대 금지.** 이것만은 teardown을 건너뛰어 보드를 wedged로 만든다.
   유일하게 위험한 종료 방식.
3. **클라이언트(캡셔닝)는 죽여도 안전**하고 체크포인트로 이어진다. 단 클라이언트 kill이
   서버를 죽이는 건 아니므로, 급하게 서버까지 `-9`하지 말 것.
4. **첫 컴파일(느린 구간)은 죽이지 말고 기다린다.** `--num-video-frames 1`로 이 창을 짧게 유지.
   `TT_VLM_TIMEOUT=3600`으로 클라이언트 타임아웃도 넉넉히.
5. device 점유 확인: `fuser -v /dev/bos/0` (빈 결과 = 프리), `lsmod | grep bos`(usage 0 = 미사용).

---

## 6. 보드 wedged 복구 (FW init 실패 시)

증상: 서버 기동 시
```
Device 0: Timeout waiting for physical cores to finish
Device 0 init: failed to initialize FW! Try resetting the board.
```

### 진단 (⚠️ device 노드는 `/dev/bos` — Tenstorrent 아님)
```bash
fuser -v /dev/bos/0            # 누가 device 쥐고 있나 (빈 결과 = 프리)
lsmod | grep bos              # bos 모듈 usage count (0 = 아무도 안 씀)
pgrep -af tt_infer_server     # 서버 프로세스 남아있나
uptime -p                     # 재부팅 했는지
# 보드가 실제로 죽었는지 결정적 테스트 (서버 내린 뒤):
#   python3 -m ttnn.examples.usage.run_op_on_device
#   -> tensor 출력 나오면 보드 정상, FW init 에러면 wedged
```

### 복구 순서
1. **먼저 좀비 서버 확인·정리.** hang된 서버가 device를 **안 쥐고 있으면**(`fuser`에 없거나
   op가 device를 열 수 있으면) `kill -9 <pid>`로 정리해도 **안전**하다(끊길 device 연산이 없음).
   → 정리 후 서버 재기동만으로 복구되는 경우가 많다(실제 사례).
2. **보드가 정말 wedged면(FW init 실패)** → 워엄 리셋 필요. 이 환경 제약:
   `tt-smi` 미설치 + sudo 비번 필요 + `/sys/bus/pci/rescan` root 전용이라, sudo 계정이
   있을 때만 `sudo <bos reset 도구>` 가능.
3. **sudo 불가 시 → 재부팅.** 단 **이 PC는 NPU를 먼저 켜고 부팅해야 인식**되므로, 단순
   `sudo reboot`이 아니라 올바른 전원 순서(전원 차단 → NPU on → PC on)가 필요할 수 있다.
   심하게 wedged면 warm reboot로 안 풀려 cold power cycle이 필요.
4. 복구 확인: `fuser -v /dev/bos/0`(프리) + `python3 -m ttnn.examples.usage.run_op_on_device`(정상 연산).

> 이 환경 제약: **`bos` reset 도구 미설치 + sudo 비번 필요 + `/sys/bus/pci/rescan` root 전용**.
> 그래서 "NPU만 명령으로 리셋"은 sudo 계정이 있을 때만 가능. 대부분은 **좀비 서버 정리 + 재기동**으로
> 해결된다(보드가 실제로 죽은 경우가 아니라면).

---

## 7. 관련 파일

| 파일 | 역할 |
|---|---|
| `remembr/scripts/preprocess_coda_rosfree.py` | ROS-free CODa→.pkl 추출기 |
| `remembr/scripts/preprocess_captions.py` | 캡셔닝(`--captioner tenstorrent` 지원) |
| `remembr/scripts/question_scripts/form_question_jsons.py` | data.csv+캡션 → 질문 JSON |
| `remembr/scripts/eval.py` | 평가(`tt/` 백엔드 라우팅) |
| `remembr/tt_server/run_server.sh` | NPU 서버 기동 |
| `docs/TENSTORRENT_INTEGRATION.md` | 아키텍처·설계 결정 |
```
