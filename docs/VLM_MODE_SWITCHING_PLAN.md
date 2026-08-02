# 계획 — 모드 기반 모델 전환 (클라이언트가 모드 선택)

목표: 클라이언트가 요청에서 **모드(모델 조합)**를 고르면 서버가 해당 구성으로 서빙.

| 모드 | 구성 | 엔드포인트 | 상태 |
|---|---|---|---|
| **1. 3b+3b** | Llama-3.2-3B + Qwen2.5-VL-3B | `/caption` + `/v1/chat` | ✅ 검증됨(현재) |
| **2. caption-8b** | Qwen3-VL-8B (VLM 단독) | `/caption`만 | ❌ 미검증(진짜 리스크) |
| **3. llm-8b** | Llama-3.1-8B (LLM 단독) | `/v1/chat`만 | ✅ 검증됨(WORKLOG 8B run) |

"요청마다 교체"는 물리적으로 per-frame 불가(로드 수 분) → **run 단위 모드 선택**. 요청 모드가 현재와
다르면 전환(수 분), 같으면 즉시.

---

## 결정: 단일 venv 통일 (1순위) — 스모크 테스트로 근거 확보

**핵심 분기점이었던 "transformers 통일 가능?"에 대해 보드 없이 검증 완료:**

`python_env_tf457`(python_env 복제 + transformers 4.57.1)에서, device 안 열고:
- ✅ 세 모델 클래스 존재 (`Qwen2_5_VL` / `Qwen3VL` / `Llama`)
- ✅ tt 모듈 6개 import (qwen25·llama32·qwen3)
- ✅ Qwen2.5-VL-3B `Qwen2_5_VLProcessor`+config, Llama-3.2-3B tokenizer+config 로드
- ✅ qwen25_vl ModelArgs 구성

→ **4.57.1이 기존 3B 코드(4.53용)를 import/config/processor 레벨에서 안 깨뜨림.** qwen3가 요구하는
4.57.1로 3종 백엔드를 한 venv에 통일 가능성 매우 유력.

**통일 시 아키텍처 (단순):**
```
Client → 서버 (단일 프로세스, venv=4.57.1, device 계속 소유)
           ├─ GET  /mode          현재 모드
           ├─ POST /mode {"mode":"caption-8b"}   in-process 전환
           ├─ POST /caption / /v1/chat   활성 모드에 맞게 처리(없으면 503)
           └─ 워커: 모드 전환 = 모델 언로드 + 로드 (프로세스 유지)
```
- router/supervisor/멀티프로세스 **전부 불필요**
- 프로세스가 device를 계속 쥠 → **핸드오프 wedge 위험 제거** (이번 세션 먹통 조건 회피)

**미확인(보드 필요)**: 실제 가중치 로드 + ttnn 추론이 4.57.1에서 3B와 **동일 결과**인지. processor
출력 형식/텐서 shape 변화 가능성 → device 열어야 확정.

---

## 모드 불일치 요청 처리 (정책 확정)

**핵심 결정: 요청이 모드를 자동 전환하지 않는다. 전환은 오직 명시적 `/mode`.**
이유: 로드가 수 분 → 자동전환 시 클라이언트들이 3B↔8B 번갈아 요청하면 매번 수 분 핑퐁으로 서버
사용 불가. 캡셔닝은 phase 단위(한 시퀀스=한 모드)라 run 시작에 모드 1회 설정이 자연스러움.
원칙: 요청은 "모드 맞으면 빠르게 성공" 또는 "즉시 명확한 에러", **절대 조용히 수 분 블록 금지**.

| 상황 | 응답 | 메시지 |
|---|---|---|
| 활성 모드에 엔진 있음 | 200 | 정상 |
| 활성 모드에 엔진 없음 (예: llm-8b에서 /caption) | **409** | `"/caption unavailable in mode 'llm-8b'. POST /mode first."` |
| 요청이 다른 모델 지정 | **409** | `"model X requires mode Y (active: Z). Switch via /mode."` |
| 모드 전환 진행 중 | **503** | `"mode switch in progress, retry."` |

- 409 = 클라이언트가 `/mode`로 고칠 수 있는 상태(actionable). 503 = 일시적(로딩/전환 중).
- 단일 워커 큐라 전환 잡(수 분) 동안 들어온 요청이 큐에서 대기하면 수 분 블록됨 → **"전환 중"
  플래그**로 그 사이 요청은 큐에 넣지 않고 즉시 503. ("조용히 블록 안 함" 원칙 일관)
- 클라이언트(preprocess)는 run 시작에 `/mode` 1회 → 이후 요청 전부 매치.

---

## Phase 0 — 보드 없이 지금 (전부 가능)

### 0-1. ✅ (완료) 통일 venv 씨앗
`/home/user/tt-metal/python_env_tf457` = python_env 복제 + transformers 4.57.1. 원본 4.53 무손상.

### 0-2. Qwen3-VL-8B 가중치 다운로드
- HF repo id 확정(`Qwen/Qwen3-VL-8B-Instruct` 등) → HF 캐시(`HF_HOME=/home/user/tt-metal/huggingface`)로.
- 다운로드만. 로드/추론은 보드.

### 0-3. 서버 엔진 모듈화 리팩터 (순수 코드, device 무관)
현재 `tt_infer_server.py`: qwen25_vl 모듈 레벨 하드코딩 import + `_engines={"llm","vlm"}`.

목표:
```python
# 모드 레지스트리
MODES = {
  "3b+3b":      {"llm":"meta-llama/Llama-3.2-3B-Instruct", "vlm":"Qwen/Qwen2.5-VL-3B-Instruct"},
  "caption-8b": {"llm":None,                                "vlm":"Qwen/Qwen3-VL-8B-Instruct"},
  "llm-8b":     {"llm":"meta-llama/Llama-3.1-8B-Instruct",  "vlm":None},
}
VLM_BACKEND = {"Qwen/Qwen2.5-VL-3B-Instruct":"qwen25_vl", "Qwen/Qwen3-VL-8B-Instruct":"qwen3_vl"}

# 공통 인터페이스 + 백엔드 lazy import 팩토리 (모듈 레벨 하드코딩 제거)
class VLMEngine: def load; def describe(images,prompt,max_gen_len)->str; def unload()
def make_vlm_engine(model_name, mesh_device) -> VLMEngine   # backend별 import
# describe() 본문은 qwen25/qwen3 공통(API 평행) → 파라미터화

# 워커 상태 + in-process 전환
_active = {"mode":None, "llm":None, "vlm":None}
def _apply_mode(name):        # 워커에서만(직렬화)
    cfg = MODES[name]
    # 필요없어진 엔진 unload → 필요한 엔진 load (모델명 같으면 유지)
```
- 기존 QwenVLEngine → `make_vlm_engine("...3B", ...)` 결과로 흡수, **모드1 동작 동일** 검증.
- llama도 동일하게 모델명 파라미터화(3B/8B).

### 0-4. `/mode` 엔드포인트 + 클라이언트 모드 지정
- 서버: `POST /mode {"mode":...}` → 워커에 `_apply_mode` 잡 제출, ready까지 블록/503.
- `/caption`·`/v1/chat`은 활성 모드에 해당 엔진 없으면 503(명확한 메시지).
- 클라이언트(`tenstorrent_captioner.py`, preprocess): `--mode`/`--vlm-model` 인자로 선택. 기본 3b+3b(하위호환).

### 0-5. 언로드(자원 회수) — 통일 경로의 최대 리스크 + 보드 죽음 연관

**조사 결과 (보드 무관):**
- Generator `__del__`은 단일 device(data_parallel=1)에서 **사실상 no-op** — trace 해제·weight
  dealloc·program cache clear 전부 안 함. **파이썬 참조만 버리면 device 자원 안 비워짐.**
- 모델이 trace 캡처(`begin_trace_capture`, 30MB region) 후 붙듦. 서버엔 `release_trace` **없음**
  (지금은 1회 로드라 무해하나, 언로드/재로드 시 trace region 누적).
- weights/KV는 `model.layers` 등에 흩어진 ttnn 텐서 → 명시적 걷어내기 필요. tt_transformers에
  teardown 헬퍼 없음 → 직접 구현.

**보드 죽음과의 연결**: 과거 죽음 = force-kill/크래시로 device를 dirty하게 남김 → 다음 open FW init
실패. 모드 전환 언로드가 부실하면 **매 전환이 같은 dirty 상태를 반복 생성** → 죽음 상시화. (사용자
지적 확인됨.)

**회수 방식 3가지 트레이드오프:**

| 방식 | tt-metal | 회수 확실성 | 위험 |
|---|---|---|---|
| per-tensor dealloc (device 유지) | 내부 의존 ❌ | 텐서 누락 위험 | reopen 없음 |
| **close+reopen (1순위)** | 공개 API ✅ | device 통째 반환(확실) | **반복 in-process reopen = 지금 죽는 FW init 동작** |
| 프로세스 재시작(폴백) | 무손상 ✅ | **OS 보장** | 복잡·느림·warm 상태 상실 (venv 분리 아님) |

> **주의(독립 축)**: "회수 방식"과 "venv 통일 여부"는 별개다. 프로세스 재시작은 **같은 통일 venv**로
> subprocess만 모드별 env로 재spawn하면 되므로 venv 분리가 필요 없다. venv 분리는 오직 4.57이 3B를
> 깨뜨릴 때(Phase 1-1 게이트 실패)만 강제되는, 회수와 무관한 별도 폴백이다.

**1순위: close+reopen (ttnn 공개 API만, tt-metal·모델 내부 무접근):**
```python
def _switch_mode(new_mode):
    ttnn.synchronize_device(mesh_device)   # 진행 op 완료 (op 중 close 금지 = wedge 핵심)
    ttnn.close_mesh_device(mesh_device)    # device 전체(weights/KV/trace/program cache) 통째 반환
    mesh_device = _open_device()           # 서버 기존 함수 재사용
    _load_models_for(new_mode)             # try/except로 감싸 로드 실패 시 재close
```
- per-tensor 방식(`release_trace`+`deallocate_weights_and_kv`)은 `trace_ids`·`model.layers` 등
  **tt-metal 내부 의존** → 방침 위배 + 누락 위험 → **접음**. close_mesh_device가 통째 반환하니 불필요.
- 단점(정직히): reopen은 FW init 재실행 = **이번 세션 계속 실패하는 그 동작**. 위험을 "누수/OOM"에서
  "반복 reopen 안정성"으로 **옮긴 것**이지 없앤 게 아님. 속도 이점도 없음(새 모델 재컴파일 불가피).
  크래시 격리도 안 됨(같은 프로세스) → 로드 try/except 필수.

**게이트 (통일 vs 폴백 판단, Phase 1-3 실측)**: **한 프로세스에서 close+reopen 반복이 보드에서
안정적인가.** `synchronize→close`(=데모의 clean teardown)는 안전할 가능성 높으나, **연속 reopen**이
미검증. 안정 → 통일 단일 프로세스 확정. 불안정 → 모드 전환=프로세스 재시작(OS가 device 반환 보장).
→ **회수 방식을 교체 가능하게 설계**(close+reopen 먼저, 실패 시 프로세스 재시작 폴백).

---

## Phase 1 — 보드 복구 후 (실패 조기 발견 순)

1. **4.57.1 3B 추론 동일성** (Gate): `python_env_tf457`로 기존 3b+3b 서버 기동 → 캡션·질의가 4.53과
   동일한지. 통과 → 통일 확정. 실패 → 원인 파악, 최악 시 멀티프로세스 폴백(아래).
2. **Qwen3-VL-8B 단독 검증**(서버 배선 전): `demo_qwen3_vl.py`로 이미지 1장 캡션. 로드/메모리
   (BFP8로 12GiB)/vision encoder. **여기서 막히면 모드2 불가** → 조기 판단.
3. **모드 전환 검증**: 3b+3b → caption-8b → llm-8b 왕복. device DRAM 누수·OOM 없는지(0-5 실측).
4. **캡션 A/B**: 3B vs 8B 같은 프레임 (caption_viewer 재사용).

---

## 폴백 (두 게이트는 독립 — 각각 다른 것을 강제)

두 축이 별개이므로 폴백도 둘로 나뉜다:

**폴백 A — 회수 게이트 실패 (in-process reopen 불안정)**: venv는 **통일 그대로**. 모드 전환만
close+reopen(단일 프로세스) → **프로세스 재시작**으로 교체. 앞단 router가 백엔드 subprocess를 모드별
env로 재spawn(같은 통일 venv). 핸드오프는 **SIGTERM→device 완전 반환 대기→spawn**(겹치면 wedge).
회수는 OS 보장. 단점: 전환 느림·router/supervisor 코드. **venv 분리 없음.**

**폴백 B — 호환 게이트 실패 (4.57이 3B 추론을 깸, Phase 1-1)**: 이때만 venv 분리 강제 —
python_env(4.53, 모드1·3) + qwen3_venv(4.57, 모드2). 프로세스 경계가 따라오고 회수는 폴백 A와 동일.

→ 회수(A)와 venv(B)는 독립. A만 발동하면 통일 venv + 프로세스 재시작. B가 발동하면 venv도 분리.

---

## 리스크 정리
| 리스크 | 확인 | 현재 |
|---|---|---|
| 4.57 import/config 호환 | 보드 무관 | ✅ 통과 |
| 4.57 3B **추론** 동일성 | 보드 | Phase 1-1 |
| 8B 로드 + 메모리(BFP8/12GiB) | 보드 | Phase 1-2 |
| 언로드 DRAM 반환(통일 필수) | 조사 지금/실측 보드 | 0-5 → 1-3 |
