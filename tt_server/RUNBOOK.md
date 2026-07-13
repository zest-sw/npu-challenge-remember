# ReMEmbR on Tenstorrent NPU — 실행 가이드 (처음부터)

VLM(캡션)=Qwen2.5-VL-3B, LLM(추론)=Llama-3.2-3B 를 **Tenstorrent 보드 한 장**에서
띄우고, remembr 코어 파이프라인(`이미지 → 캡션 → 메모리 저장 → 질의응답`)을
end-to-end로 돌리는 절차입니다.

---

## 0. 구조 한눈에

```
 remembr (remembr_venv)                     tt_infer_server.py (tt-metal python_env, 보드 소유)
   NonAgent ── HTTP /v1/chat/completions ─────► Llama-3.2-3B  ┐  프로세스 1개, device 1회 open,
   Captioner ─ HTTP /caption ─────────────────► Qwen2.5-VL-3B ┘  두 모델 co-resident,
                                                                 단일 워커+큐로 모든 NPU 실행 직렬화
```

- **venv 2개**를 쓴다. 서버는 tt-metal `python_env`(ttnn/transformers), remembr는
  자기 전용 venv(langchain/pymilvus). 버전이 충돌하므로 HTTP로 분리한다.
- NPU는 한 순간에 그래프 하나만 실행 → 캡션과 LLM은 큐로 자동 직렬화된다(동시 요청 안전).

---

## 1. 사전 준비 (최초 1회)

### 1-1. HF 토큰 (gated 모델)
`meta-llama/Llama-3.2-3B-Instruct`는 gated. 토큰이 `~/.bashrc`에 `export HF_TOKEN=...`
형태로 있어야 한다. `run_server.sh`가 비대화형 셸에서도 이 줄을 직접 읽어온다.
(Qwen2.5-VL-3B는 open이라 토큰 불필요.)

### 1-2. Milvus (메모리 DB)
```bash
# remembr 레포의 docker 스크립트로 기동 (127.0.0.1:19530)
bash /home/user/remembr/launch_milvus_container.sh start   # 경로는 레포 기준
docker ps | grep milvus                                    # 기동 확인
```
최초 실행 시 임베딩 모델 `mxbai-embed-large-v1`이 자동 다운로드된다.

### 1-3. remembr 전용 venv 만들기
```bash
python3 -m venv ~/remembr_venv
source ~/remembr_venv/bin/activate
pip install --upgrade pip          # ★ 먼저 pip 업그레이드 (구버전 pip 리졸버 버그 회피)

pip install "langchain-core==0.2.40" "langchain-community==0.2.16" \
            "langchain-openai==0.1.25" "langchain==0.2.16" \
            "langchain-huggingface==0.0.3" \
            pillow requests pymilvus "sentence-transformers==3.1.0"
```
> pip가 `AssertionError: len(weights) == expected_node_count`로 죽으면 구버전 pip
> 버그다. 위처럼 `pip install --upgrade pip`를 먼저 하면 해결된다. 그래도 안 되면
> `--use-deprecated=legacy-resolver` 플래그를 붙이거나 패키지를 나눠서 설치한다.

### 1-4. 서버 쪽 추가 의존성 (tt-metal python_env에 최초 1회)
```bash
source /home/user/tt-metal/env_set.sh          # python_env 활성화
pip install fastapi "uvicorn[standard]" lm-format-enforcer
```
`lm-format-enforcer`는 JSON 제약 디코딩(Ollama `format="json"` 대체)에 필요하다.

---

## 2. NPU 서버 기동 (tt-metal python_env)

```bash
cd /home/user/tt-metal
TT_LLM_MAX_SEQ_LEN=2048 TT_VLM_MAX_SEQ_LEN=1024 \
  bash /home/user/remembr/tt_server/run_server.sh
```

- **`TT_LLM_MAX_SEQ_LEN=2048` 는 필수.** NonAgent 프롬프트가 2048까지 패딩되므로
  기본값(1024)이면 `Padded prefill end idx 2048 exceeds max seq len` 에러가 난다.
- VLM은 1024면 충분(이미지는 서버에서 224×224로 리사이즈).
- 두 모델 로드 + weight 캐시 빌드로 **수 분** 걸린다. 아래 로그가 뜨면 준비 완료:
  ```
  INFO:     Application startup complete.
  ```

### 준비 상태 확인 (다른 터미널에서)
```bash
curl -s http://localhost:8100/health
# -> {"ready":true,"llm":true,"vlm":true}
```

### 단독 동작 확인 (선택)
```bash
# LLM
curl -s http://localhost:8100/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Reply with JSON {\"ok\":true}"}]}'
```

---

## 3. 스모크 테스트 실행 (remembr_venv)

```bash
source ~/remembr_venv/bin/activate
cd /home/user/remembr
export OPENAI_API_KEY=local
PYTHONPATH=$(pwd) python examples/tt_smoke_test.py <이미지경로1> <이미지경로2> ...
```

이미지 경로 예시(레포에 있는 데모 이미지 사용):
```bash
PYTHONPATH=$(pwd) python examples/tt_smoke_test.py \
  /home/user/tt-metal/models/bos_model/qwen25_vl/demo/images/*.jpg
```
인자 없이 실행하면 합성(단색) 프레임으로 경로만 점검한다.

기대 출력:
```
[1/4] Captioning ... on the NPU (Qwen2.5-VL)...
      caption: <실제 캡션>
[2/4] Inserting into Milvus memory...
[3/4] Building NonAgent (Llama-3.2-3B on the NPU)...
[4/4] Querying...
      AgentOutput: AgentOutput(type=..., text=..., ...)
OK: end-to-end path completed.
```

---

## 4. 종료

```bash
pkill -f tt_infer_server.py          # 정상 종료(SIGTERM) — device를 깔끔히 닫음
# 또는
kill $(lsof -t -i:8100)

lsof -i :8100                        # 결과 없으면 완전히 내려간 것
```
안 죽으면 그때만 `pkill -9 -f tt_infer_server.py` (단, `-9`는 device를 못 닫고 죽어
다음 기동 시 드물게 보드 리셋 필요 → 우선 `-9` 없이 시도).

---

## 5. 트러블슈팅

| 증상 | 원인 / 해결 |
|------|------------|
| **첫 캡션이 매우 느림(수 분), 클라이언트 타임아웃** | fresh 서버의 **첫 요청은 ttnn trace(prefill/decode)를 처음 컴파일**하는 일회성 비용. 데드락 아님(서버 CPU가 계속 돈다). 끝나면 이후 요청은 훨씬 빠르다. 클라이언트 타임아웃은 기본 900초이며 `TT_VLM_TIMEOUT`(초)로 조절. |
| `address already in use (8100)` | 서버가 이미 떠 있음. 새로 안 띄워도 됨. 내리려면 4번 참고. |
| `Padded prefill end idx 2048 exceeds max seq len` | 서버를 `TT_LLM_MAX_SEQ_LEN=2048`로 재기동. |
| LLM이 코드펜스/깨진 JSON, NonAgent 재시도 무한루프 | JSON 제약 디코딩 미적용. `lm-format-enforcer` 설치 확인(1-4). |
| HF 403 / gated 오류 | `~/.bashrc`의 `HF_TOKEN` 확인. `run_server.sh`가 읽어온다. |
| OOM | 이 보드는 DRAM ~12GB(3 bank)만 노출. 3B+3B는 co-fit 되지만 8B LLM은 안 됨 → 시간분리 폴백(`TT_ENABLE_VLM=1 TT_ENABLE_LLM=0`로 캡션 → 스왑) 또는 작은 모델. |
| pip `AssertionError ... expected_node_count` | 구버전 pip 리졸버 버그. `pip install --upgrade pip` 후 재설치(1-3). |

---

## 6. 주요 env 정리

**서버 (`run_server.sh` / tt_infer_server.py)**

| var | 기본값 | 설명 |
|-----|--------|------|
| `TT_LLM_MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | LLM |
| `TT_VLM_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | VLM |
| `TT_LLM_MAX_SEQ_LEN` | `1024` (→ **2048로 실행**) | NonAgent는 2048 필요 |
| `TT_VLM_MAX_SEQ_LEN` | `1024` | 캡션 |
| `TT_ENABLE_LLM` / `TT_ENABLE_VLM` | `1` / `1` | 폴백 시 한 개만 로드 |
| `TT_TRACE_REGION_SIZE` | `15000000` | 모델별 trace 버퍼 |
| `TT_PORT` | `8100` | |

**remembr 측**

| var | 기본값 | 설명 |
|-----|--------|------|
| `TT_LLM_BASE_URL` | `http://localhost:8100/v1` | NonAgent → 서버 |
| `TT_VLM_URL` | `http://localhost:8100/caption` | Captioner → 서버 |
| `TT_VLM_TIMEOUT` | `900` | 캡션 클라이언트 타임아웃(초) |
| `TT_LLM_MAX_TOKENS` | `256` | LLM 최대 생성 토큰 |
