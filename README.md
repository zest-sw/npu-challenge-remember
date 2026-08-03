<h1 align="center">ReMEmbR on an NPU</h1>

<p align="center">
<a href="docs/DEV_STATUS.md">상태</a> ·
<a href="docs/PAPER_EVAL_RUNBOOK.md">논문 eval</a> ·
<a href="tt_server/RUNBOOK.md">서버 실행</a> ·
<a href="docs/TENSTORRENT_INTEGRATION.md">아키텍처</a> ·
<a href="README.upstream.md">Upstream README</a>
</p>

**[ReMEmbR](https://github.com/NVIDIA-AI-IOT/remembr)**(NVIDIA의 로봇용 장기 시공간 메모리)를
전체 파이프라인이 **로컬 NPU에서** 돌아가도록 포팅한 것 — 클라우드 LLM도, 별도 GPU도 없이.

Upstream은 **VILA-1.5 (13B)**로 캡션을 만들고 **GPT-4o / llama3.1-8B**로 추론합니다. 이 포트에서는
두 모델 모두 소형·로컬이며 단일 보드에 함께 상주합니다:

| | Upstream | 이 포크 |
|---|---|---|
| Captioner (VLM) | VILA 1.5 13B | **Qwen2.5-VL-3B** (NPU) |
| Reasoner (LLM) | GPT-4o / llama3.1 8B | **Llama-3.2-3B** (NPU) |
| Memory | Milvus + `mxbai-embed-large-v1` | 동일 |
| Eval | NaVQA / CODa | 동일 |

> 원본 프로젝트의 README는 **[README.upstream.md](README.upstream.md)**에 그대로 보존돼 있습니다 —
> NVIDIA가 문서화한 setup, usage, evaluation. 아래 내용은 전부 NPU 포트에 한정된 것입니다.

## 어떻게 도는가

```
 remembr (remembr/remembr_venv)             tt_server/tt_infer_server.py (tt-metal python_env, 보드 소유)
   NonAgent ── HTTP /v1/chat/completions ────► Llama-3.2-3B  ┐  프로세스 1개, device를 1회만 open,
   Captioner ─ HTTP /caption ────────────────► Qwen2.5-VL-3B ┘  두 모델 co-resident,
                                                                모든 NPU op는 단일 워커 큐로 직렬화
```

두 스택은 의존성 핀이 충돌하고(ttnn/transformers vs langchain/pymilvus), device는 한 프로세스만
쥘 수 있어서, 모델들은 작은 HTTP 서버 뒤에 두고 remembr가 그걸 호출합니다. API는
[tt_server/README.md](tt_server/README.md), 그 이유는
[docs/TENSTORRENT_INTEGRATION.md](docs/TENSTORRENT_INTEGRATION.md) 참고.

## 하드웨어

이 보드는 tt-metal 호환 스택으로 구동되는 **BOS 칩**(Tenstorrent 아님)이라, 로그에는
`BlackholeTTDevice`가 찍히지만 device 노드는 `/dev/bos/0`, 커널 모듈은 `bos`입니다.
`python tt_server/npu_info.py`로 실측:

| | |
|---|---|
| arch | `BLACKHOLE`, device 1개 |
| DRAM | **12 GiB** (4 GiB × 3 banks) |
| Tensix grid | 5×4 = 20 cores |
| L1 | 1.5 MB / worker core |

모델이 3B인 이유가 이 12 GiB입니다: Llama-3.2-3B + Qwen2.5-VL-3B는 함께 들어가지만, 8B LLM은
VLM과 공존이 안 됩니다 — 그리고 로봇은 둘 다 동시에 상주해야 합니다.

## 빠른 시작

```bash
# 1) NPU 서버 (venv는 run_server.sh가 알아서 켠다 — 미리 활성화할 필요 없음)
cd /home/user/remembr/tt_server
TT_LLM_MAX_SEQ_LEN=2048 TT_VLM_MAX_SEQ_LEN=1024 ./run_server.sh
curl -s localhost:8100/health          # {"ready":true,"llm":true,"vlm":true}

# 2) end-to-end 스모크 테스트 (remembr_venv)
cd /home/user/remembr && source remembr_venv/bin/activate
export OPENAI_API_KEY=local
PYTHONPATH=$(pwd) python examples/tt_smoke_test.py path/to/frame.jpg
```

최초 셋업(venv, 핀, HF 토큰, Milvus)과 **성능에 직결되는 profiler-build 체크**:
[tt_server/RUNBOOK.md](tt_server/RUNBOOK.md).

> **서버는 `Ctrl-C` 한 번으로 내리고** `NPU released cleanly`를 기다리세요.
> 절대 `kill -9` 하지 마세요 — device teardown을 건너뛰어 보드가 먹통이 됩니다.

## 문서

| | |
|---|---|
| [docs/DEV_STATUS.md](docs/DEV_STATUS.md) | **여기서 시작.** 상태 보드, 알려진 이슈, 설계 결정, 변경 이력 |
| [docs/TENSTORRENT_INTEGRATION.md](docs/TENSTORRENT_INTEGRATION.md) | 아키텍처, 설계 결정, 디버깅 로그 |
| [docs/PAPER_EVAL_RUNBOOK.md](docs/PAPER_EVAL_RUNBOOK.md) | NPU에서 NaVQA/CODa 평가 재현 |
| [tt_server/README.md](tt_server/README.md) | 서버 구조, HTTP API, 환경변수 |
| [tt_server/RUNBOOK.md](tt_server/RUNBOOK.md) | 셋업, 시작/종료, 트러블슈팅, 보드 복구 |

## 추가된 것

```
tt_server/
├── tt_infer_server.py                    # 통합 NPU 추론 서버 (LLM + VLM)
├── run_server.sh                         # 런처 (env, Inspector off)
└── npu_info.py                           # 하드웨어/스펙 덤프
remembr/agents/non_agent.py               # + 'tt/<model>' 백엔드 -> 서버
remembr/captioners/tenstorrent_captioner.py   # Captioner -> POST /caption
remembr/scripts/preprocess_coda_rosfree.py    # ROS-free CODa -> .pkl 추출기
remembr/scripts/preprocess_captions.py    # + --captioner tenstorrent, checkpoint/resume
examples/tt_smoke_test.py                 # image -> caption -> memory -> query
```

tt-metal SDK는 있는 그대로 사용합니다(`PYTHONPATH`로 import). 그 소스 트리 안으로 복사하거나
수정하는 것은 없습니다.
