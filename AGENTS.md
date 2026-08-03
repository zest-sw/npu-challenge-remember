# AGENTS.md — ReMEmbR on an NPU

이 파일은 이 레포에서 처음 작업하는 AI 에이전트(어떤 도구든)를 위한 빠른 오리엔테이션이다.
자세한 내용은 각 절 끝의 문서로 가고, 여기엔 **직접 겪어서 확인한 것만** 남긴다.

## 이게 뭔지 (1분)

NVIDIA의 [ReMEmbR](README.upstream.md)(로봇용 장기 시공간 기억 시스템)를 클라우드/GPU 없이
**로컬 NPU(BOS 칩, Tenstorrent 호환 스택) 한 장**에서 돌리도록 포팅한 포크. `remembr/` 안 코드는
NVIDIA 원본, `tt_server/`가 이번 포팅으로 추가된 NPU 추론 서버. **여기서 시작**:
[docs/DEV_STATUS.md](docs/DEV_STATUS.md)(상태 보드·변경이력) → [docs/README.md](docs/README.md)(문서 지도).

## ⚠️ 절대 하지 말 것 (보드를 몇 시간 날림, 여러 번 확인됨)

0. **NPU 서버를 올리고 내리는 작업은 사람 사용자가 지정된 tmux 세션에서만 직접 수행한다.**
   AI 에이전트는 `run_server.sh` 실행, 새 서버/background process 기동, `tmux send-keys`,
   `Ctrl-C` 전송, signal/`kill`/`pkill`, 서버 재시작을 하지 않는다. 상태 확인이 필요하면
   `/health`, process 목록, tmux pane/log를 **읽기 전용**으로 조회할 수 있지만, lifecycle 변경이
   필요하면 작업을 멈추고 사용자에게 기동·종료를 요청한다. 사용자가 명시적으로 요청하더라도
   서버 lifecycle은 사용자가 tmux에서 직접 조작하는 원칙을 우선한다.
1. **서버에 `kill -9`(SIGKILL) 쓰지 말 것.** device teardown을 건너뛰어 보드가 FW-init 실패
   상태로 먹통이 된다. `Ctrl-C`(SIGINT)만 쓸 것 — 안전 종료가 구현돼 있다(아래 참고).
2. **`Ctrl-C` 연타하지 말 것.** teardown 도중 두 번째 인터럽트가 오면 같은 결과.
3. **PCIe FLR(`/sys/.../reset`)로 "리셋" 시도하지 말 것.** 정상 리셋이 아니라 보드를 D3cold로
   빠뜨려 PCI 버스에서 사라지게 만든다 — 재부팅밖에 답이 없어진다. 실제 겪은 일이다.
4. **VLM에 6프레임 이상 한 번에 넣지 말 것.** `qwen25_vl`의 vision attention이 다중 이미지에서
   검증된 적이 없고, Tensix 코어 16개가 데드락한다(FW init 영구 실패). 1프레임만 검증됨.
5. **공유 보드다.** `/dev/bos/0`는 프로세스 하나만 배타적으로 쥔다. 뭔가 시작하기 전에
   `curl localhost:8100/health`로 누가 쓰고 있는지 먼저 확인할 것. (`out/` 아래
   `jung_results`/`yu_results` 같은 폴더가 이미 있다 — 실제로 여러 명이 씀.)

**보드가 먹통이면** (`failed to initialize FW! Try resetting the board.`):
`sudo rmmod bos && sudo modprobe bos` → 안 되면 재부팅(NPU 먼저 켜고 PC). FLR은 위에서 금지.
자세히: [docs/TENSTORRENT_INTEGRATION.md](docs/TENSTORRENT_INTEGRATION.md)의 디버깅 로그.

## 서버 (`tt_server/tt_infer_server.py`)

- **소유권**: 서버 기동·종료·재시작은 사람 사용자가 tmux에서 직접 한다. 에이전트는 서버를
  조작하지 않고 `/health`와 로그만 확인하며, lifecycle 변경이 필요하면 사용자에게 요청한다.
- **내부 구조**: `tt_infer_server.py`는 호환 entrypoint이고 실제 코드는
  `presentation/http → application → domain` + `infrastructure/tenstorrent`로 분리돼 있다.
  route에서 엔진/device를 직접 참조하지 말고 `InferenceService`와 `RuntimePort` 경계를 유지할 것.
- **기동**: `bash tt_server/run_server.sh` (tt-metal 루트 아무 데서나 실행해도 됨 — 스크립트가
  알아서 `cd`한다. 직접 `env_set.sh`를 다른 위치에서 source하면 `TT_METAL_HOME`이 깨진다, 이미
  겪음). **모드 지정**: `TT_MODE=caption-8b bash run_server.sh` — 없으면 `3b+3b` 기본.
  잘못된 모드명은 device 건드리기 전에 즉시 실패한다(조용히 안 멈춤). launcher는 전역
  `flock` + `/health` + 기존 process/device holder를 먼저 검사하므로 이미 서버가 있으면 새로
  띄우지 않고 상태 메시지만 출력한다. 이 guard를 우회해 Python entrypoint를 직접 실행하지 말 것.
- **모드**: 보드 DRAM이 ~12GB라 한 번에 조합 하나만. `3b+3b`(기본) / `caption-3b` / `llm-8b` /
  `caption-8b`(Qwen3-VL-8B, `TT_PYTHON_ENV=tf457` 필요). 전환은 `POST /mode` — **요청이 자동으로
  모드를 안 바꾼다**, 안 맞으면 409. 전환은 device를 완전히 반환(`close_mesh_device`)했다가
  다시 여는 방식이라 자원 누수 없음(20+ 회 반복 검증됨).
- **종료**: `Ctrl-C` 후 몇 초~수십 초 걸릴 수 있다(진행 중이던 실제 device 작업은 끝까지 실행
  후 정리 — 정상). `TT_SHUTDOWN_TIMEOUT`(기본 5s)로 uvicorn이 무한 대기하던 버그는 고쳐짐.
- 상세: [tt_server/README.md](tt_server/README.md)(HTTP API), [tt_server/RUNBOOK.md](tt_server/RUNBOOK.md)(운영).

## 안 자명한 것들 (직접 겪은 것만)

- **`_name_or_path` 빈 문자열 버그(transformers 4.57).** tt-metal `qwen25_vl/tt/model_config.py`가
  이 필드로 model_name을 판별하는데, 4.57에선 빈 문자열이라 잘못된 AutoModel 클래스로 로드 시도.
  `model_config.py`에 1줄 패치 적용됨(하위호환, 4.53엔 영향 없음) — tt-metal 소스를 건드린
  몇 안 되는 곳 중 하나, load-bearing.
- **Milvus 컬렉션이 시퀀스 ID로만 이름 지어짐** (`eval_memory_{sequence_id}`, 실행자/프로세스
  구분 없음). 같은 시퀀스로 두 사람이 동시에 eval 돌리면 서로의 캡션이 섞여 **조용히** 틀린
  결과가 나온다. 에러도 크래시도 없음 — 알아채기 어렵다.
- **`remembr+` 에이전트 질문은 정상적으로 70~100초+ 걸린다.** 안 멈춘 것이다. 조급하게
  강제종료하면 위 규칙 1번을 밟게 된다.
- **eval.py는 `/mode`를 호출하지 않는다.** 서버가 지금 어떤 모드든 그대로 쓴다 — LLM 필요한데
  VLM 전용 모드면 계속 409.
- 서버 문법 강제 JSON(`response_format`)은 lm-format-enforcer로 host-side full-vocab logit
  마스킹 — 소형 모델의 깨진 JSON을 막아준다. **느리다고 오해하지 말 것**: 실측하면 같은
  토큰 수에서 device-side greedy와 차이가 ~3%다(8B: plain 28.6s vs json 29.6s). 토큰마다
  호스트로 읽는 513KB(128256×fp32)는 8B forward pass에 묻힌다.
- **지연은 거의 전부 "생성 토큰 수 × 토큰당 비용"이다.** 실측(8B, accuracy): 요청당 고정비
  ~3초 + **토큰당 ~0.26초(≈3.8 tok/s)**. 프롬프트 길이 영향은 작다(22→529 words에서
  3.0→4.3초). 느리면 프롬프트나 grammar를 의심하기 전에 **생성 토큰 수와 재시도 횟수**를 보라.
- **`temperature=0`은 재시도를 무의미하게 만든다.** 생성이 바이트 단위로 동일해져서 실패한
  답을 그대로 반복한다. 실측(동일 9문제, 8B): temp 0 → LLM 824초·attempts 25·미답변 4,
  temp 0.7 → **445초·12·0**. eval 기본값 0.7을 이유 없이 0으로 바꾸지 말 것.
- **grammar 스키마의 문자열에 `maxLength`가 없으면 모델이 그 안에서 폭주한다.** `position`에
  제한이 없어서 `[0, 0, 0], `를 끝없이 반복하다 max_tokens에 잘려 JSON이 깨졌고, 상한 없는
  재시도와 겹쳐 eval이 27분간 멈췄다([c997922]). 새 필드를 추가할 때 길이를 반드시 준다.
- tt-metal 소스는 원칙적으로 무손상 유지 — 편집한 곳은 위 model_name 패치가 거의 유일하고,
  이유가 `docs/DEV_STATUS.md` changelog에 기록돼 있다.

## 변경 후 검증

코드만 읽고 "될 것 같다"로 끝내지 말 것 — 이 세션에서 반복 확인된 것: 실제로 보드에서 돌려야
드러나는 문제가 많다(모델 로드 성공 ≠ 추론 성공, import 성공 ≠ 4.57 호환). 최소:
1. `curl localhost:8100/health` — ready 확인
2. 실제 요청 하나(`/caption` 또는 `/v1/chat/completions`)로 끝까지 응답 확인
3. 모드/서버를 건드렸으면 [docs/NPU_STABILITY_TEST_PLAN.md](docs/NPU_STABILITY_TEST_PLAN.md)의
   축소판(순차 요청 몇십 회 + 모드 전환 몇 회)만이라도

## 파일 어디에 두나 (홈 디렉터리 지도)

**코드는 전부 레포 안.** 예전에 스크립트를 `~/ycc/`에 뒀다가, 데이터가 옮겨지면서 스크립트
안 절대경로가 전부 죽어 있었다(아무도 몰랐다). 레포 밖 = git 추적 안 됨 = 다음 세션이 못 찾음.

| 무엇 | 어디 | 비고 |
|---|---|---|
| 파이프라인/도구 스크립트 | `remembr/scripts/` (Samsung 계열은 `scripts/samsung/`) | git 추적 |
| 보드·서버 스트레스 테스트 | `tt_server/tests/board/` | 실제 device 필요, 일반 테스트와 분리 |
| 생성 산출물(HTML 뷰어·비교 결과·로그) | `out/` | **gitignored.** `/tmp`에 두지 말 것 — 재부팅 시 소멸 |
| 데이터셋 | `remembr/data/` (레포 안, gitignored) | Samsung CODa: `remembr/data/samsung_dataset/0/` |

**홈에 있고 옮기면 안 되는 것들** (실제로 확인한 이유가 있음):

| 경로 | 왜 그대로 둬야 하나 |
|---|---|
| `~/milvus` | 실행 중인 `milvus-standalone` 컨테이너가 bind-mount 중. 옮기면 벡터 DB가 깨진다 |
| `~/coda_raw` (85G) | `eval.md`와 스크립트 docstring에 절대경로로 박혀 있다 |
| `~/tt-metal` | `TT_METAL_HOME`, weight 캐시(`model_cache/`, 39G), venv 2개가 여기 기준 |
| `~/ycc/samsung_dataset/.venv` | rosbags·kiss-icp 전용 venv. venv는 절대경로를 내장해서 옮기면 깨진다 |
| `~/ycc/samsung_demo3`, `samsung_demo4` | 원본 rosbag. `scripts/samsung/01_*.py`가 참조 |

**weight 캐시는 상대경로다.** `model_config.py:476`이 `model_cache/...`를 CWD 기준으로 잡아서,
서버를 다른 디렉터리에서 띄우면 6.9G 캐시가 거기 또 생긴다(실제로 2벌 중복돼 있었다).
`run_server.sh`의 `cd "$TT_METAL_HOME"`가 이걸 막는다 — 지우지 말 것.
`TT_CACHE_PATH`로 고정하려 하지 말 것: 같은 파일 478행이 device 이름만 붙이고 **모델 이름을
안 붙여서** Llama와 Qwen 캐시가 한 경로에서 충돌한다.

## 문서 지도

| 파일 | 용도 |
|---|---|
| [docs/DEV_STATUS.md](docs/DEV_STATUS.md) | **상태 보드 + 변경 이력.** 작업 전후로 항상 갱신 |
| [docs/TENSTORRENT_INTEGRATION.md](docs/TENSTORRENT_INTEGRATION.md) | 아키텍처, 디버깅 로그(보드 wedge 사례들) |
| [docs/VLM_MODE_SWITCHING_PLAN.md](docs/VLM_MODE_SWITCHING_PLAN.md) | 모드 시스템 설계 배경 |
| [docs/NPU_STABILITY_TEST_PLAN.md](docs/NPU_STABILITY_TEST_PLAN.md) | 안정성 테스트 계획+결과 |
| [docs/PAPER_EVAL_RUNBOOK.md](docs/PAPER_EVAL_RUNBOOK.md) | NaVQA/CODa 평가 재현 |
| [docs/WORKLOG.md](docs/WORKLOG.md) | eval 디버깅 상세 로그 |
| [tt_server/README.md](tt_server/README.md) / [RUNBOOK.md](tt_server/RUNBOOK.md) | 서버 API / 운영 |
| [README.upstream.md](README.upstream.md) | NVIDIA 원본 README(그대로 보존) |

## 작업 원칙 (이 레포에서 확인된 사용자 선호)

- **tt-metal 소스는 최대한 안 건드림.** 꼭 필요한 최소 수정만, 이유를 주석+changelog에 남김.
- **우회보다 정석.** 기존 코드/구조를 재사용하고, 비슷하게 새로 안 만듦.
- 문서는 전부 **한국어로 통일**돼 있음(원본 NVIDIA README만 예외로 영어 보존).

## 커밋 & 문서화 정책

작업하는 AI 세션(도구 무관)은 아래를 기본 동작으로 삼는다 — 매번 사용자가 "커밋해줘"라고
말할 필요 없이, **검증된 변경 단위가 끝날 때마다** 스스로 커밋하고 문서를 갱신한다.

**브랜치 — 모든 수정은 논리적 변경 단위별 작업 브랜치에서 한다:**
- `main`에서 직접 파일을 수정하거나 커밋하지 않는다. 작업 시작 전에 working tree를 확인하고,
  최신 `main`을 기준으로 **한 가지 논리적 변경만 담는 새 브랜치**를 만든다. 선행 변경 위에
  쌓아야 하는 stacked 작업은 사용자가 명시했을 때만 예외로 한다.
- 브랜치 이름은 변경 성격과 목적이 드러나게 짓는다: `feat/<name>`, `fix/<name>`,
  `refactor/<name>`, `docs/<name>`, `test/<name>`, `chore/<name>`.
- 진행 중 별개의 수정 요청이 생기면 기존 브랜치에 섞지 않는다. 현재 변경을 검증·커밋해
  정리하거나 안전하게 보존한 뒤, `main`에서 별도 브랜치를 만든다.
- 한 브랜치에는 관련 코드·테스트·문서만 포함한다. 기존 dirty worktree나 다른 사람의 변경을
  새 브랜치에 조용히 끌고 가지 말고, 겹치면 먼저 범위를 확인한다.
- 검증된 변경은 작업 브랜치에 커밋하고 원격에 push한 뒤 PR로 `main`에 병합한다.
  `main` 직접 push, 서로 무관한 변경의 한 PR 묶기, 검증 전 병합은 하지 않는다.

**커밋 시점 — "논리적으로 하나인 검증된 변경" 단위로, 크게 묶지 않는다:**
- 버그 수정 하나, 기능 하나, 문서 정리 하나 = 커밋 하나. (예: 이 세션에서 실제로 이렇게
  나눔 — `run_server.sh` 경로 버그 / `.gitignore` / 엔진 리팩터 / qwen3 배선 / uvicorn
  shutdown 버그 / `TT_MODE` 노출, 전부 별도 커밋.)
- **커밋 전 반드시 검증부터.** "될 것 같다"가 아니라 실제로 돌려서 확인한 뒤 커밋한다
  (위 "변경 후 검증" 절 참고). 검증 안 된 상태로 커밋하지 않는다.
- tt-metal처럼 다른 저장소를 건드렸다면 그 저장소에도 별도로 커밋한다(레포 섞지 않음).
- **하지 말 것**: 작업 중간 상태 커밋, `--amend`로 기존 커밋 덮어쓰기(사용자가 명시 요청 안
  하는 한), `.env`/토큰 등 비밀 포함 여부 확인 없이 `git add -A`.
- push는 원격에 반영해도 안전한 시점에 한다(로컬 커밋 자체를 미루지 않는다).

**문서 갱신 — 커밋과 같은 타이밍에, 별도로 챙기지 않아도 되게:**
- **load-bearing 변경**(버그 수정, 새 기능, 새로 발견한 함정)은 예외 없이
  [docs/DEV_STATUS.md](docs/DEV_STATUS.md) `## 6. Changelog`에 한 항목 추가. 형식은 기존
  항목들 참고(무엇을·왜·어떻게 검증했는지).
  - 그 변경이 어떤 하위 시스템(서버 모드, tt-metal 통합, eval 파이프라인 등)을 다루는 전용
    문서가 이미 있으면 그 문서도 같이 갱신한다 — changelog 한 줄로 끝내지 않는다.
  - 이 파일(`AGENTS.md`)도 갱신 대상이다. 새로 겪은 보드 wedge 원인이나 새로 발견한
    비직관적 버그는 "안 자명한 것들" 절에 추가한다.
- 문서 갱신 없이 코드만 커밋하지 않는다 — 특히 이 레포는 여러 세션/여러 사람이 이어받는
  구조라, 문서가 낡으면 다음 세션이 이미 고친 걸 또 고치거나 이미 실패한 걸 또 시도한다
  (이 세션에서 실제로 여러 번 있었던 일).
