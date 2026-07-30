# NaVQA 평가 실행 가이드

이 문서는 현재 브랜치의 `remembr/scripts/preprocess_captions.py`와
`remembr/scripts/eval.py` 기준으로 CODa/NaVQA 데이터를 캡션화하고 평가하는 절차를 정리한다.

## 0. 실행 환경

Tenstorrent backend를 쓸 때는 서버와 client/eval을 서로 다른 Python 환경에서 실행한다.

```text
Terminal A: TT server
  - tt-metal의 python_env 사용
  - `bash tt_server/run_server.sh`로 실행
  - `/caption`, `/v1/chat/completions`를 제공

Terminal B: remembr client/eval
  - remembr venv 사용
  - `preprocess_captions.py`, `form_question_jsons.py`, `eval.py` 실행
  - 데이터 파일을 직접 읽고, TT server에는 HTTP 요청만 보냄
```

서버가 준비됐는지는 다음으로 확인한다.

```bash
curl -s localhost:8100/health
```

captioning을 하려면 health 응답에서 `vlm`이 `true`여야 하고, `tt/` LLM eval을 하려면 `llm`이 `true`여야 한다.

`eval.py --model`에서 `tt/` prefix가 없으면 TT server가 아니라 Ollama(`localhost:11434`)로 연결된다. 예를 들어 `optimal+tt/Llama-3.2-3B-Instruct`는 TT server를 쓰지만, `optimal+llama3`는 Ollama를 쓴다.

`optimal+...` 평가는 question JSON의 oracle context를 쓰므로 retrieval tool 호출은 하지 않는다. 반면 `remembr+...` 평가는 Milvus 기반 memory/retrieval 경로를 사용하므로 Milvus가 필요하다.

## 1. CODa/NaVQA 데이터 준비

평가에는 세 종류의 데이터가 필요하다.

```text
1. CODa raw sequence
2. CODa frame별 .pkl 파일
3. NaVQA question/answer CSV (`data/navqa/data.csv`)
```

NaVQA CSV는 repo의 data root 아래에 있어야 한다.

```text
data/navqa/data.csv
```

### 1-1. CODa devkit과 raw sequence 다운로드

처음부터 데이터를 준비한다면 ReMEmbR repo 바깥의 큰 디스크에 CODa devkit을 받는 것을 권장한다. CODa는 전체 크기가 크고, 전처리 중 중간 산출물도 생기므로 충분한 여유 공간이 필요하다.

```bash
git clone https://github.com/ut-amrl/coda-devkit.git
cd coda-devkit
mkdir -p data
```

원본 ReMEmbR 평가에서 쓰는 sequence ID는 다음 7개다.

```text
0, 3, 4, 6, 16, 21, 22
```

편의를 위해 다음 환경변수를 설정해 둘 수 있다.

```bash
export CODA_ROOT_DIR=/path/to/coda-devkit/data
export REMEMBR_PATH=/path/to/npu-challenge-remember/remembr
```

CODa devkit의 download script를 쓰려면 CODa devkit 환경도 필요하다.

```bash
cd /path/to/coda-devkit
conda env create -f environment.yml
conda activate coda
```

기존 upstream 방식은 `remembr/scripts/bash_scripts/preprocess_coda_all.sh`가 sequence 다운로드, 원본 `preprocess_coda.py` 실행, raw data 삭제까지 한 번에 처리한다.

```bash
cd /path/to/npu-challenge-remember/remembr
bash scripts/bash_scripts/preprocess_coda_all.sh
```

다만 원본 `preprocess_coda.py`는 ROS 의존성이 크다. 현재 NPU 평가 경로에서는 raw CODa를 직접 받은 뒤, 아래의 ROS-free extractor를 쓰는 방식을 권장한다.

### 1-2. ROS-free `.pkl` 추출

현재 NPU 평가 파이프라인은 raw CODa를 직접 caption하지 않는다. 먼저 `preprocess_coda_rosfree.py`로 frame별 `.pkl`을 만들어야 한다.

raw CODa sequence는 다음처럼 `2d_rect/`, `poses/` 등을 포함한 디렉토리라고 가정한다.

```text
/home/user/coda_raw/seq0/
├── 2d_rect/
├── poses/
└── ...
```

`.pkl` 추출:

```bash
python remembr/scripts/preprocess_coda_rosfree.py \
  --seq_id 0 \
  --coda_root /home/user/coda_raw/seq0 \
  --out_dir /home/user/coda_raw/coda_data/0 \
  --frame_skip 1
```

출력은 다음 구조가 된다.

```text
/home/user/coda_raw/coda_data/0/{timestamp}.pkl
```

각 `.pkl`에는 `cam0`, `position`, `rotation`, `timestamp`가 들어간다. 이 파일들이 이후 captioning과 eval의 `--data_path`/`--coda_dir` 입력이 된다.

## 2. 입력 데이터 구조

`preprocess_captions.py`는 raw CODa 디렉토리를 직접 읽지 않고, 이미 `.pkl`로 전처리된 데이터를 기대한다.

```text
{coda_dir}/{sequence_id}/*.pkl
```

예를 들어 `--coda_dir /home/user/coda_raw/coda_data --sequence_id 0`이면 실제 입력은 다음 위치다.

```text
/home/user/coda_raw/coda_data/0/*.pkl
```

`eval.py`는 `--data_dir` 아래에서 question과 caption JSON을 찾는다.

```text
{data_dir}/questions/{sequence_id}/{qa_file}.json
{data_dir}/captions/{sequence_id}/captions/{caption_file}.json
```

`--caption_file`에는 `.json` 확장자를 붙이지 않는다.

## 3. Caption 생성

repo root에서 실행한다.

```bash
python remembr/scripts/preprocess_captions.py \
  --captioner tenstorrent \
  --seq_id 0 \
  --seconds_per_caption 3 \
  --num-video-frames 1 \
  --captioner_name tt-qwen25-vl-3b \
  --data_path /home/user/coda_raw/coda_data \
  --out_path data/captions/0/captions
```

주요 인자:

- `--captioner`: `vila` 또는 `tenstorrent`. Tenstorrent NPU 서버를 쓰려면 `tenstorrent`.
- `--seq_id`: CODa sequence ID.
- `--data_path`: `{seq_id}/*.pkl` 폴더들의 부모 디렉토리.
- `--out_path`: caption JSON을 저장할 디렉토리. eval 기본 구조는 `data/captions/{seq_id}/captions`.
- `--captioner_name`: 결과 파일 이름에 들어가는 captioner label.
- `--seconds_per_caption`: 몇 초 단위로 frame들을 묶어 caption 하나를 만들지 결정.
- `--num-video-frames`: segment에서 captioner에 넘길 frame 수.

출력 파일 이름은 다음 형식이다.

```text
captions_{captioner_name}_{seconds_per_caption}_secs.json
```

긴 실행이 중간에 끊겨도 각 caption item은 먼저 JSONL checkpoint에 저장된다.

```text
captions_{captioner_name}_{seconds_per_caption}_secs.json.partial.jsonl
```

같은 명령을 다시 실행하면 이미 완료된 segment는 건너뛰고 이어서 진행한다. 전체가 끝나면 최종 JSON을 쓰고 partial 파일은 삭제된다.

각 caption item에는 다음 정보가 들어간다.

```json
{
  "id": ".../1673884197.590497.pkl",
  "position": [1.30, 1.99, -0.14],
  "theta": 3.14,
  "time": 1673884199.04,
  "caption": "...",
  "caption_latency_s": 30.65,
  "file_start": "1673884197.590497.pkl",
  "file_end": "1673884197.690506.pkl",
  "text_embedding": [...]
}
```

`caption_latency_s`는 client-side에서 `captioner_model.caption(images)` 호출에 걸린 wall-clock 시간이다. Tenstorrent captioner 기준으로는 `/caption` HTTP 요청을 보내고 응답을 받을 때까지의 시간이며, `text_embedding` 계산과 JSON write 시간은 포함하지 않는다.

## 4. Question JSON 생성

caption JSON을 만든 뒤 question JSON을 생성한다.

```bash
python remembr/scripts/question_scripts/form_question_jsons.py \
  --caption_file captions_tt-qwen25-vl-3b_3_secs
```

이 단계는 `data/navqa/data.csv`와 caption JSON을 이용해 다음 파일을 만든다.

```text
data/questions/{sequence_id}/human_qa.json
```

caption을 다시 생성했거나 `seconds_per_caption`, `captioner_name`이 바뀌면 question JSON도 다시 생성해야 한다. `human_qa.json` 안의 optimal context가 caption 파일에서 추출되기 때문이다.

## 5. 평가 실행

`eval.py`는 `--model` 값에 따라 backend가 달라진다.

- `optimal+tt/Llama-3.2-3B-Instruct`: question JSON의 oracle context를 쓰고 TT 서버 LLM으로 답변.
- `remembr+tt/Llama-3.2-3B-Instruct`: Milvus retrieval agent를 쓰고 TT 서버 LLM으로 답변.
- `optimal+llama3`, `remembr+llama3` 등 `tt/` prefix가 없으면 Ollama backend로 간다.

Tenstorrent LLM 서버를 쓰려면 `tt/` prefix가 반드시 필요하다.

```bash
python remembr/scripts/eval.py \
  --model optimal+tt/Llama-3.2-3B-Instruct \
  --sequence_id 0 \
  --qa_file human_qa \
  --caption_file captions_tt-qwen25-vl-3b_3_secs \
  --data_dir /home/user/npu-challenge-remember/data \
  --coda_dir /home/user/coda_raw/coda_data \
  --out_dir /home/cho/eval_out \
  --postfix _latency_test
```

주요 인자:

- `--model`: 사용할 eval 방식과 LLM backend.
- `--sequence_id`: 평가할 sequence ID.
- `--qa_file`: `{data_dir}/questions/{sequence_id}/{qa_file}.json`의 basename.
- `--caption_file`: caption JSON basename. `.json`은 붙이지 않는다.
- `--data_dir`: `questions/`와 `captions/`를 포함하는 data root.
- `--coda_dir`: `{sequence_id}/*.pkl` 폴더들의 부모 디렉토리.
- `--out_dir`: 결과 JSON 저장 root. 실행 계정이 쓸 수 있는 경로를 권장한다.
- `--postfix`: 결과 파일 이름 suffix.
- `--temperature`: LLM temperature. 기본값은 `0.7`.
- `--num_ctx`: LLM context 길이.

결과 파일은 다음 위치에 저장된다.

```text
{out_dir}/{sequence_id}/{qa_file}/{model}__{caption_file}{postfix}.json
```

파일 이름에서 `/`는 `_`로 바뀐다. 예를 들어 `optimal+tt/Llama-3.2-3B-Instruct`는 다음처럼 저장된다.

```text
optimal+tt_Llama-3.2-3B-Instruct__captions_tt-qwen25-vl-3b_3_secs_latency_test.json
```

## 6. Eval 결과 JSON의 latency 필드

결과 JSON 최상위에는 전체 eval 실행 시간이 저장된다.

```json
{
  "version": 0.1,
  "eval_total_latency_s": 1136.88,
  "responses": [...]
}
```

각 scored response에는 다음 latency metadata가 저장된다.

```json
{
  "elapsed": 38.88,
  "query_latency_s": 38.88,
  "last_attempt_latency_s": 15.86,
  "attempts": 3,
  "attempt_latencies": [
    {
      "attempt": 1,
      "latency_s": 12.10,
      "status": "ok",
      "unanswered": true,
      "error": null
    }
  ],
  "eval_latency_s": 38.94
}
```

필드 의미:

- `eval_total_latency_s`: `eval.py` 전체 실행 wall-clock 시간.
- `eval_latency_s`: 질문 하나를 처리하는 전체 client-side 시간. memory loading, `agent.set_memory`, query attempt, scoring이 포함된다.
- `query_latency_s`: `model.query(question)`에 사용한 총 시간. retry와 exception attempt도 포함한다.
- `elapsed`: 기존 호환성을 위해 남겨둔 필드이며 현재는 `query_latency_s`와 같은 누적 query 시간이다.
- `last_attempt_latency_s`: 마지막 attempt 하나의 시간. 기존 `elapsed`가 표현하던 값에 가장 가깝다.
- `attempts`: 실제 query attempt 횟수.
- `attempt_latencies`: attempt별 상세 기록.

`attempt_latencies[*].status`는 다음 값을 가진다.

- `ok`: `model.query()`가 반환했고 `evaluate_output`까지 진행됨.
- `exception`: 해당 attempt 중 exception 발생.

`unanswered`는 모델 응답은 있었지만 해당 question type에 필요한 typed field를 채우지 못해 retry 대상이 되었는지를 나타낸다.

text question은 현재 eval에서 채점하지 않고 skip한다. 이 경우 빈 dict 대신 다음 metadata를 저장한다.

```json
{
  "unscored": true,
  "skipped": true,
  "skip_reason": "text_question",
  "eval_latency_s": 0.001,
  "question": "...",
  "id": "..."
}
```

## 7. 주의 사항

- client-side latency는 서버 내부 breakdown이 아니다. prefill, decode, queue wait 같은 세부 시간은 아직 `tt_server/tt_infer_server.py`에서 따로 기록하지 않는다.
- `caption_latency_s`는 captioner 호출 시간만 측정한다. embedding이나 JSON write까지 포함한 전체 preprocessing throughput을 보려면 별도 계측이 필요하다.
- `query_latency_s`는 실제 사용자가 기다린 query 시간을 더 잘 나타내기 위해 retry와 exception attempt를 모두 누적한다.
- `tt/` prefix 없이 eval을 실행하면 TT 서버가 아니라 Ollama(`localhost:11434`)로 연결된다.
- `optimal+...`은 retrieval을 하지 않고 question JSON의 oracle context를 사용한다. 그래도 LLM이 typed field를 못 채우면 retry가 발생할 수 있다.
