"""Load a captioned recording into Milvus and query it with RetrieveAgent.

The PKLs are the source recording, while the caption JSON contains the text and
1024-dimensional embeddings produced for those frames.  Reusing those stored
embeddings makes loading deterministic and avoids running the captioner again.

Examples (from the repository root)::

    .venv/bin/python remembr/scripts/interactive_retrieve.py --load-only
    .venv/bin/python remembr/scripts/interactive_retrieve.py
    .venv/bin/python remembr/scripts/interactive_retrieve.py \
        --question "Where is the white plastic box?"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from remembr.agents.retrieve_agent import RetrieveAgent  # noqa: E402
from remembr.memory.memory import MemoryItem  # noqa: E402
from remembr.memory.milvus_memory import MilvusMemory  # noqa: E402
from remembr.utils import trace  # noqa: E402


DEFAULT_DATA_DIR = REPO_ROOT / "remembr/data/try0_plane_preprocessed_5/0"
DEFAULT_CAPTIONS = (
    REPO_ROOT
    / "remembr/data/captions/try0_plane_preprocessed_5"
    / "captions_tt-try0-qwen3-vl-8b-terse-512-1f_3_secs.json"
)
DEFAULT_COLLECTION = "try0_plane_preprocessed_5_retrieve"
REQUIRED_FIELDS = {
    "position",
    "theta",
    "time",
    "caption",
    "file_start",
    "file_end",
    "text_embedding",
}


def default_server_url() -> str:
    explicit = os.environ.get("TT_SERVER_URL")
    if explicit:
        return explicit
    openai_base = os.environ.get("TT_LLM_BASE_URL")
    if openai_base and openai_base.rstrip("/").endswith("/v1"):
        return openai_base.rstrip("/")[:-3]
    return "http://localhost:8100"


def load_caption_entries(captions_path: Path, data_dir: Path) -> list[dict[str, Any]]:
    """Read captions and prove that they belong to the requested PKL directory."""
    if not data_dir.is_dir():
        raise ValueError(f"PKL directory does not exist: {data_dir}")
    if not captions_path.is_file():
        raise ValueError(f"Caption JSON does not exist: {captions_path}")

    with captions_path.open(encoding="utf-8") as caption_file:
        entries = json.load(caption_file)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Caption JSON must contain a non-empty list: {captions_path}")

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Caption entry {index} is not an object")
        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            raise ValueError(f"Caption entry {index} is missing: {sorted(missing)}")
        for field in ("file_start", "file_end"):
            referenced = data_dir / Path(entry[field]).name
            if not referenced.is_file():
                raise ValueError(
                    f"Caption entry {index} references a missing PKL: {referenced}"
                )
        if len(entry["position"]) != 3:
            raise ValueError(f"Caption entry {index} position is not 3D")
        if len(entry["text_embedding"]) != 1024:
            raise ValueError(f"Caption entry {index} embedding is not 1024D")
    return entries


def build_memory(args: argparse.Namespace, entries: list[dict[str, Any]]) -> MilvusMemory:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.collection):
        raise ValueError(
            "Milvus collection must start with a letter/underscore and contain "
            "only letters, numbers, and underscores"
        )

    memory = MilvusMemory(
        args.collection,
        db_ip=args.db_ip,
        db_port=args.db_port,
        time_offset=0.0,
    )
    collection = memory.milv_wrapper.collection
    existing = collection.num_entities

    if args.load_mode == "rebuild":
        memory.reset(drop_collection=True)
        collection = memory.milv_wrapper.collection
        existing = 0
    elif args.load_mode == "reuse":
        if existing == 0:
            raise RuntimeError(
                f"Collection {args.collection!r} is empty; use --load-mode auto or rebuild"
            )
        print(f"[Milvus] 기존 컬렉션 재사용: {args.collection} ({existing}개)")
        return memory
    elif existing:
        if existing != len(entries):
            raise RuntimeError(
                f"Collection {args.collection!r} has {existing} entities, but the caption "
                f"file has {len(entries)}. Use --load-mode rebuild after checking the target."
            )
        print(f"[Milvus] 적재 완료 컬렉션 재사용: {args.collection} ({existing}개)")
        return memory

    for entry in entries:
        item = MemoryItem.from_dict(entry)
        memory.insert(item, text_embedding=entry["text_embedding"])
    collection.flush()
    loaded = collection.num_entities
    if loaded != len(entries):
        raise RuntimeError(f"Milvus insert count mismatch: expected {len(entries)}, got {loaded}")
    print(f"[Milvus] 적재 완료: {args.collection} ({loaded}개)")
    return memory


def server_health(server_url: str, timeout: float = 3.0) -> dict[str, Any]:
    health_url = server_url.rstrip("/") + "/health"
    try:
        with urlopen(health_url, timeout=timeout) as response:  # noqa: S310 - local URL
            payload = json.load(response)
    except (OSError, URLError, ValueError) as error:
        raise RuntimeError(
            f"NPU server is unavailable at {health_url}. Start an LLM-capable mode "
            f"yourself in the designated tmux session, then retry. ({error})"
        ) from error
    if not payload.get("ready") or not payload.get("llm"):
        raise RuntimeError(
            "NPU server is not ready with an LLM: "
            f"mode={payload.get('mode')!r}, state={payload.get('state')!r}, "
            f"llm={payload.get('llm')!r}"
        )
    return payload


def show_answer(output: Any) -> None:
    values = asdict(output)
    filled = {key: value for key, value in values.items() if value is not None}
    print("답변:", json.dumps(filled, ensure_ascii=False))


def ask(agent: RetrieveAgent, memory: MilvusMemory, question: str) -> None:
    # A REPL reuses one memory object. RetrieveAgent's eval path normally gets a
    # fresh object per question, so explicitly preserve that isolation here.
    memory.working_memory.clear()
    output = agent.query(question)
    show_answer(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="try0_plane PKLs/captions를 Milvus에 적재하고 RetrieveAgent로 질의"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--db-ip", default="127.0.0.1")
    parser.add_argument("--db-port", type=int, default=19530)
    parser.add_argument(
        "--load-mode",
        choices=("auto", "rebuild", "reuse"),
        default="auto",
        help="auto: empty collection만 적재, rebuild: 지정 컬렉션 재생성, reuse: 적재 생략",
    )
    parser.add_argument("--load-only", action="store_true", help="적재/검증 후 질의 없이 종료")
    parser.add_argument("--question", help="대화식 입력 대신 질문 하나만 보내고 종료")
    parser.add_argument(
        "--server-url",
        default=default_server_url(),
    )
    parser.add_argument(
        "--model",
        help="OpenAI 요청의 model 값. 생략하면 /health의 현재 llm_model 사용",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-ctx", type=int, default=65536)
    parser.add_argument("--quiet", action="store_true", help="검색어와 검색 hit trace 숨김")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_caption_entries(args.captions.resolve(), args.data_dir.resolve())
    print(
        f"[data] PKL={args.data_dir.resolve()} captions={args.captions.resolve()} "
        f"({len(entries)}개)"
    )
    memory = build_memory(args, entries)
    if args.load_only:
        return

    health = server_health(args.server_url)
    active_model = args.model or health["llm_model"]
    print(f"[NPU] mode={health['mode']} model={active_model}")
    # RetrieveAgent reads this environment variable when it builds ChatOpenAI.
    # Keep the completion endpoint on the same server whose health we checked.
    os.environ["TT_LLM_BASE_URL"] = args.server_url.rstrip("/") + "/v1"
    if active_model.startswith("tt/"):
        active_model = active_model[len("tt/") :]
    agent = RetrieveAgent(
        llm_type=f"tt/{active_model}",
        num_ctx=args.num_ctx,
        temperature=args.temperature,
    )
    agent.set_memory(memory)
    trace.ENABLED = not args.quiet

    if args.question:
        ask(agent, memory, args.question)
        return

    print("질문을 입력하세요. 종료: /quit 또는 Ctrl-D")
    while True:
        try:
            question = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"/quit", "/exit", "quit", "exit"}:
            break
        if not question:
            continue
        try:
            ask(agent, memory, question)
        except Exception as error:  # keep the REPL alive after one failed request
            print(f"질의 실패: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
