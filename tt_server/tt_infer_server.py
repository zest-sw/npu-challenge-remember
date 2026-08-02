#!/usr/bin/env python3
"""Compatibility entrypoint for the layered Tenstorrent inference server."""

from pathlib import Path
import sys


# run_server.sh executes this file by absolute path while cwd is the tt-metal
# repository. Add the ReMEmbR repository root so package imports remain stable.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tt_server.main import main  # noqa: E402


if __name__ == "__main__":
    main()
