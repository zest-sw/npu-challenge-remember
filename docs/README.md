# ReMEmbR × NPU — Docs

Documentation for the project porting ReMEmbR to run on the NPU (BOS chip).

| Document | When to read it |
|---|---|
| **[DEV_STATUS.md](DEV_STATUS.md)** | **Start here.** Progress, milestones, known issues, design decisions, changelog. |
| [TENSTORRENT_INTEGRATION.md](TENSTORRENT_INTEGRATION.md) | When you want to understand the architecture, the design decisions, and the debugging history. |
| [PAPER_EVAL_RUNBOOK.md](PAPER_EVAL_RUNBOOK.md) | When running the evaluation on the paper's NaVQA/CODa data (data → captions → questions → eval) + board recovery. |
| [../tt_server/README.md](../tt_server/README.md) | The NPU inference server's structure, requirements, env vars, JSON decoding. |
| [../tt_server/RUNBOOK.md](../tt_server/RUNBOOK.md) | Starting the server, venv setup, troubleshooting, safe shutdown procedure. |

> **HW note**: this machine has a **BOS chip** (not Tenstorrent). It uses ttnn via a tt-metal-compatible stack, so
> the logs show `BlackholeTTDevice`, but the device node is `/dev/bos/0` and the kernel module is `bos`.
