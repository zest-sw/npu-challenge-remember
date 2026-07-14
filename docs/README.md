# ReMEmbR × NPU — 문서

ReMEmbR를 NPU(BOS 칩)에서 구동하기 위한 포팅 프로젝트 문서 모음.

| 문서 | 언제 보나 |
|---|---|
| **[DEV_STATUS.md](DEV_STATUS.md)** | **여기서 시작.** 진행 상태·마일스톤·알려진 이슈·설계 결정·변경 로그. |
| [TENSTORRENT_INTEGRATION.md](TENSTORRENT_INTEGRATION.md) | 아키텍처와 설계 결정, 디버깅 기록을 이해하고 싶을 때. |
| [PAPER_EVAL_RUNBOOK.md](PAPER_EVAL_RUNBOOK.md) | 논문 NaVQA/CODa 데이터로 평가를 돌릴 때(데이터→캡션→질문→평가) + 보드 복구. |
| [../tt_server/README.md](../tt_server/README.md) | NPU 추론 서버의 구조·요구사항·env·JSON 디코딩. |
| [../tt_server/RUNBOOK.md](../tt_server/RUNBOOK.md) | 서버 기동·venv 셋업·트러블슈팅·안전한 종료 절차. |

> **HW 주의**: 이 장비는 **BOS 칩**이다(Tenstorrent 아님). tt-metal 호환 스택으로 ttnn을 쓰므로
> 로그엔 `BlackholeTTDevice`로 뜨지만, device 노드는 `/dev/bos/0`, 커널 모듈은 `bos`.
