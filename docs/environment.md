# Environment

Use Python 3.11+ on Linux/macOS. Install `requirements.txt`; tests additionally
need `requirements-dev.txt`. Hosted inference requires no local GPU. Both model
roles must support image inputs over a chat-completions endpoint.

The shared rate limiter uses `fcntl`. On another platform use
`VIEWCRAFT_LIMITER=process` (no cross-process rate coordination).

Start with the README and `configs/models.yaml`. For backend authentication,
separate role endpoints, output roots and cache isolation see
[configuration](configuration.md).

`requirements-docs.txt` adds PDF-page tooling for benchmark adapters;
`requirements-analysis.txt` adds optional plotting tools.
`requirements-lock.txt` is a historical environment snapshot from the original
package. It is not required for the new examples and does not guarantee identical
API model outputs or reproduce archived paper tables.

## Publication verification · 2026-10-08

The prepared public repository passed **146 offline tests** on macOS with
Python 3.12.14. Both README `--check` commands also passed without model calls.

Verified dependency versions:

- Pillow 12.3.0
- numpy 2.5.3
- pyarrow 25.0.1
- PyYAML 6.0.3
- requests 2.34.2
- pytest 9.1.1

These checks cover the software path and bundled inputs, not real-model accuracy.
Other supported Python/dependency combinations have not been individually tested.
This release does not distribute internal model-response recordings.
