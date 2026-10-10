# Environment

Use Python 3.11+ on Linux/macOS. Install `requirements.txt`; tests additionally
need `requirements-dev.txt`. Hosted inference requires no local GPU. Both model
roles must support image inputs over a chat-completions endpoint.

The shared rate limiter uses `fcntl`. On another platform use
`VIEWCRAFT_LIMITER=process` (no cross-process rate coordination).

Start with the README and `configs/models.yaml`. For backend authentication,
separate role endpoints, output roots and cache isolation see
[configuration](configuration.md).

`requirements-media.txt` adds direct PDF and video input for the skill runner, using pypdfium2 and PyAV without a separate ffmpeg command-line installation.

`requirements-docs.txt` adds PDF-page tooling for benchmark adapters;
`requirements-analysis.txt` adds optional plotting tools.

Run `python -m pytest -q` after installing the development requirements to check your environment. Tests use offline model doubles and do not require API credentials.
