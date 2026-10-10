# Configuration

The custom-task commands accept `--config configs/models.yaml`. The benchmark
runner reads `configs/experiment.yaml`, overridden by `VIEWCRAFT_CONFIG`.

## Models

Both executor and optimizer must accept text plus image attachments in a
chat-completions request. Evidence extraction/verification reuse the executor
model with fixed prompts. Executor temperature defaults to 0; optimizer to 0.7.
Choose token limits and `max_tokens_field` supported by your service.

```yaml
models:
  executor:
    resolved_id: my-vision-model
    transport: standard
    api_key_env: EXECUTOR_API_KEY
    temperature: 0.0
    answer_max_tokens: 2048
    max_tokens_field: max_tokens
  optimizer:
    resolved_id: my-other-vision-model
    transport: standard
    api_key_env: OPTIMIZER_API_KEY
    temperature: 0.7
    max_tokens_field: max_tokens
```

Set `VIEWCRAFT_EXECUTOR_ENDPOINT` and `VIEWCRAFT_OPTIMIZER_ENDPOINT` for different
services. A role-specific endpoint overrides `VIEWCRAFT_ENDPOINT`, which overrides
the YAML endpoints. URLs must include `/v1/chat/completions` (or your server's
equivalent full route). `VIEWCRAFT_EXECUTOR_MODEL` and `VIEWCRAFT_OPTIMIZER_MODEL`
override YAML model IDs. Export the key variables you named above.

For a local service with no authentication, add `auth: none` to each role.
For the legacy AIHub transport, explicitly choose `transport: passthrough` and
export `AIHUB_APP_ID`/`AIHUB_API_KEY` (or `APP_ID`/`APP_KEY`). For its standard
route use `transport: standard` plus `auth: aihub`. Ordinary bearer-token services
do not need APP_ID/APP_KEY.

The environment template is `.env.example`. The library reads environment
variables; it does not automatically load `.env`. To use a local file:

```bash
cp .env.example .env
# Edit .env, then export its entries:
set -a
source .env
set +a
```

## Paths and concurrency

| Variable | Default |
|---|---|
| `VIEWCRAFT_RUNS_ROOT` | `<repo>/runs` |
| `VIEWCRAFT_CACHE_ROOT` | `<runs>/_cache` for model responses |
| `VIEWCRAFT_DATA_ROOT` | `<repo>/data` for benchmark adapters |
| `VIEWCRAFT_MEDIA_ROOT` | `<data>/_media` for materialized benchmark images |
| `VIEWCRAFT_FFMPEG` / `VIEWCRAFT_FFPROBE` | binaries on PATH |

Custom JSONL paths are resolved relative to task/data files and do not use the
benchmark data root. `--workers` controls concurrent task executions. Throughput
limits are shared across local processes; `VIEWCRAFT_LIMITER=process` selects an
in-process limiter. The default shared limiter requires a Unix-like OS (`fcntl`).

Executor responses are cached by request, model ID, and endpoint. Optimizer
requests are not cached by default. Use an isolated `VIEWCRAFT_CACHE_ROOT` for
independent repeated measurements or after changing a server behind the same URL
and model alias. Credential values are not used as cache keys.

## Common startup problems

- Missing model ID/key: replace placeholders or export role-specific variables.
- HTTP 400: verify image support, token-field name, and decoding options for your
  backend. Configuration does not translate arbitrary provider-native APIs.
- Empty gate/test split: add source groups or supply explicit split labels.
- Dataset or model settings changed on resume: choose a new `--run-id`.
- Observation budget exhausted: increase `budget.observations` in the task YAML
  and use a new run ID. The model may need several note/zoom steps before its
  final answer; incomplete gate execution stops the run.
- No accepted edits: inspect history. A candidate may not improve the task,
  violate execution constraints, or leave rendered views unchanged.

Use `--check` to validate local inputs without making API calls. It does not test
endpoint connectivity or guarantee that a backend accepts your request format.

## GPT-6 endpoint profile

Use `--config configs/models-gpt6.yaml` with your compatible endpoint and key.
The tested GPT-6 deployment rejects temperature=0; this profile omits temperature
and uses max_completion_tokens. A gateway may require a routed model identifier;
set VIEWCRAFT_EXECUTOR_MODEL to its exact served name. Credentials and private
provider routes are not distributed.
