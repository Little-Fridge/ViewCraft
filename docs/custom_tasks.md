# Custom tasks

Start with `examples/counting/task.yaml`. `scripts/evolve.py` accepts ordinary
local image files, including ordered page images and video frames. It uses the
same renderer, proposal generator, paired gate, and snapshot format as the
benchmark entry point.

## JSONL contract

Each nonempty line is one JSON object:

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Unique string, for your own item identity |
| `question` | yes | Question visible to the executor |
| `images` | yes | Nonempty list of image paths relative to this JSONL |
| `answer` | yes | JSON reference value visible only to the scorer |
| `group_key` | yes | Source form, image, document, or video identity |
| `split` | optional | `train`, `gate`, or `test`; supply on all rows or none |
| `options` | optional | Answer choices in their original order |
| `meta` | optional | `frame_times_s` and/or `duration_s` only |

Example multiple choice:

```json
{"id":"scene-1","question":"Which color is the car?","images":["images/car.png"],"options":["red","blue"],"answer":"A","group_key":"car-photo-1","split":"train"}
```

Set `answer_style: letter` for this task. Default exact matching then compares
the extracted answer with `A`. Choose `short_text`, `yes_no`, or `point` as
appropriate for other tasks. Point tasks need a custom region/coordinate scorer.

Example ordered frames:

```json
{"id":"clip-1-q1","question":"Which object leaves the room?","images":["frames/001.png","frames/002.png","frames/003.png"],"meta":{"frame_times_s":[0.0,1.0,2.0],"duration_s":2.0},"answer":"the dog","group_key":"clip-1","split":"train"}
```

The selection expression operates on this supplied frame pool. Preparing a larger
pool gives the optimizer more frames to choose from. Raw videos and PDFs must
first be decoded/rasterized; this entry point does not decode them automatically.

## Split validation

`--check` checks IDs, images, group isolation, nonempty splits, scorer loading,
and the initial skill. Exact duplicate image bytes must use the same group key.
You remain responsible for assigning the same group key to different pages of
one document, frames from one video, or differently encoded copies of an image.

If no row has `split`, groups are shuffled with the task seed and split with
`split_ratios` (default `[0.6, 0.2, 0.2]`). This counts **groups**, not questions.
Small data can leave a split empty; add groups or specify explicit splits.
Training feeds the optimizer, gate selects candidates, and test is evaluated only
when you request `--stage test`.

## Reward function

Built-ins:

- `exact_match`: case-insensitive string equality after stripping outer whitespace.
- `edit_similarity`: lowercased `1 - Levenshtein / max_length`, without a threshold.

For custom rewards, write `score.py` beside `task.yaml` and set
`scorer: score.py:score`. Signature:

```python
def score(prediction, answer, item):
    # prediction: executor's parsed final answer (str)
    # answer: the reference JSON value from your row
    # item: PublicItem; question, media_ids, options, neutral metadata
    return 1.0 if prediction.strip() == str(answer).strip() else 0.0
```

Use a deterministic scorer where possible. Invalid values, NaN, and values
outside `[0, 1]` stop the run. References are stored in the source data and are
not injected into model prompts. The optimizer receives predictions, scores,
sent views, and evidence-support feedback.

## Budgets

`budget.observations` limits executor turns per item;
`budget.max_images_per_request` caps image attachments. `train_batch_size` is
the number of training questions used per round. `gate_question_cap: 0` and
`test_question_cap: 0` mean all questions; nonzero caps use the engine's
group-preserving selection. A group can contain multiple questions.

`max_submissions_per_round: 4` bounds candidate submissions, including eligible
projection retries. `max_gate_evaluations_per_round: 2` allows one ordinary
candidate evaluation and one projection. Preflight retries consume submissions.
`strategy_book: true` enables model-written strategy memory and adds reflection
requests. These limits do not constitute a currency budget; service pricing and
the number of executor turns determine cost.

## Output and resume

`configs/task_manifest.json` records input hashes, item-to-split assignments,
model IDs/settings, initial skill identity, and a map from internal IDs to your
original IDs. Prediction and trajectory files use safe internal IDs.

Completed rounds and ledger items are reused on resume. An interrupted round can
repeat its proposal; within-round proposal attempts are not transactionally
checkpointed. Use a new run ID after changing data, task settings, code, or model
configuration. A missing final snapshot causes test to stop, rather than testing
an initialization accidentally.

## Beyond the JSONL interface

For benchmarks with custom decoding or interactive execution, see
`viewcraft/adapters/registry.py:DomainSpec` and `viewcraft/schema.py:Sample`.
An adapter produces `Sample(public=PublicItem(...), private=PrivateItem(...),
group_key=...)`; a scorer takes `(answer_text, private, public)`.
Use `registry.register(spec)` in your own runner before constructing `Chain`.
The bundled runtime handles static images/frame pools, not arbitrary interactive
environments. Integrating an interactive environment requires an executor adapter.
