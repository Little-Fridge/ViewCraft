# Evolve your own multimodal skill

Start with the [installation and model setup](../README.md#quickstart), then follow this guide. Run all commands from the repository root.

**1. Copy the example and edit its initial skill.**

```bash
cp -r examples/counting examples/my_task
```

Replace `initial_skill/SKILL.md` with your instructions. Keep the two YAML
files as a starting point, or specify your own selection, rendering, and layout.
Unspecified YAML fields use library defaults.

**2. Replace `data.jsonl` with your questions and answers.**

```json
{"id":"form-01-total","question":"What is the total?","images":["images/form-01.png"],"answer":"125.00","group_key":"form-01","split":"train"}
```

Use `train`, `gate`, and `test` splits, with at least one source group in each.
Questions about the same form/image/video must share a `group_key` and split.
Image paths are relative to the JSONL file. For multi-page or video tasks, supply
ordered page images or extracted frames in `images`; raw PDFs/videos use the
benchmark adapters or your own preprocessing.

You may omit `split` from **every** row to split groups deterministically using
`seed` and `split_ratios: [0.6, 0.2, 0.2]`. Reference answers go to the
scorer; they are not passed to the executor or proposal generator.

**3. Choose your reward and run.**

Use `scorer: exact_match`, `scorer: edit_similarity`, or supply a local function:

```python
def score(prediction, answer, item):
    # item contains the public question, options, and image paths.
    return float(str(prediction).strip() == str(answer).strip())
```

Set `scorer: score.py:score` in `task.yaml`. The function must return a finite
number in `[0, 1]`; it runs as ordinary local Python.

```bash
python scripts/evolve.py --task examples/my_task/task.yaml --check
python scripts/evolve.py --task examples/my_task/task.yaml --run-id my-task
python scripts/evolve.py --task examples/my_task/task.yaml --run-id my-task --stage test
```

No registry or engine edits are needed for this JSONL path. See
[custom tasks](custom_tasks.md) for multiple choice, frames, split validation,
and the lower-level adapter interface.

## Use the evolved skill

```bash
python scripts/predict.py \
  --skill runs/counting-demo/arms/viewcraft/snapshots/final \
  --images examples/counting/images/09.png \
  --question 'How many red circles are in the image?' \
  --output prediction.json
```

This loads the saved skill, constructs views, and runs the executor. No optimizer
or reference answer is needed. You can also point `--skill` at your initial skill
directory. Pass multiple `--images` for ordered frames/pages and use
`--answer-style letter --options 'choice one' 'choice two'` for multiple choice.
