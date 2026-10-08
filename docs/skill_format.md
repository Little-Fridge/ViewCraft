# Skill files

`scripts/evolve.py` takes the initial skill path from the task YAML.
`scripts/predict.py --skill <directory>` loads an initial or final snapshot.
Every skill has `SKILL.md`, `VIEW.yaml`, and `LAYOUT.yaml`.

## SKILL.md

Write your reusable inspection and reasoning procedure as plain Markdown.
Avoid copying answers or rules keyed to particular training examples. The engine
adds the question, attached views, and observation protocol separately.

## VIEW.yaml

These examples illustrate independent fields; use the ones relevant to your task:

```yaml
overview_long_edge_px: 1024
crop_long_edge_px: 1536
select_expr: "uniform(budget)"
pixel_expr: "rgb(red, green, blue)"
canvas_expr: ""
zoom_gate_expr: "zoom_gain > 1.05 or region_frac < 0.12"
```

`select_expr` selects indices from your supplied image/frame pool under a cap.
`pixel_expr` transforms pixels; the identity above makes no change.
`canvas_expr` controls how executor-generated notes are drawn. An empty
`zoom_gate_expr` permits requested zooms; a nonempty expression decides using
facts about the requested region. Upscaling does not recover missing detail.

## LAYOUT.yaml

```yaml
max_images_per_request: 4
history_keep_last_n: 2
layout_expr: "grid(views, cols=2, label=index)"
view_contract: "The attached sheet contains indexed source views."
answer_contract: "Return the requested value in the final answer field."
protocol_contract: ""
```

`layout_expr` composes views. `view_contract` describes their organization;
`answer_contract` describes answer formatting. An empty contract uses the runtime
default. Keep composite layouts and their descriptions consistent. On point
grounding tasks, retain addressable source views rather than combining them into
a sheet with ambiguous coordinates.

Supported fields are declared in `viewcraft/views/config.py`. Expression helpers
are exposed by `selection_help()`, `function_help()`, `layout_help()`, and
`zoom_gate_help()` in their respective `viewcraft/views/` modules. The optimizer
receives these helpers in its proposal prompt. Expressions use a bounded AST
interpreter; they are not arbitrary Python programs.

Final snapshots also contain `CONFIG_HASH`, computed from all three coordinates.
Copy the entire directory when deploying a skill. The paper's logical
reader/interface partition is based on clause meaning; these three files are the
runtime storage format, not a semantic classifier of each instruction clause.
