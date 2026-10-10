<p align="center"><img src="../website/assets/brand/viewcraft-mark.png" alt="ViewCraft" width="108"></p>

# ViewCraft Skill Library

Ten ready-to-run multimodal task skills. Each combines an evidence-gathering workflow, task-specific view settings and an output contract. Use them on your own files without running an optimization stage first.

| Skill | What you get | Input |
| --- | --- | --- |
| [PDF Research & Evidence Extraction](pdf-reading/SKILL.md) | Cross-page answers with citations, units and evidence gaps | PDF or page images |
| [Dense Object Counting](dense-counting/SKILL.md) | Instance counts with regional reconciliation | Image |
| [Visual Difference Inspection](spot-differences/SKILL.md) | Aligned additions, removals and movements with paired locations | Two images |
| [Visual Maze Navigation](maze-solving/SKILL.md) | A route checked against corridor connectivity | Maze image |
| [Chart Analysis & Quantitative Reasoning](chart-reading/SKILL.md) | Values and calculations that preserve scale and uncertainty | Chart image or PDF |
| [Visual Table Extraction](table-extraction/SKILL.md) | CSV, JSON or Markdown with correct row and column ownership | PDF or table images |
| [Document Revision Review](document-comparison/SKILL.md) | Substantive changes with paired old/new references | Two PDFs or labeled page images |
| [Screenshot Diagnosis](screenshot-troubleshooting/SKILL.md) | Evidence-based diagnosis and a discriminating next check | Screenshot |
| [Video Event Search & Temporal Grounding](video-event-localization/SKILL.md) | Event order and time intervals supported by observed frames | Video or timestamped frames |
| [Diagram & Workflow Reasoning](diagram-understanding/SKILL.md) | Traced paths, dependencies, branches and loops | Diagram image or PDF |

## Run a skill

Install the [core requirements and configure a vision-model executor](../README.md#quickstart). PDF and video inputs additionally need:

```bash
python -m pip install -r requirements-media.txt
```

From the repository root:

```bash
# Read a PDF directly, with automatic page rendering and source labels.
python scripts/use_skill.py --skill pdf-reading \
  --pdf report.pdf --pages 2-8,21-25 \
  --question "Compare the reported methods and results. Cite a page for each claim."

# Compare document versions in the specified old/new order.
python scripts/use_skill.py --skill document-comparison \
  --pdf old.pdf new.pdf \
  --question "Document 1 is OLD and document 2 is NEW. List changed prices and delivery terms with sources."

# Count instances with an explicit inclusion rule.
python scripts/use_skill.py --skill dense-counting \
  --images inventory.png \
  --question "Count visible screws, including partly occluded screws identifiable as one instance. Give regional subtotals."

# Inspect paired images.
python scripts/use_skill.py --skill spot-differences \
  --images before.png after.png \
  --question "Image 1 is BEFORE and image 2 is AFTER. List removed and relocated objects with paired locations."

# Find a video event in an original-timeline window.
python scripts/use_skill.py --skill video-event-localization \
  --video clip.mp4 --start 12 --end 25 --frame-interval 0.5 \
  --question "When is the blue object placed in the box? Give the narrowest supported time interval."
```

Add `--check` to validate the skill and prepare the inputs without model calls or persistent run output. Add `--output runs/my-task` to choose a new output folder. The answer is printed and saved in `answer.md`; `result.json` reports execution status. Model replies, trajectories and visual replay are saved only when you explicitly add `--record`.

## Input controls

- **PDF:** `--pages` uses 1-based file pages, not printed page labels. Omit it for documents up to 64 pages. For longer documents, select up to 64 pages per document; the runner never silently discards later pages. With multiple PDFs, the same page selection applies to each. Pages are rendered at a 2560-pixel long edge and remain individually addressable during execution.
- **Video:** timestamps stay relative to the original clip start. The default spacing is one second, with a 120-frame limit. An over-limit clip produces an actionable error instead of silently truncating it. Refine an uncertain interval with a smaller `--frame-interval`; events between sampled frames may remain unobserved. Audio is not processed.
- **Prepared frames:** pass `--images frame1.png frame2.png` with `--frame-times 0 0.5`. Timestamps must be finite, nonnegative, strictly increasing and match the image count.
- **Several images:** supply them in the order described by your question. The runner exposes a zero-based source-unit map; answer locations refer to the named original image, not a temporary crop.

The observation budget controls how many inspection steps are available. Long tables and documents may need narrower questions or larger budgets in `skill.yaml`. Failed or incomplete execution is reported rather than treated as a successful answer.

## Multi-step inspection

The bundled runner exposes `view`, `zoom`, `compose`, `note` and `answer` to the model. It also evaluates the image-processing and layout expressions in the skill configuration. It does not expose a shell, a general-purpose Python interpreter or an external graph solver. The workflows here must use those supplied actions; tools available in another host agent are not automatically available in ViewCraft.

The runtime labels every attachment with its source unit and file/page or timestamp. `view` and `compose` use zero-based **source units**; `zoom` and canvas markers use the current one-based **Image numbers**. Image numbers can change between steps, so keep working records keyed by source unit or filename/page.

Each action's top-level `note` carries concise observations to subsequent calls, including the final call. Store new facts and explicit corrections, rather than repeating the entire history. For example, a counting step can save:

```json
{"action":"zoom","image":1,"regions":[{"x0":0.5,"y0":0,"x1":1,"y1":0.5}],"note":"Region A complete: A01-A12 accepted, subtotal 12. Inspect region B next."}
```

Canvas markers are optional. Give each object a stable `id`; reusing that ID on the same source updates its marker. Distinct IDs remain separate even when objects are close. These records track the model's observations; they do not independently verify that an observation is correct.

## Use in another visual agent

Give the agent the relevant `SKILL.md`, or copy its folder into the agent's supported skills directory. The agent needs image inspection and cropping tools; PDF and video tasks also need rendering or frame extraction. The reading workflow can be used directly. `VIEW.yaml` and `LAYOUT.yaml` are executed by ViewCraft and are not automatically interpreted by other agents.

Each folder contains:

| File | Purpose |
| --- | --- |
| `SKILL.md` | Task workflow, verification rules and deliverable |
| `VIEW.yaml` | Source selection, image resolution and crop settings |
| `LAYOUT.yaml` | Source context, retained views and answer contract |
| `skill.yaml` | Task identity and execution budget |

The library is maintained separately from the evolved benchmark artifacts evaluated in the paper. Paper scores do not measure these ten workflows. See [release scope](../docs/release.md) for the artifact boundary.
