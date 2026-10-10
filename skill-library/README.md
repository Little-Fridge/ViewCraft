# ViewCraft Practical Skill Library

Ten reusable task skills in ViewCraft's native snapshot format. These are hand-authored skills, not automatically evolved checkpoints. Bring your own inputs; no evaluation scores, model comparison logs, or benchmark claims are included.

| Skill | Useful for | Input |
| --- | --- | --- |
| [PDF reading](pdf-reading/SKILL.md) | Manuals, reports, papers, cross-page evidence with page references | Rendered page images |
| [Dense counting](dense-counting/SKILL.md) | Inventory, parts, repeated objects or symbols | Image |
| [Spot differences](spot-differences/SKILL.md) | Changed objects, moved items, visual design revisions | Two corresponding images |
| [Maze solving](maze-solving/SKILL.md) | Routes through visible corridors and junctions | Maze image |
| [Chart reading](chart-reading/SKILL.md) | Series comparisons, units, trends and derived calculations | Chart image |
| [Table extraction](table-extraction/SKILL.md) | Screenshots or scans to CSV, JSON or Markdown | Table images |
| [Document comparison](document-comparison/SKILL.md) | Changed quotations, specifications and clauses across versions | Labeled old/new page images |
| [Screenshot troubleshooting](screenshot-troubleshooting/SKILL.md) | Read visible errors and identify a useful diagnostic next step | Application screenshot |
| [Video event localization](video-event-localization/SKILL.md) | Action order, object transitions and event time intervals | Frame images with timestamps |
| [Diagram understanding](diagram-understanding/SKILL.md) | Flowchart branches, loops and system dependencies | Diagram image |

Each folder contains `SKILL.md` (reading procedure), `VIEW.yaml` (observation configuration), `LAYOUT.yaml` (view context and answer contracts), `CONFIG_HASH`, and `skill.yaml` (runner metadata). None contains example-specific answers or fixed evidence coordinates.

## Run with your own input

From the ViewCraft repository root, configure your executor in `configs/models.yaml` using the existing setup documentation. Keep credentials in the configured environment variables.

```bash
python scripts/use_skill.py \
  --skill skill-library/dense-counting \
  --images your-image.png \
  --question "How many target objects are visible? Give region subtotals." \
  --config configs/models.yaml
```

You can also use the short skill ID, such as `--skill dense-counting`. Add `--check` to validate files and skill configuration without calling a model. These folders intentionally do not contain a bundled scored case, so provide both `--images` and `--question`.

For differences, supply both original images and identify their order in the question:

```bash
python scripts/use_skill.py \
  --skill skill-library/spot-differences \
  --images before.png after.png \
  --question "Image 1 is BEFORE, image 2 is AFTER. List removed and relocated objects with locations."
```

## PDF and video inputs

The current `scripts/use_skill.py` entry point consumes images, not raw PDF/video files. Render relevant PDF pages to images before running PDF reading, table extraction or document comparison. Name them by file-page index, such as `report-page-012.png`, and state document version and page mapping in the question. Preserve printed page numbers in the image. For long documents, select relevant sections first; a finite view budget does not guarantee full-document coverage. Cross-page claims require all relevant pages to be supplied.

For video, supply frame images in increasing time order with one timestamp in seconds per frame:

```bash
python scripts/use_skill.py \
  --skill skill-library/video-event-localization \
  --images frame-000.png frame-005.png frame-010.png \
  --frame-times 0 5 10 \
  --question "When does the target enter the container? Report the observed time interval."
```

The skill cannot recover an event absent from the supplied frame pool. Add denser frames around unresolved intervals. Time precision must reflect the frames actually observed.

## Reuse

The library contains task instructions and execution settings. Supply your own source images; no gallery images or recorded model responses are distributed.

The format and runtime-expression checks have been run for all ten skills. Task-level reliability on new PDFs, mazes, charts and videos has not been established. Observation budgets are editable starting configurations, not claims of optimality. See the repository license for the code distribution terms.
