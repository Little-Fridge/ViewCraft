# ViewCraft Skill Gallery

[Open the gallery](index.html) — one self-contained page with ten tasks, case
images, recorded answers and a concise explanation of each ViewCraft skill.
The page embeds its images and font. No videos, pop-ups or download menus.

| Skill | Task |
|---|---|
| [Find the tiny detail](detail-reading/README.md) | Fine print |
| [Read the right field](form-reading/README.md) | Forms |
| [Let the chart speak](chart-reading/README.md) | Charts |
| [Follow the right rows](table-reasoning/README.md) | Tables |
| [Count without losing track](object-counting/README.md) | Counting |
| [Spot what changed](image-comparison/README.md) | Before & after |
| [Put the story in order](temporal-order/README.md) | Time & events |
| [Notice who stayed still](motion-state/README.md) | Motion |
| [Connect the pages](multi-page-qa/README.md) | Documents |
| [Point to the right control](screen-grounding/README.md) | Interfaces |

Run a skill from the repository root after configuring a model:

```bash
python scripts/use_skill.py --skill chart-reading --record
python scripts/use_skill.py --skill form-reading --images my-form.png --question 'What is the purchase order number?' --record
```

The code directories retain SKILL.md, VIEW.yaml, LAYOUT.yaml, runtime metadata,
case inputs and actual model records. Use a gallery directory as `initial_skill`
with `scripts/evolve.py` to adapt it. Rebuild this page using
`python scripts/build_gallery.py`. Presentation text summarizes the saved skill;
it does not claim that every optional action occurred in the recorded case.

Font: Comic Neue, distributed with its SIL Open Font License in assets/fonts/.
