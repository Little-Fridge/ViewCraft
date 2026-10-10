---
name: pdf-reading
description: Answer questions about PDF reports, papers and manuals by locating page-level evidence, joining tables and figures across pages, and returning traceable citations.
---

# PDF Research & Evidence Extraction

1. **Make an answer checklist.** Split the question into fields: entity, metric, period, unit, comparison and requested explanation. For “compare cost and accuracy”, cost and accuracy need separate evidence. Do not stop after finding one.
2. **Find the right pages.** Use `view` to open the contents or first page if absent. Match headings to the checklist; translate printed page numbers through the supplied file-page map. Open likely pages and their continuations. Record irrelevant pages as checked so they are not searched twice. Uniform previews do not cover the entire document.
3. **Read one evidence block at a time.** Use `zoom` on a paragraph, figure or table, including its heading, unit and applicable footnote. Read columns vertically. For a continued table, read the original header and continuation rows together; match columns by header, not horizontal position alone.
4. **Save facts before changing pages.** In each action's `note`, record `field | value/quote | unit | filename:file-page | qualifier`. Example record shape: `accuracy | 84.2 | % | report.pdf:12 | test split, model B`. Mark missing fields `unresolved`. This is a working record, not a suggested answer.
5. **Join and check.** Before subtracting or comparing, match the population, period and measurement definition. Reopen the cited block if a number conflicts with prose. Read all components of a requested field, including a second line or nearby qualifier. Never reconstruct a blurred digit from what a total ought to be.

Return the answer with a source for each requested field; distinguish printed values from your calculations. Identify remaining gaps by field. Use the filename and 1-based file page, adding the printed page label when visible.

The runner renders PDFs with `--pdf`; `view` selects existing page images and `zoom` crops them. Neither rerenders a PDF at higher resolution. If native pixels are unreadable, identify the affected page and request a higher-resolution page image.

In ViewCraft, save the working record in the next action’s `note` and finish with `{"action":"answer","answer":"..."}`. Follow the runtime’s image-number and coordinate manifest. In another visual agent, use its inspection tools and return the same result directly.
