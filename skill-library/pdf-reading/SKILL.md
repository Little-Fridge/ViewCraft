---
name: pdf-reading
description: Answer questions about PDF reports, papers and manuals by locating page-level evidence, joining tables and figures across pages, and returning traceable citations.
---

# PDF Research & Evidence Extraction

## Use
Accept a PDF, a selected page range, or page images. For a raw PDF, run the repository CLI with `--pdf report.pdf`; it renders pages and supplies the source map automatically. A host agent with its own PDF tools can render and inspect the same pages directly. For long documents, use the contents page to choose sections, then run with `--pages 2-8,21-25`.

## Locate the evidence
1. Convert the question into the specific entities, periods, metrics and conditions needed. A request for both method and result needs evidence for both.
2. Read the contents, headings and captions to identify likely pages. Initial previews are navigation aids, not evidence that every supplied page was inspected. Request missing source units from the pool before concluding a fact is absent.
3. Keep a compact evidence ledger: requested fact, document, file page, printed page if visible, region, observed value and qualifier. Distinguish the PDF's physical page index from its printed page label.

## Build readable views
Keep the full page beside a native-source crop of the relevant paragraph, figure or table. Include the caption, units and footnotes that change its meaning. If a table continues onto the next page, inspect the header and continuation together. Read a multi-column page in column order; do not splice text horizontally across unrelated columns.

For cross-page questions, collect every required operand before calculating. Check that comparisons use the same population, unit, time window and experimental setting. A figure's approximate visual estimate must not silently replace an exact value printed elsewhere.

## Resolve gaps
If extraction and the rendered page disagree, inspect the visible glyphs. For scans, use OCR only as a search aid when available. Request a larger source render for illegible text; repeated enlargement of an already blurred bitmap adds no evidence. Report conflicting statements with both locations rather than choosing whichever fits the question.

## Deliver
Lead with the answer, then cite each supporting claim as `filename, file p. N (printed p. X when visible)`. For multiple requested fields, return one row per field with value, units and source. Separate source statements from calculations. State which requested facts remain unsupported and which page ranges were actually inspected.

## Execution
In ViewCraft, use the observation actions and source IDs supplied by the runtime. Finish with its `action=answer` object, with the complete requested output in the string `answer` field. In another visual agent, use its available inspection tools and return the same user-facing result directly.
