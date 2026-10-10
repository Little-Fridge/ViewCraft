---
name: document-comparison
description: Compare document versions by matching sections and records, then report substantive text, numeric and table changes with paired source references.
---

# Document Revision Review

## Use
Accept two versions as PDFs or labeled page images. Establish old/new order and the comparison scope. The CLI accepts `--pdf old.pdf new.pdf` and labels sources by document number and file page.

## Align content before comparing
Map sections through clause identifiers, headings, product codes or other stable keys. Page numbers, line wraps and row positions can change during reflow. Split or merged sections may require one-to-many alignment; record unmatched sections separately until the surrounding pages are checked.

## Build paired evidence
Keep one overview per version. Inspect matched blocks at native detail with relevant captions, table headers and footnotes. When only selected pages are available, distinguish 'not present in supplied pages' from a verified deletion from the complete version.

## Inspect substantive changes
Check additions, deletions and substitutions in requested clauses, requirements, dates, quantities and entities. Preserve negation and qualifiers such as 'at least', 'unless', 'per unit' and 'excluding'. A one-word change may matter more than a paragraph moving to another page.

For changed numbers, compare unit, currency, denominator, tax basis and covered period together. For tables, align rows by stable item identity and inspect headers independently: a renamed header can change every cell's interpretation without changing the printed numbers. Classify formatting-only and movement-only changes separately when the user wants them.

## Reconcile the report
Each finding needs evidence in both versions or a clearly established absence. Merge duplicate findings caused by overlapping crops. Check whether a changed summary agrees with its detailed table; retain both if the new version is inconsistent.

## Deliver
Return one row per finding: field/clause, old content, new content, factual change and source locations in both versions. Include unresolved matches separately. Quote only enough text to identify the difference. Do not infer commercial, legal or technical consequences that the documents do not establish.

## Execution
In ViewCraft, use the observation actions and source IDs supplied by the runtime. Finish with its `action=answer` object, with the complete requested output in the string `answer` field. In another visual agent, use its available inspection tools and return the same user-facing result directly.
