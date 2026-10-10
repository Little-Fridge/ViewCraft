---
name: table-extraction
description: Convert scanned tables, PDF pages and screenshots to CSV, JSON or Markdown while preserving header hierarchy, merged labels and row ownership.
---

# Visual Table Extraction

## Use
Accept table images or PDF pages and the desired output format and columns. When raw PDF input is available, use the repository CLI's `--pdf` and optional `--pages`. Preserve source order across pages.

## Recover the schema
Identify table boundaries, header levels, row keys, units and footnotes before extracting values. Flatten a multi-level header into an unambiguous name such as `2025 / Revenue (USD)` if a flat format is requested. Distinguish data columns from row labels and subtotal columns.

## Read in bands
1. Inspect overlapping row bands at native source detail while retaining the relevant headers.
2. Keep a ledger keyed by a stable row label plus its source location. Row position alone is insufficient when repeated sections restart numbering.
3. Propagate a merged label only across its visible span or an explicit document convention. Do not fill every blank with the previous value.
4. For continued tables, match column structure across pages, remove repeated headers from the data and reconcile rows split at page boundaries.

## Preserve values
Keep identifiers with leading zeros as strings. Distinguish zero, blank, dash, not-applicable and illegible. Preserve signs, decimals, currency and displayed precision. If a glyph is uncertain, re-read its original crop; totals may flag an issue but cannot supply the missing digit. Do not silently normalize ambiguous date or decimal conventions.

## Validate structure
Check row widths, header ownership, duplicate boundary rows and whether merged headings shifted a value into the wrong column. Recompute visible totals only as an audit; report discrepancies without correcting the source. Apply requested filters after extraction so they do not change interpretation of neighboring labels.

## Deliver
Return valid CSV, JSON or Markdown as requested. Escape embedded quotes and separators in CSV. For strict machine-readable output, keep commentary outside the payload only when allowed; use null for illegible JSON values and a consistent explicit token for CSV if needed. Include page/row provenance when requested, with an uncertainty list keyed to affected cells.

## Execution
In ViewCraft, use the observation actions and source IDs supplied by the runtime. Finish with its `action=answer` object, with the complete requested output in the string `answer` field. In another visual agent, use its available inspection tools and return the same user-facing result directly.
