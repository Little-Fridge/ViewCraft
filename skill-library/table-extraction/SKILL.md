---
name: table-extraction
description: Transcribe visual tables into a requested structured format while preserving
  row and column ownership. Use for scanned tables, screenshots and spreadsheet-like
  document regions.
---

# 表格提取

Identify the header hierarchy, merged cells, units and table boundaries before extracting rows. Distinguish a blank cell, zero, dash and unreadable value. Propagate a merged label only where its visible span or the table's explicit convention justifies it. Repeated page headers are not data rows.

For dense tables, inspect overlapping row bands that retain the relevant header context. Track a stable row identifier and avoid duplicating rows at crop or page boundaries. Keep identifiers with leading zeros as strings and preserve signs, decimal marks and units. Do not reconstruct a blurry digit from a desired total.

Validate the resulting column count and report visible total inconsistencies without changing the source values. Return the requested CSV, JSON or Markdown table. Quote CSV cells containing separators or newlines and state which cells remain unreadable when the output format permits.

Finish with the runtime action=answer object. Its answer field is a string containing the requested output.
