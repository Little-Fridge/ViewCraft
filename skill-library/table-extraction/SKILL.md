---
name: table-extraction
description: Convert scanned tables, PDF pages and screenshots to CSV, JSON or Markdown while preserving header hierarchy, merged labels and row ownership.
---

# Visual Table Extraction

1. **Write the column schema before the rows.** Locate the table boundaries, all header levels, row-label columns, units and footnotes. Flatten nested headers into names such as `2025 / Revenue (USD)`. Record the column order in the action's `note`; use it for every subsequent band.
2. **Read a small row band.** Crop about 5–10 readable rows with the left-hand row keys and all requested columns. For a wide table, crop left and right halves with one shared key column. Join by that key, never by guessed row alignment. Include the last row of the previous band as a boundary check.
3. **Save each completed row.** Record `page + row key -> [cell1, cell2, ...]` in `note` before moving on. Propagate a merged group label only to rows inside its visible span. Keep IDs such as `0017` as strings; preserve signs, decimals and units. Distinguish literal zero, blank, dash and illegible text. Use JSON `null` for unreadable values, not for a printed zero.
4. **Join pages carefully.** Read the next page's header before appending rows. Remove repeated headers from data. If the first row continues the previous page's last row, match the key and complete that row instead of adding a duplicate. A repeated product code in a different section needs a section-qualified key.
5. **Audit before formatting.** Check every row has the schema's column count. Re-read disputed cells with neighboring row labels. Recalculate printed subtotals to flag transcription errors, but never change a visible source value merely to force agreement. If the source itself disagrees, preserve it and report the discrepancy when the output format permits.

Return valid CSV, JSON or Markdown as requested. In CSV, quote cells containing separators, quotes or newlines and double embedded quotes. Apply requested filtering after row extraction. If the budget cannot cover all rows, identify the last completed source row and the unprocessed range rather than returning a partial table as complete.

In ViewCraft, save the working record in the next action’s `note` and finish with `{"action":"answer","answer":"..."}`. Follow the runtime’s image-number and coordinate manifest. In another visual agent, use its inspection tools and return the same result directly.
