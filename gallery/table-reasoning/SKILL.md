---
name: table-reasoning
description: Read an image of a table, filter rows using the requested conditions, and calculate a simple total from the correct numeric column.
---

# Table reasoning

This ViewCraft skill supports questions about selecting records and aggregating values in a visible table. Preserve the association between a row's identifying fields, filter fields, and numeric values throughout the calculation. Use the question to decide which columns matter; nearby prices, counts, and printed totals may describe different quantities.

Read the table title, header labels, units, and any footnotes that affect interpretation. Locate the requested measure by its header rather than by its position or appearance. Distinguish a quantity column from a unit price or extended amount. If the question requests a sum of quantities, add the quantities directly; multiplication is appropriate only when the question explicitly requires a derived amount.

Translate the question into explicit conditions before selecting records. Treat conditions joined by “and” as an intersection: every included row must satisfy all of them. Preserve inclusive and exclusive boundaries for numeric comparisons. Match categorical labels by their visible meaning, including qualifiers such as pending, returned, or cancelled. Do not infer that similar colors or adjacent rows belong to the requested category.

Scan each data row in order. Use its identifier and horizontal rules to carry the selection across columns without drifting into a neighboring record. For every qualifying row, retain its identifier and the value from the target column in a compact internal ledger. Repeated category labels describe separate records unless the table explicitly identifies duplicates. Skip headers, footers, subtotals, and grand totals when the task asks for individual records; otherwise these can double-count the same values.

Keep the overview available to preserve column and row context. If all required cells are readable, complete the calculation directly. If a cell is ambiguous, request a crop spanning that row's identifier, relevant filter fields, and measure; retain nearby headings when practical. Prefer a contiguous horizontal band over isolated digits. Inspect only the uncertain evidence, and avoid another observation when it cannot provide additional detail. Do not guess a missing digit or silently treat an unreadable value as zero.

Add the selected values, then check the selection against the original conditions once more. Verify that the number of addends matches the number of qualifying rows and that no excluded status or neighboring numeric column entered the calculation. Preserve any requested unit, rounding rule, or answer format. When the result is supported by the visible evidence, finish with action=answer and place the requested short result in the final field. Do not require a tool call or a longer explanation merely to finish a readable table.
