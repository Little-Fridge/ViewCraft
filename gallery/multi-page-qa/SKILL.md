---
name: multi-page-qa
description: Answer concise questions about a short report by locating relevant pages and combining visible facts across pages.
---

# Multi-page QA

This ViewCraft skill reads a small collection of report pages and answers a question using evidence distributed across the document. Use it when a requested comparison, calculation, or identification depends on more than one page. Preserve page identity throughout inspection so that a value remains attached to its section, reporting period, and unit.

Start by identifying what the question asks for: the entity, time period, measure, and required operation. Distinguish quantities that sound similar, such as accepted output, gross output, pending work, and planned output. Determine which two facts would resolve the question before reading every detail of the report.

Scan the visible headings, contents list, table captions, and printed page numbers. Build a compact mental page map. Use section names to locate likely evidence; an overview or contents page can identify where a measure is defined without itself supplying that measure. Do not assume adjacent pages concern the same period or that the largest number on a page is the requested value.

Keep the original pages separately addressable. For a short report that fits the image budget, inspect all page overviews in document order. The selection expression preserves this coverage without encoding any example's evidence locations. If a necessary page is absent, request its source using the runtime's supplied identifier. Printed page numbers and runtime source indices may differ; use the visible source mapping rather than assuming they are interchangeable.

Read each relevant fact together with its row label, column heading, unit, and period. Carry that context forward when moving to the second page. Match the entity and scope explicitly: a monthly target cannot be compared with a year-to-date total, and a count of people cannot stand in for a count of completed items. Treat neighboring figures as separate measures unless the report states a relationship.

If essential text is genuinely unreadable, request a focused crop that includes both the value and its defining labels. Preserve the other evidence page in context. Avoid repeated crops of already readable content; enlarging existing pixels does not recover absent detail.

Combine only the requested facts. Check subtraction direction, denominator, units, and whether the question asks for remaining work, a difference, a ratio, or a named item. Do not add unrelated pending quantities unless the question requires them. When the evidence is readable and consistent, finish immediately with action=answer and put the concise result in the final field. Follow the requested answer format, including numeric-only output when requested. If a necessary fact is missing or conflicting, state the specific limitation instead of inventing a value.
