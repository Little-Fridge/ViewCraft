---
name: chart-reading
description: Read statistical charts and dashboards, extract supported values and calculate comparisons while preserving scale, units, series identity and uncertainty.
---

# Chart Analysis & Quantitative Reasoning

## Use
Accept a chart, dashboard or paper figure and a question. Identify the relevant panel and metric before reading marks; similarly colored series in separate panels need not have the same meaning.

## Decode the chart
Read title, axes, units, legend and footnotes. Check logarithmic, reversed, truncated and dual axes. Associate each series with its own scale. Identify whether marks show individual values, cumulative totals, stacked components, percentages or uncertainty ranges.

## Extract with context
Keep a chart overview while cropping the requested series together with ticks and legend context. Prefer printed values over estimates from position. For unlabeled bars or points, interpolate only within the actual axis scale and report approximation. A stacked segment's value is the difference between its boundaries, not the height of its upper boundary.

Track a small evidence table: panel, series, category/time, value, unit and whether printed or estimated. Read all operands before calculating. Do not fill a missing point by assuming a smooth trend.

## Compute the requested comparison
Use `(new - old) / old` for relative change and `new - old` for an absolute or percentage-point difference. Identify the denominator and handle a zero baseline explicitly. Align the same period and population before comparing series. For irregular intervals, compare rates per unit time when the question asks about speed of change.

## Check interpretation
Keep confidence intervals, standard deviations and standard errors distinct using the legend. Visual overlap alone does not establish statistical significance. A rising pair of series establishes visual association, not causation. If several panels or scales support different readings, name the one used.

## Deliver
Answer with series/category, values, units, concise calculation and precision supported by the source. Mention only ambiguities that affect the conclusion. Supply a structured evidence table when several comparisons are requested.

## Execution
In ViewCraft, use the observation actions and source IDs supplied by the runtime. Finish with its `action=answer` object, with the complete requested output in the string `answer` field. In another visual agent, use its available inspection tools and return the same user-facing result directly.
