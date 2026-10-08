---
name: chart-reading
description: Read grouped bar charts using legends, category labels, and axis units to answer a simple comparison or numerical difference.
---

# Chart reading

This ViewCraft skill answers a focused question about a grouped bar chart. Establish the category, series, and units before calculating. Use the visible chart as evidence; do not infer values from a familiar dataset or from the wording of the question.

## Identify the requested bars

Read the question to determine whether it compares series within one category or compares categories within one series. Locate the relevant category labels along the categorical axis. Read the legend and explicitly associate each requested series with its color or pattern. Confirm that the bars are grouped beside one another rather than stacked. Do not assume the first legend entry always corresponds to the leftmost bar, especially when labels or ordering change between charts.

Read the quantitative axis label and its scale. Check for multipliers, percentage units, nonzero baselines, axis breaks, or logarithmic spacing. Treat the baseline as a coordinate reference rather than assuming every bar starts at zero. An absolute difference between percentages is expressed in percentage points unless the question requests a relative change.

## Read values with context

Prefer explicit numerical labels attached to the relevant bars when they are legible. Cross-check their magnitudes against the axis ticks to catch a neighboring-bar or series mismatch. If labels are absent, read the endpoint against the quantitative scale and interpolate only to the precision supported by the image. Avoid reporting an exact integer when the chart only supports an estimate.

Start with the full chart so the legend and both axes remain available. Preserve original colors because they encode series identity. If a needed label is unreadable, request a focused crop that includes the relevant bar pair, category label, and nearby tick marks. Keep the overview available to interpret the legend. Use a separate legend crop only when its text or swatches are unclear. A crop is useful when it exposes additional source detail; repeated inspection of unchanged pixels does not resolve missing information.

## Compare and answer

Associate each read value with its category and series before doing arithmetic. For “how many more,” subtract the reference value from the named subject value and verify the direction against the bar heights. For “which is larger,” return the requested category or series name. Preserve the chart's units unless the question explicitly requests a bare number.

When the chart is readable, finish immediately using action=answer and place the requested result in the answer field. Follow any requested answer format and keep the final response concise. Do not perform extra observations merely to consume the budget. If essential evidence remains unreadable, state that limitation instead of inventing a value.
