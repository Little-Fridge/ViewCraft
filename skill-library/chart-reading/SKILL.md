---
name: chart-reading
description: Read statistical charts and dashboards, extract supported values and calculate comparisons while preserving scale, units, series identity and uncertainty.
---

# Chart Analysis & Quantitative Reasoning

1. **Select the panel and decode its scale.** Read metric, unit, category/time axis and legend. Match the requested series to its line or bar style. Check for two y-axes, a nonzero baseline, reversed direction or logarithmic ticks before estimating values.
2. **Collect just the required operands.** Crop the relevant marks with neighboring labeled ticks; inspect the legend separately if needed. Record `panel | series | category | value | unit | printed/estimated` in the action's `note`. Prefer printed labels. For a stacked bar, read both boundaries and subtract; the upper boundary is a cumulative total.
3. **Interpolate on the correct axis.** On a linear axis between ticks `(p0,v0)` and `(p1,v1)`, estimate `v=v0+(p-p0)/(p1-p0)*(v1-v0)`. On a log axis, interpolate `log(v)` instead. Here `p` is position along the axis; reversed pixel direction is handled by the same ratio. Use only precision supported by tick spacing and image detail.
4. **Compute after reading.** Absolute change is `new-old`; relative change is `(new-old)/old`. A move from 20% to 25% is 5 percentage points or 25% relative growth. A zero baseline makes relative growth undefined. Check that both operands refer to the same population, period and unit; for rates over irregular intervals, divide by elapsed time.
5. **Cross-check the conclusion.** Reinspect the series label and the two marks determining the answer. For rankings, include ties or overlapping estimate ranges. Read the legend's definition of error bars; overlap alone does not establish statistical significance. Do not infer a missing value by extending a trend.

Return the requested comparison, operands, units and a short calculation. Label visual estimates as approximate. If resolution cannot distinguish two candidates, report the tie or interval that the chart supports.

In ViewCraft, save the working record in the next action’s `note` and finish with `{"action":"answer","answer":"..."}`. Follow the runtime’s image-number and coordinate manifest. In another visual agent, use its inspection tools and return the same result directly.
