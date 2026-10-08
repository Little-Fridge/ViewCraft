---
name: screen-grounding
description: Locate a requested GUI button or control in a screenshot and return its normalized click point while preserving source geometry.
---

# Screen grounding

This ViewCraft skill identifies a requested interactive control in a GUI screenshot and reports a click point in the original screenshot. Use the visible interface, the request, and the spatial relationship between labels and controls. A screenshot establishes a visible target; reporting a point does not execute a click or establish what a live application would do afterward.

Read the complete request before choosing a target. Separate the control label or icon from contextual qualifiers such as workspace, section, object name, environment, status, and row. Locate the relevant container first, then find its action. When identical buttons appear in several rows, trace the requested row horizontally from its identifying cells to the action column. Check all supplied qualifiers together. Do not select a page-level action just because its wording resembles a row-level action.

Inspect the full screenshot first. Preserve its aspect ratio, orientation, and coordinate frame. Keep every source image individually addressable. Do not tile, stitch, rearrange, rotate, or composite interface fragments. Do not add visual marks over candidate controls. Treat native highlights and disabled styling as interface evidence, not as instructions about which answer to give. If a requested control is absent or genuinely ambiguous, explain that limitation rather than inventing a location.

If the target and its context are readable, answer immediately. Additional observations are optional, not a checklist. When small text prevents discrimination, request a source-addressed crop that includes both the identifying label and the candidate button. Keep the full overview available. Crop only when it can reveal more source detail, and stop once the identity and clickable boundary are clear. Enlarging already visible pixels is not additional evidence.

Choose the center of the button's clickable interior, away from borders, adjacent menu triggers, and overlapping elements. Normalize against the original full screenshot: x equals horizontal pixel position divided by source width, and y equals vertical pixel position divided by source height. The origin is the top-left; x increases rightward and y downward. If using a crop to reason, map its position back through the recorded source region before answering. Never treat crop-local coordinates as source coordinates.

Return the runtime response object with action=answer and an answer containing only the decimal pair x, y. Name the attached full-source overview using its actual image index so the runtime applies the correct coordinate frame. Keep both values within [0, 1] and give enough decimal precision to remain comfortably inside the target. Do not return a bounding box, pixel coordinates, multiple candidates, or a fabricated interaction history.
