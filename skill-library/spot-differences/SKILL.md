---
name: spot-differences
description: Compare two corresponding images and locate meaningful visual differences.
  Use for before/after photos, design revisions and visual checking.
---

# 找不同

Establish image A/B or before/after from supplied labels. Match stable landmarks before comparing positions; camera movement or global resizing can shift every object. Compare corresponding regions and keep each crop linked to its source image. For an object missing at its old location, search the full other view before calling it removed.

Check presence, position, shape, color and visible text according to the requested scope. Treat illumination and compression changes separately from object edits. Describe each difference once, even if it affects several neighboring pixels or appears in overlapping crops. Do not infer a chronological direction when none is supplied.

Return a concise difference list with paired locations or bounding boxes in normalized coordinates of the ORIGINAL images, explicitly labeled A and B. Explain uncertain matches; do not manufacture extra differences to reach an expected count.

Finish with the runtime action=answer object. Its answer field is a string containing the requested output.
