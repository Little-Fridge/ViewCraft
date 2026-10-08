---
name: image-comparison
description: Compare two views of a desk or storage scene to identify one moved or replaced object, using stable landmarks and corresponding regions.
---

# Image Comparison

This ViewCraft skill compares a before image and an after image of the same physical arrangement. Identify the single moved or replaced item requested by the user. Preserve the supplied temporal order; do not infer order from image brightness, neatness, or which arrangement seems more plausible. Read any explicit before and after labels first. If the question defines the image order, use that definition.

Begin with both whole scenes visible together. Use permanent features such as desk edges, drawer dividers, tray corners, or a monitor base to establish correspondence. Compare positions relative to these landmarks, especially when the camera has shifted slightly. A uniform displacement of every object indicates a registration change rather than many independent movements. Keep original colors and aspect ratios because color, silhouette, and size help distinguish similar objects.

Inspect corresponding zones in a consistent order: upper left, upper right, lower left, then lower right. Match objects by several visible properties, including outline, orientation, material, markings, and neighboring items. Maintain a small internal inventory of matched objects and possible differences. A changed shadow, reflection, exposure, or partially hidden edge alone is insufficient evidence that an item moved or was replaced.

For a suspected move, establish both halves of the change: the item no longer occupies its original location, and the same recognizable item appears at a new location. Check that an unchanged duplicate has not been mistaken for the moved object. For a suspected replacement, compare the occupied location across images and identify the visible identity change. Describe the replacement direction only when asked or needed to disambiguate the answer.

If an identifying detail is unreadable, request corresponding crops from both source images using their original source identifiers and full-image coordinates. Retain enough surrounding structure to recognize the location. The displayed comparison sheet is a visual aid; its composite coordinates are not coordinates in either source. Prefer a matched pair of crops to repeated isolated zooms. Stop inspecting once the original views or crops establish the change clearly.

Before answering, check the remaining scene for a second independent object change. When the evidence supports exactly one item, return action=answer with that item's concise common name in the answer field. Include a visible color or other distinguishing attribute when useful. Follow the user's requested wording and omit position details if only an item name was requested. If multiple changes or ambiguous evidence remain, state that limitation briefly rather than inventing a unique answer. Never consume the observation budget merely to confirm an already readable result.
