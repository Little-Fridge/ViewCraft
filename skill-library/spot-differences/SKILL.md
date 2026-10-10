---
name: spot-differences
description: Compare paired images to localize additions, removals, movements and text changes while separating camera, lighting and rendering differences.
---

# Visual Difference Inspection

1. **Fix A/B identity.** Use the supplied before/after order. Find two or more unchanged landmarks, such as corners or fixed fixtures. Compare their spacing and orientation to separate camera shift, zoom and rotation from an object's movement. If viewpoints differ, compare only mutually visible areas.
2. **Sweep matching areas.** Split the scene at stable landmarks, or use quadrants. Inspect each A region beside its B counterpart, then `zoom` both when needed. Crop coordinates may differ between images. Include a shared landmark in each crop to verify correspondence.
3. **Record candidates once.** Save `D1 | object | A location | B location | candidate change` in the action's `note`, plus checked regions. Check presence, shape, color, orientation and text. Consolidate several changed parts of one object into one finding unless the user requests individual edits.
4. **Distinguish movement from removal.** For every apparent disappearance, search the entire B image for matching shape, markings and accessories. Pair departure and arrival as one relocation. For an apparent addition, search A. If several identical objects prevent identity matching, report the observable positional difference without inventing identity.
5. **Make a reverse pass.** Sweep B against A to catch additions missed by the first pass. Reinspect candidates at matched scale. Lighting, shadows, compression and newly uncovered background are not automatically object changes. An advertised number of differences is a reason to recheck, never a quota to fill.

Return one finding per change, with type and paired locations. For requested boxes, report normalized coordinates in the original A and B images; map crop-local coordinates through the runtime's crop bounds. Use `null` only for a verified missing counterpart. Leave unresolved identity matches explicit.

In ViewCraft, save the working record in the next action’s `note` and finish with `{"action":"answer","answer":"..."}`. Follow the runtime’s image-number and coordinate manifest. In another visual agent, use its inspection tools and return the same result directly.
