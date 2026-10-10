---
name: spot-differences
description: Compare paired images to localize additions, removals, movements and text changes while separating camera, lighting and rendering differences.
---

# Visual Difference Inspection

## Use
Accept two labeled images and the comparison scope. Preserve the supplied A/B or before/after order; two unlabeled pictures do not establish chronology.

## Align the views
Find at least two stable landmarks where possible. Check global translation, scale, rotation and perspective before treating displacement as an object change. For screenshots, align the same panel and scroll position. For photographs, exclude regions newly revealed by a viewpoint change from confident removal claims.

## Compare corresponding regions
1. Sweep matched regions in a consistent order, retaining a record of checked areas.
2. For each candidate, inspect the corresponding native-source crop in both images at comparable scale, together with enough surrounding context to establish location.
3. Check presence, shape, orientation, attributes and visible text according to the user's request. Use pixel-level differences only to find candidates: anti-aliasing, compression and illumination can change pixels without changing content.
4. When an object disappears at its old position, search the other full image before classifying it as removed. Match identity by distinctive features; repeated identical objects may prevent a unique match.

## Consolidate findings
Group several changed pixels belonging to one edit into a single finding. Do not count both departure and arrival of one relocated object as two independent changes. Distinguish a changed label from a change in the depicted physical object. An expected number of differences is a check to investigate, not a quota to fill.

## Deliver
Provide an ordered list or table with change type, concise description, location in A, location in B and any uncertainty. If coordinates are requested, use normalized original-image boxes `[x0,y0,x1,y1]`, top-left origin, separately labeled A and B; use null for a genuinely absent counterpart. Never report crop-local coordinates as original-image coordinates.

## Execution
In ViewCraft, use the observation actions and source IDs supplied by the runtime. Finish with its `action=answer` object, with the complete requested output in the string `answer` field. In another visual agent, use its available inspection tools and return the same user-facing result directly.
