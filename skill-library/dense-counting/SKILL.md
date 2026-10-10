---
name: dense-counting
description: Count crowded objects under category and attribute constraints using region ownership, instance tracking and a separate ambiguity review.
---

# Dense Object Counting

## Use
Accept an image and a target definition, such as visible screws of one type, occupied seats, or red markers. Establish whether partially visible instances count. A visible count cannot establish the hidden inventory behind occlusions.

## Define an instance
Identify the object's distinguishing parts and the distractors. One package, one reflection and one component are not automatically three objects. If a target has several visible fragments, match them before assigning identities.

## Partition without duplication
1. Keep an unmarked overview as the coordinate reference. Divide it along shelves, rows or other stable boundaries; use a regular spatial partition only when natural boundaries are unavailable.
2. Assign every object to the region containing its center. If the center is hidden, choose one consistent ownership rule from the visible extent and record the ambiguity.
3. Scan each region in a fixed order. Maintain a ledger of region, counted instance IDs, subtotal and unresolved candidates. Do not infer a subtotal from density or repeated spacing.

## Inspect and mark
Request source crops for touching, occluded or tiny candidates. Overlap neighboring crops for boundary inspection, but count original-image identities, not crop appearances. Use small numbered notes beside already counted instances when helpful; avoid covering color or shape features used for inclusion. A note is bookkeeping, not a new target.

## Reconcile
Review only the uncertain candidates and partition boundaries on a second pass. Check objects split across cells and repeated appearances in overlapping crops. Sum the accepted regional subtotals independently. If a region changes, update its subtotal and total together rather than appending another count.

## Deliver
Return the total under the stated inclusion rule. If detail is requested, give regional subtotals and the unresolved candidate count. Separate an exact visible count from any estimate of occluded objects. If exactness is impossible, report the supported lower bound or a justified interval with the specific source of uncertainty.

## Execution
In ViewCraft, use the observation actions and source IDs supplied by the runtime. Finish with its `action=answer` object, with the complete requested output in the string `answer` field. In another visual agent, use its available inspection tools and return the same user-facing result directly.
