---
name: dense-counting
description: Count crowded objects under category and attribute constraints using region ownership, instance tracking and a separate ambiguity review.
---

# Dense Object Counting

1. **Define one countable object.** State the target category and attribute, and whether visible partial objects count. Identify its distinguishing parts. A reflection or two exposed fragments of one object must not receive two IDs.
2. **Divide the image into countable regions.** Prefer shelves or rows. Otherwise begin with four quadrants; subdivide a crowded quadrant when individual centers cannot be distinguished. Assign an object to the region containing its center. On a dividing line, assign it to the region on the right or below. Keep this rule through every crop.
3. **Count by location, not appearance density.** Sweep each region left-to-right, top-to-bottom. Give each accepted instance a stable ID such as `A01`. Save `region A: accepted=A01..A12; uncertain=A13; subtotal=12; checked=yes` in the action's `note`. Record approximate original-image centers for ambiguous or boundary objects so they can be matched after zooming.
4. **Resolve only uncertain objects.** Crop a touching group with its immediate neighbors. Match visible parts into instances; check target color/type before inclusion. Overlapping crops may show the same ID again. Never add the number of objects in a new crop directly to the running total. Optional canvas markers use these same IDs; updating `A13` corrects its marker instead of creating a new object.
5. **Audit boundaries and add subtotals.** Check each shared region boundary once for split or repeated instances. Count accepted IDs within each region, then add the regional subtotals. Revise the affected subtotal when a candidate is accepted or rejected. Marker count alone is not the object count: rejected candidates may still have marks.

Return the total in the requested format. Include regional subtotals when requested or when the format allows; keep them in the working record for an integer-only answer. If unresolved objects remain, give the confirmed count plus the number of uncertain candidates; only give an upper bound when each candidate can contribute at most one object. Do not estimate hidden inventory unless requested.

In ViewCraft, save the working record in the next action’s `note` and finish with `{"action":"answer","answer":"..."}`. Follow the runtime’s image-number and coordinate manifest. In another visual agent, use its inspection tools and return the same result directly.
