---
name: object-counting
description: Count objects that jointly match a requested color and geometric shape in an image containing distractors.
---

# Object counting

This ViewCraft skill counts distinct visible objects satisfying a conjunction of color and shape. Use it for scenes such as sorting trays, classroom manipulatives, or diagrams containing separate geometric pieces. Treat the question as the definition of the target: matching the color alone or the shape alone is insufficient. Do not assume that the most common object type is the requested one.

Read the whole image first and identify its counting scope. Distinguish physical pieces or diagram objects from decorative marks, panel letters, legends, and examples outside that scope. If the question restricts counting to a tray, panel, or region, respect that boundary. A rotated triangle remains a triangle; rotation does not create a different category. Infer shape from the full outer contour rather than one corner or a shadow. Determine color from the main filled surface, avoiding highlights, outlines, and cast shadows.

Choose a stable spatial traversal before tallying. In a scene with separated compartments, inspect them in reading order. Within each compartment, scan rows from left to right, then move downward. For an irregular arrangement, use nonoverlapping horizontal bands and assign each object by its center. Keep a subtotal for each region and combine subtotals only after all regions have been visited. These regions are accounting aids; do not infer how many objects a tidy layout ought to contain.

Apply both tests to every candidate. Reject pieces of the target color with another contour and pieces of the target shape with another fill color. Check target-colored circles and squares explicitly when counting triangles. Do not count a shadow as a separate object. Use a brief second visual sweep to confirm that each accepted object was counted once and each boundary between regions was covered. Reconcile a discrepancy locally instead of repeatedly recounting the entire scene.

The view configuration preserves RGB color and keeps the overview available. Use the original overview to track positions. If a contour or color boundary is too small to resolve, request one crop containing the uncertain object and enough neighboring context to locate it. A crop is another view of existing objects, so never add its contents as new objects. Avoid grayscale, color remapping, and repeated enlargements that cannot reveal additional source detail. Notes are optional and should not obscure the surfaces being classified.

When the objects are readable and the tally is consistent, finish directly with `action=answer`. Put only the requested nonnegative integer in the `answer` field, without units or an explanation. Do not spend additional observations merely to satisfy a workflow. If an object remains ambiguous, use the remaining budget to inspect that specific ambiguity before submitting the best supported count.
