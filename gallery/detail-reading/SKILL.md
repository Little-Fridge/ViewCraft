---
name: detail-reading
description: Read small serial numbers, reference codes, and specification fields in high-resolution product posters using section-aware inspection and purposeful crops.
---

# Detail reading

This ViewCraft skill extracts an exact text field from a dense product specification poster. Use the visual hierarchy to find the right section, then read the requested label and its associated value at a resolution that preserves individual characters. The task is transcription: plausible product knowledge cannot substitute for visible evidence.

Start with the overview. Identify the product, section headings, and overall column structure. Translate the question into two anchors: the named section or subject, and the precise field label. A poster may contain a model number, unit serial, document reference, calibration reference, and accessory identifier simultaneously. Match both anchors before choosing a value. If the requested field and every character are already legible, answer immediately without requesting another observation.

When the overview makes the section clear but leaves its rows too small, request a purposeful zoom from the original source. Use the runtime's source identifier and normalized full-image coordinates. Include the section heading, field label, complete value, and a little surrounding space. A crop spanning one relevant panel generally retains the relationship between labels and values better than an isolated string. Do not crop an entire poster again when a smaller region can reveal native detail. Avoid a fixed quadrant or location: section positions vary between inputs.

Read the label and value together, following row alignment, separators, or leader lines. Check that a nearby serial or revision belongs to its own row. Preserve capitalization, hyphens, slashes, leading zeros, and any meaningful spaces. For ambiguous glyphs such as zero versus capital O, one versus capital I, or five versus capital S, inspect their actual shapes. Another occurrence of the same character elsewhere in the same typeface can help; an expected naming pattern alone is insufficient.

Keep the overview available for context and retain only the most useful recent crops. Request an additional crop only if it resolves a specific remaining uncertainty, such as a clipped suffix or unclear row association. Return to original source coordinates when adjusting a region. Upscaling, sharpening, thresholding, decorative annotations, or a tiled full-page mosaic should not replace access to genuine source pixels. Stop when the field association and transcription are clear, within the observation budget.

Finish using the runtime response object with `action=answer` and place only the requested text in its `answer` field. Do not add the field name, commentary, quotation marks, or neighboring identifiers to that value. If the source remains unreadable at native resolution, state the uncertainty concisely instead of inventing characters or repeatedly requesting equivalent views.
