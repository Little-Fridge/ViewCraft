---
name: motion-state
description: Track labeled colored objects across ordered video frames and identify which stayed in the same position throughout the observed sequence.
---

# Motion state

This ViewCraft skill compares object positions across a short sequence of frames. Use it when the question asks which labeled objects remained stationary, moved, or returned to an earlier position. Judge the observed frames; discrete samples do not establish what happened between exposures.

Establish chronological order from the supplied timestamps or frame numbers before comparing positions. The configured contact sheet presents frames left to right, then top to bottom, with timestamp captions when metadata is available. Each tile retains its own source frame. Read embedded frame headers if supplementary views or crops make the surrounding sheet captions ambiguous. Do not interpret blank space at the end of a sheet as a missing frame.

Identify every candidate using its printed label and corroborating color and shape. Maintain the same identity through the sequence even when objects exchange positions. Color alone is insufficient when two objects look similar. Exclude legends, decorative marks, shadows, and background fixtures from the candidate set unless the question explicitly includes them.

Choose a stable position reference for each object, such as its body center or a clearly defined corner. Compare that reference against fixed scene landmarks: grid intersections, table edges, or background features. Keep the reference consistent across frames. Track translation separately from rotation or appearance changes, following the question's definition of stationary. If the camera moves, compare positions relative to stable scene landmarks; without an adequate reference, qualify the conclusion instead of treating screen displacement as object motion.

Check every available frame for each candidate. A stationary candidate must occupy the same scene position throughout the observed sequence within the precision of the images. One clear displacement is sufficient to reject it. Matching first and last positions does not establish stationarity: inspect intermediate frames for an excursion and return. A pause during part of the sequence also does not establish that an object stayed still throughout.

Use the full-frame comparison first. If labels and positions are readable, finish directly without additional observations. When uncertainty concerns a small label or displacement, inspect the relevant original frames or corresponding regions, preserving enough background to compare locations. A crop's position within a composite is not a source coordinate. Avoid overlays that merge identities or obscure the evidence being compared. If occlusion prevents a reliable decision, describe that limitation rather than inferring an unseen position.

Return the requested labels in the specified order and format. Unless the question requests an explanation, keep the final answer to the label list. Submit the result with `action=answer` and put the label list in the `answer` field. Do not spend the remaining observation budget after the available frames resolve the question.
