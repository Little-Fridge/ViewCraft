---
name: video-event-localization
description: Locate actions and state transitions in a video or timestamped frames, track object identity, and report event intervals at the precision actually observed.
---

# Video Event Search & Temporal Grounding

1. **Define observable before/after states.** For “put the blue object in the box”, require a visible outside state and a supported inside/post-release state; touching the box is insufficient. For “last” or “still”, identify the end of the relevant time interval.
2. **Search the supplied timeline.** Inspect widely spaced frames to locate candidate intervals. Keep a record `unit | timestamp | target identity | observed state` in the action's `note`. Match the object using color, shape, markings and nearby landmarks. Compare against fixed scene anchors to distinguish camera movement from object movement.
3. **Narrow the transition.** Request unshown source units between the last observed before-state and first observed after-state. For a single transition, inspect the midpoint and retain the half still bracketing it. If the action may repeat or reverse, inspect intervening frames in time order instead of assuming a monotonic transition. Use `compose` with chronological unit order to compare states; use original single-source frames for crops.
4. **Check completion and identity.** Inspect a later frame for the postcondition. Occlusion does not prove disappearance or transfer. For “objects remaining still near the end”, compare each object's scene-relative position at several separated final timestamps; one final frame cannot establish stillness.
5. **Respect temporal gaps.** `view` can retrieve only frames already in the prepared pool. If adjacent sampled frames bracket a missing event, return that interval and identify the need for denser sampling. A host can rerun the clip window with `--start`, `--end` and a smaller `--frame-interval`; the model cannot extract unsampled frames through `zoom`.

Return the object/event and supporting timestamps. Report onset as `(last observed before, first observed after]` when exact timing is unknown; use “present by t” if no before-state was observed. Do not report frame-sparse stability as proof that no intervening movement occurred, or infer audio from images.

In ViewCraft, save the working record in the next action’s `note` and finish with `{"action":"answer","answer":"..."}`. Follow the runtime’s image-number and coordinate manifest. In another visual agent, use its inspection tools and return the same result directly.
