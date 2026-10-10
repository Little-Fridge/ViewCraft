---
name: video-event-localization
description: Locate actions and state transitions in a video or timestamped frames, track object identity, and report event intervals at the precision actually observed.
---

# Video Event Search & Temporal Grounding

## Use
Accept a video and event definition, or timestamped frames. The CLI supports `--video clip.mp4`; control sampling with `--frame-interval` and refine a window with `--start` and `--end`. Timestamps refer to the original clip start, even for a selected window.

## Define the event
Translate the question into observable states. Separate approaching, contacting, moving, releasing and remaining still. For 'last', 'before' or 'after', identify the reference event and the temporal scope before selecting frames.

## Search broadly, then locally
1. Inspect broad temporal coverage to find candidate intervals. Uniform sampling, high motion and scene changes are cues, not guarantees that the decisive event is visible.
2. Track the target by stable appearance and surrounding context. Camera motion can shift every object; compare against scene anchors before attributing movement.
3. Request neighboring source frames around a candidate. Keep timestamps attached and compare preceding state, transition and following state in time order.
4. If the prepared pool misses the transition, say which interval needs denser sampling. With access to the raw video, rerun that interval at a smaller `--frame-interval`. Enlarging one frame cannot recover a missing moment.

## Verify identity and completion
Distinguish disappearance behind an occluder from exit or removal. Match multiple similar objects using their trajectories and distinguishing features. For a completed action, look for its postcondition; a hand touching a container does not prove an object was placed inside. For a static-state question, inspect separated times near the end rather than only the last frame.

## Deliver
Report the last observed pre-event timestamp and first observed post-event timestamp as a bracket when onset is uncertain. Give event order, object identity and the evidence times supporting it. Match precision to frame spacing. Do not infer sound or dialogue from frames. Label unobserved intervals and avoid claiming that a brief event never happened from sparse sampling alone.

## Execution
In ViewCraft, use the observation actions and source IDs supplied by the runtime. Finish with its `action=answer` object, with the complete requested output in the string `answer` field. In another visual agent, use its available inspection tools and return the same user-facing result directly.
