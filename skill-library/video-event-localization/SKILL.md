---
name: video-event-localization
description: Locate actions and state changes in supplied timestamped video frames.
  Use for event order, object entry/exit and identifying relevant moments.
---

# 视频事件定位

Establish the supplied time range and sampling intervals. Track the target by appearance and surrounding context across frames; absence under occlusion is not proof of removal. Distinguish a hand approaching an object from contact, movement and completed placement.

Locate candidate time windows from broad coverage and inspect neighboring supplied frames around the relevant transition. When the full frame pool is available to the runtime, request those source frames rather than re-cropping an irrelevant instant. Keep timestamps attached when comparing views. Do not treat uniform sampling as proof that a short event was captured.

Report directly observed timestamps and bracket an uncertain transition between the last preceding state and first following state. Do not claim second-level precision from widely spaced frames or infer audio evidence from pictures. If the decisive interval is missing, say that denser frames from that interval are required. The current library runner accepts frame images and timestamps, not a raw video file.

Finish with the runtime action=answer object. Its answer field is a string containing the requested output.
