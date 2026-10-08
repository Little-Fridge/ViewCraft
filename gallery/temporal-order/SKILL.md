---
name: temporal-order
description: Identify the event immediately before or after a queried event in a short sequence of timestamped scene frames.
---

# Temporal order

This ViewCraft skill answers temporal adjacency questions from four to six scene frames. Use visible actions, object states, and timestamps to identify the immediate predecessor or successor of a queried event. The requested relation concerns the supplied sequence; sparse still images do not establish every action that occurred between samples.

Read the question first and identify its anchor event, requested direction, and any time qualifier. Distinguish “immediately before” from “at any earlier time,” and “immediately after” from a later outcome. If answer choices are supplied, keep their labels available while inspecting the frames, but do not use their wording as evidence that an action occurred.

Establish chronological order from the provided timestamps. The default layout presents a contact sheet with timestamp captions, and the source frames remain available through their original indices. Read tiles left to right, then top to bottom, checking their times rather than assuming spatial position establishes chronology. Embedded timestamps help preserve the association if a view is cropped or the runtime presents a subset. If time metadata conflicts with an embedded timestamp, acknowledge the conflict rather than silently choosing a convenient order.

Locate the anchor using a distinctive visual transition. Track the same actor, object, and scene region across neighboring frames. Separate an action in progress from a persistent resulting state: an object remaining inside a container is not another insertion, and a label remaining attached is not another labeling event. A change in hands, tools, contact points, or object position can disambiguate repeated states. For repeated anchor actions, use the occurrence identified by the question; do not merge separate occurrences.

Inspect the adjacent earlier frame for a before question or adjacent later frame for an after question. Compare it with the anchor and one surrounding frame when needed to name the event accurately. Prefer the smallest event description supported by visible evidence. Do not substitute a plausible workflow step for an observed action. If the anchor is at a sequence boundary, report that the requested neighbor is not shown.

Answer directly when the relevant frames are readable. Use an additional source view or crop only to resolve a specific ambiguity, retaining enough context to identify the frame and timestamp. Do not repeatedly request an already readable overview. Stay within six observations and six images per request. Return action=answer using the runtime's final-answer schema. For multiple-choice questions, put only the selected letter in the answer field; otherwise provide a short event description. If the sequence cannot resolve the relation, state that limitation instead of inventing an intervening event.
