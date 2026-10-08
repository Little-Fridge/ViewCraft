"""Grouping question types by what they need from the imagery.

WHY THIS EXISTS
---------------
We evolve one coordinate triple per domain, and a benchmark is usually
several tasks that want opposite things. Measured on VSI-Bench: a contact
sheet moved `object_size_estimation` from 0.669 to 0.551 while every other
type stayed at 0.283 -> 0.285, and all five rejected visual candidates
improved 1-4 types while degrading 4-5. Choosing the best config PER TYPE
from the same candidate pool would have scored 0.580 against 0.453 for the
best single config.

WHY NOT ONE GROUP PER TYPE
--------------------------
Because that headroom is mostly selection noise. Splitting each type's
questions in half, choosing the config on one half and scoring on the other,
the +0.127 oracle gain becomes +0.067 median -- with a 5th percentile of
0.408 against a baseline of 0.418. VSI's types hold 8-24 gate questions
each; a gate that small makes eps = 1/n large and the acceptance rule
mostly noise.

So types are merged into groups that want the SAME thing, which keeps each
gate large enough to decide anything. The grouping is read off the data
rather than asserted: correlating each type's score vector across the 11
measured configs gives two clusters on VSI,

    metric      size <-> room_size 0.86, abs_distance <-> size 0.79
    relational  counting <-> rel_direction_hard 0.61,
                rel_direction_easy <-> route_planning 0.57

which is the physical distinction the failure analysis predicted: absolute
scale needs native resolution per frame, cross-frame comparison needs the
frames composed.

Held out the same way, two groups give +0.057 median over baseline and
+0.022 over the best single config. That is a real but modest edge, and its
5th percentile is still slightly negative, so this is a hypothesis to TEST
rather than a result to report. It is worth running because the test split
cannot answer it from existing data: only two configs were ever measured on
both gate and test.

FAIRNESS
--------
The grouping is applied to every run being compared, or to none. Giving one
run a finer grouping than another turns a comparison between them into a
measurement of the grouping.
"""
from __future__ import annotations

# Types whose questions need ABSOLUTE SCALE: answerable only if a frame is
# delivered at enough resolution to read real-world size from it. Composing
# frames into a sheet is what breaks them.
METRIC_TYPES = {
    # VSI-Bench
    "object_size_estimation",
    "room_size_estimation",
    "object_abs_distance",
    # MVBench's counting/attribute tasks depend on resolving small objects
    "action_count",
    "moving_count",
    "object_count",
    "fine_grained_action",
}


def group_of(question_type: str) -> str:
    """Which requirement group a question type belongs to.

    Two groups, not one per type: see the module docstring for the held-out
    measurement that motivates merging.
    """
    t = (question_type or "").strip()
    if not t:
        return "all"
    return "metric" if t in METRIC_TYPES else "relational"


def group_of_sample(sample) -> str:
    """Group for a loaded sample, from adapter metadata then the item id.

    Reads `question_type_name`/`question_type` first. Parsing the type out
    of `<domain>/<type>/<id>` works for VSI and the document benchmarks and
    fails silently for NExT-GQA, whose ids carry no type -- which is how an
    earlier analysis concluded NExT-GQA was single-task when its adapter
    records five types.
    """
    meta = getattr(sample.public, "meta", None) or {}
    t = meta.get("question_type_name") or meta.get("question_type")
    if t in (None, ""):
        parts = str(getattr(sample.public, "item_id", "")).split("/")
        t = parts[1] if len(parts) >= 3 else ""
    return group_of(str(t))
