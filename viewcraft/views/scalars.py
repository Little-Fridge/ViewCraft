"""The scalar names every expression language binds -- in ONE place.

WHY THIS MODULE EXISTS
----------------------
Three languages (selection, pixel, layout) each grew their own ad-hoc
namespace. The optimizer, which writes all three in a single joint proposal,
kept reaching for a name that exists in a sibling language or in our own
prompt text, and losing the WHOLE proposal to `unknown name`:

    nextgqa   phi.select_expr   unknown name 'n_frames'
                                (available: ['budget', 'n'])
    chartqa   psi.layout_expr   unknown name 'height'
                                (available: crops, views, times, index, ...)
    vsi       phi.pixel_expr    unknown name 'ગ11'

The first two are OUR defect, not the optimizer's. `selection_help()`
literally advertises "names: n (pool size), budget, duration_s, n_pages,
n_frames" -- the optimizer wrote `n_frames` because the prompt told it to,
and the validator rejected it because the env only bound `n` and `budget`.
`height` is bound in the pixel language and not in layout, though asking for
a tile height while composing a layout is the more natural of the two.

There was a second, nastier half. `eval_selection_expr` bound duration_s /
n_pages / n_frames only `if k in ctx`, and ctx comes from each item's meta.
So `n_frames` is bound on MVBench items, absent on NExT-GQA ones, and absent
entirely in the pre-flight validator, which passes `{}`. That makes a
candidate's validity depend on which item it happens to meet -- the same
recurring defect as the layout size guards: AN ITEM'S PROPERTY FAILING THE
WHOLE CANDIDATE.

THE INVARIANT, ENFORCED HERE AND NOWHERE ELSE
---------------------------------------------
A name that any language advertises is bound in EVERY language, on EVERY
item, ALWAYS -- with a defined fallback when the domain cannot supply it.
An expression's validity is then a property of the expression alone, which
is the only thing the optimizer can reason about.

Falling back is right rather than lax: `n_pages` on a video is honestly the
number of source units, and a selection written as `range(0, n_pages, 2)`
means exactly what it should. The alternative -- rejecting -- converts a
visual proposal into a text-only edit, which is the bias this whole project
exists to remove.
"""
from __future__ import annotations

# Every scalar name the optimizer may write, in any of the three languages,
# with the rule for deriving it when the domain does not report it.
#
# Keep this list and `selection_help()` / the pixel and layout prompt notes
# in agreement; `tests/test_regressions.py` asserts they do not drift.
SHARED_SCALAR_NAMES = (
    "n",           # size of the source pool (frames, pages, images)
    "budget",      # hard cap on how many units may be sent
    "n_frames",    # video: frame count; otherwise == n
    "n_pages",     # document: page count; otherwise == n
    "duration_s",  # video: seconds; otherwise 0.0
    "fps",         # video: frames per second; otherwise 0.0
    "width",       # pixels of the view being composed
    "height",
)


def shared_scalars(n: int, budget: int, ctx: dict | None = None,
                   width: float = 0.0, height: float = 0.0) -> dict:
    """Bind every shared name, whatever the domain reported.

    `ctx` is an item's meta. Names it supplies win; names it omits fall back
    to the pool size (counts) or 0.0 (physical quantities), so no name is
    ever merely absent.
    """
    ctx = ctx or {}

    def num(key, default):
        if key not in ctx:
            return float(default)
        try:
            return float(ctx[key])
        except (TypeError, ValueError):
            return float(default)

    n = float(n)
    return {
        "n": n,
        "budget": float(budget),
        # A count the domain cannot report is the pool size: on a video,
        # "pages" really are the units you may select among.
        "n_frames": num("n_frames", n),
        "n_pages": num("n_pages", n),
        # Physical quantities have no sensible stand-in, so they are 0.0 and
        # an expression dividing by them is the optimizer's own error.
        "duration_s": num("duration_s", 0.0),
        "fps": num("fps", 0.0),
        "width": float(width),
        "height": float(height),
    }
