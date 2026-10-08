"""Regression tests for the VISUAL MEMORY LAYER (canvas + note action).

The mechanism under test: the executor drops `note`s (point + own text) on the
image, and the optimizer's `canvas_expr` decides how the accumulated notes are
rendered back. These tests pin the INVARIANTS, not any one strategy:
  * a note action parses from the dialects executors actually emit
  * notes accumulate per source and are re-derived from the RAW base (no
    double-baking)
  * the default marker and an evolved canvas_expr both change the emitted view
  * the canvas cannot buy extra context (it overlays the existing overview)
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from viewcraft.runtime import parse_model_turn, _notes_from
from viewcraft.views.exprs import render_canvas, canvas_env
from viewcraft.views.config import Coordinates, PhiConfig


def _img(w=200, h=120, color=(30, 30, 30)):
    return Image.new("RGB", (w, h), color)


def test_note_action_parses_dialects():
    r = parse_model_turn(
        '{"action":"note","notes":[{"at":[0.5,0.3],"text":"3"}]}', 0)
    assert r.action == "note"
    assert r.canvas_notes[0]["x"] == 0.5 and r.canvas_notes[0]["y"] == 0.3
    assert r.canvas_notes[0]["text"] == "3"


def test_note_accepts_gemini_grid_and_aliases():
    # mark/annotate aliases + 0-1000 grid must not be silently dropped.
    for act in ("note", "mark", "annotate"):
        r = parse_model_turn(f'{{"action":"{act}","at":[500,300],"text":"a"}}', 0)
        assert r.action == "note", act
        assert abs(r.canvas_notes[0]["x"] - 0.5) < 1e-6


def test_malformed_note_gets_reminder_path_not_answer():
    # Declaring a note with no usable point is a protocol failure (reminder),
    # never a scored answer.
    r = parse_model_turn('{"action":"note","notes":[{"text":"no point"}]}', 0)
    assert r.action == "malformed_zoom"


def test_default_canvas_marks_and_labels():
    base = _img()
    notes = [{"x": 0.25, "y": 0.5, "text": "1"},
             {"x": 0.75, "y": 0.5, "text": "2"}]
    out = render_canvas(base, "", notes, {})
    assert out.size == base.size
    changed = (np.asarray(out) != np.asarray(base)).any(axis=2).sum()
    assert changed > 0, "default canvas made no visible mark"


def test_canvas_expr_reads_note_arrays():
    base = _img()
    notes = [{"x": 0.25, "y": 0.5, "text": "1"}]
    env = canvas_env(base, notes, {})
    assert env["note_count"] == 1.0
    assert float(env["dist"].min()) < 0.01      # nearest note is near its point
    assert float(env["nearest_idx"].max()) == 1.0
    # an expression keyed on dist marks only near the note
    out = render_canvas(base, "where(dist<0.05,255,red)", notes, {})
    assert (np.asarray(out) != np.asarray(base)).any(axis=2).sum() > 0


def test_canvas_expr_empty_notes_is_safe_not_fatal():
    # No notes yet: dist is +inf, so where(dist<r,...) is simply false
    # everywhere. This is an ITEM property (nothing jotted yet), so it must be
    # safe, not an error -- the recurring "item's nature fails the candidate"
    # defect.
    base = _img()
    env = canvas_env(base, [], {})
    assert np.isinf(env["dist"]).all()
    out = render_canvas(base, "where(dist<0.05,255,red)", [], {})
    assert out.size == base.size


def test_canvas_expr_is_a_phi_coordinate():
    c = Coordinates(phi=PhiConfig(canvas_expr="where(dist<0.05,255,red)"))
    assert c.phi.canvas_expr
    c2 = Coordinates(phi=PhiConfig())
    assert c.config_hash != c2.config_hash


def test_canvas_shows_global_ordinal_not_just_model_text():
    # The mechanism, not the model, renders the note's position in the sequence
    # -- so the running count is always visible and correct even when the model
    # loses track mid-trajectory (the 27-for-28 restart that motivated this).
    base = _img()
    notes = [{"x": 0.2, "y": 0.5, "text": ""},     # model gave no number
             {"x": 0.5, "y": 0.5, "text": ""},
             {"x": 0.8, "y": 0.5, "text": ""}]
    out = render_canvas(base, "", notes, {})
    # The ordinals 1,2,3 are drawn (yellow on a dark backing). We assert the
    # label pixels exist near each note rather than OCR the digits.
    arr = np.asarray(out)
    for n in notes:
        px, py = int(n["x"] * 199), int(n["y"] * 119)
        region = arr[max(0, py - 9):py + 10, px:px + 40]
        has_label = (region > 200).any()      # bright yellow/black label block
        assert has_label, f"no ordinal label near note at {n['x'], n['y']}"


def test_bare_json_array_is_an_answer_not_a_crash():
    # ARC regression: a bare grid [[...]] is a legitimate answer. Calling
    # .get("action") on the list used to raise AttributeError, turning a
    # CORRECT grid answer into a needs_review failure and breaking the paired
    # gate ("gate sets differ by N items").
    r = parse_model_turn('{"rule":"tiling"} [[7, 0, 7], [0, 7, 0]]', 0)
    assert r.action == "answer"
    assert "7" in r.answer
    # a top-level array on its own line also parses
    r2 = parse_model_turn('[[1, 2], [3, 4]]', 0)
    assert r2.action == "answer"
