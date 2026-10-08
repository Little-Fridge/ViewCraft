"""Regression tests for the VIEW-GRAPH coordinate channel.

The mechanism under test: every coordinate the model emits -- a zoom region,
a note point, a click answer -- may be expressed in the frame of the attached
image it names, and the RUNTIME maps it back to the source in code. These
tests pin the invariants, not any one strategy:

  * (view, u, v) -> source resolution is exact for crops and identity for
    full frames (backward compatible with every legacy trajectory)
  * a composite sheet is NOT addressable (no single source transform)
  * describe_visible can never silently delete the per-image manifest
    (measured: sspro joint candidate 0.4845 -> 0.1056 when it did)
  * the answer protocol text is a psi field like the view contract is
  * the spatial failure diagnostic needs no gold and steers attribution
    away from theta when the answer is not anchored to the model's own view
"""
from __future__ import annotations

import io

from PIL import Image

from viewcraft.runtime import (
    RenderedView, _map_point_to_source, _map_point_to_view,
    _map_region_to_source, _resolve_answer_point, _with_notes_on_view,
    build_task_prompt, describe_visible, parse_model_turn,
)
from viewcraft.schema import Region, ViewRecord
from viewcraft.views.config import Coordinates, PhiConfig, PsiConfig


def _view(kind="crop", region=(0.4, 0.3, 0.6, 0.5), out=(200, 100),
          src=(3840, 2160), source="/tmp/fake.png", created_by="phi.crop"):
    reg = Region(*region) if region else None
    rec = ViewRecord(
        view_id="v_test", source_media_id=source, kind=kind,
        long_edge_px=1024, out_w=out[0], out_h=out[1],
        src_w=src[0], src_h=src[1], region=reg, scale=1.0,
        created_by=created_by)
    png = io.BytesIO()
    Image.new("RGB", out, (40, 40, 40)).save(png, format="PNG")
    return RenderedView(rec, png.getvalue())


# ---------------------------------------------------------------- transforms


def test_point_maps_through_crop_frame_exactly():
    rec = _view().record                      # region x .4-.6, y .3-.5
    sx, sy = _map_point_to_source((0.5, 0.5), rec)
    assert abs(sx - 0.5) < 1e-9 and abs(sy - 0.4) < 1e-9


def test_full_frame_mapping_is_identity():
    rec = _view(kind="full", region=(0.0, 0.0, 1.0, 1.0)).record
    assert _map_point_to_source((0.123, 0.456), rec) == (0.123, 0.456)


def test_region_maps_through_crop_frame():
    rec = _view().record
    out = _map_region_to_source(Region(0.0, 0.0, 0.5, 0.5), rec)
    assert (round(out.x0, 6), round(out.y0, 6),
            round(out.x1, 6), round(out.y1, 6)) == (0.4, 0.3, 0.5, 0.4)


def test_point_to_view_inverse_and_outside():
    rec = _view().record
    u, v = _map_point_to_view((0.5, 0.4), rec)
    assert abs(u - 0.5) < 1e-9 and abs(v - 0.5) < 1e-9
    assert _map_point_to_view((0.1, 0.4), rec) is None   # left of the crop


# ------------------------------------------------------- answer resolution


def test_answer_inside_crop_resolves_to_source():
    crop = _view()
    text, note = _resolve_answer_point("0.5, 0.5", 0, [crop])
    assert text == "0.5000, 0.4000"
    assert "resolved" in note


def test_answer_on_full_frame_is_untouched():
    full = _view(kind="full", region=(0.0, 0.0, 1.0, 1.0))
    text, note = _resolve_answer_point("0.412, 0.867", 0, [full])
    assert text == "0.412, 0.867" and note == ""


def test_answer_without_view_is_untouched():
    crop = _view()
    text, note = _resolve_answer_point("0.412, 0.867", None, [crop])
    assert text == "0.412, 0.867" and note == ""


def test_answer_naming_composite_falls_back_to_full_frame():
    comp = _view(source="/tmp/a.png,/tmp/b.png",
                 created_by="psi.layout_expr")
    text, note = _resolve_answer_point("0.5, 0.5", 0, [comp])
    assert text == "0.5, 0.5" and note == ""


def test_answer_pixel_coords_within_crop_are_normalised_first():
    crop = _view(out=(200, 100))
    text, _ = _resolve_answer_point("100, 50", 0, [crop])
    # (0.5, 0.5) in the crop -> source (0.5, 0.4)
    assert text == "0.5000, 0.4000"


def test_parse_captures_answer_view():
    r = parse_model_turn(
        '{"action":"answer","answer":"0.4, 0.8","image":2}', 0)
    assert r.action == "answer" and r.answer_view == 1
    r2 = parse_model_turn('{"action":"answer","answer":"0.4, 0.8"}', 0)
    assert r2.answer_view is None


# ------------------------------------------------------------ the manifest


def test_contract_without_placeholder_appends_manifest():
    views = [_view(kind="full", region=(0.0, 0.0, 1.0, 1.0))]
    out = describe_visible(views, contract="Custom words about the images.")
    assert "Custom words" in out
    assert "Image 1" in out, "manifest was amputated by a placeholder-free contract"


def test_contract_with_placeholder_substitutes_once():
    views = [_view(kind="full", region=(0.0, 0.0, 1.0, 1.0))]
    out = describe_visible(views, contract="HEAD {image_lines} TAIL")
    assert out.startswith("HEAD") and out.endswith("TAIL")
    assert out.count("Image 1") == 1


def test_crop_line_has_four_decimal_provenance():
    views = [_view()]
    out = describe_visible(views)
    assert "0.4000-0.6000" in out and "0.3000-0.5000" in out


def test_composite_declared_unaddressable():
    views = [_view(created_by="psi.layout_expr",
                   source="/tmp/a.png,/tmp/b.png")]
    out = describe_visible(views)
    assert "COMPOSITE" in out and "cannot zoom" in out


# ---------------------------------------------------------- answer contract


def test_answer_contract_overrides_default_block():
    coords = Coordinates(psi=PsiConfig(answer_contract="ANSWER LIKE THIS."))

    from viewcraft.schema import PublicItem
    item = PublicItem("test", "custom", "q")
    prompt = build_task_prompt(item, coords, 3, None, "point")
    assert "ANSWER LIKE THIS." in prompt


def test_answer_contract_default_is_domains_style():
    coords = Coordinates()

    from viewcraft.schema import PublicItem
    item = PublicItem("test", "custom", "q")
    prompt = build_task_prompt(item, coords, 3, None, "point")
    assert "CLICK POINT" in prompt


# ------------------------------------------------------------- canvas on crops


def test_notes_overlay_only_marks_notes_inside_the_crop():
    crop = _view(region=(0.4, 0.3, 0.6, 0.5))
    coords = Coordinates()          # default marker
    notes = {"/tmp/fake.png": [
        {"x": 0.5, "y": 0.4, "text": "1"},       # inside the crop
        {"x": 0.05, "y": 0.05, "text": "2"},     # far outside
    ]}

    class _Item:
        meta = {}
        meta = {}
        options = ()
        media_ids = ("/tmp/fake.png",)
        question = "q"
    out = _with_notes_on_view(crop, notes, coords, _Item(), 1, 0)
    import numpy as np
    changed = (np.asarray(Image.open(io.BytesIO(out.png_bytes)))
               != np.asarray(Image.open(io.BytesIO(crop.png_bytes)))).any()
    assert changed, "a note inside the crop was not rendered onto it"


def test_notes_overlay_noop_when_no_note_inside():
    crop = _view(region=(0.4, 0.3, 0.6, 0.5))
    coords = Coordinates()
    notes = {"/tmp/fake.png": [{"x": 0.05, "y": 0.05, "text": "2"}]}

    class _Item:
        meta = {}
        meta = {}
        options = ()
        media_ids = ("/tmp/fake.png",)
        question = "q"
    out = _with_notes_on_view(crop, notes, coords, _Item(), 1, 0)
    assert out.png_bytes == crop.png_bytes


# ------------------------------------------------------------- attribution


def test_spatial_mismatch_steers_attribution_away_from_theta():
    from viewcraft.evolution.propose import select_coordinate
    failures = [{
        "grounding": {"per_requirement": {"r1": "supported",
                                          "r2": "supported"}},
        "spatial": {"n_crops": 1, "answer_inside_own_last_crop": False},
        "observation_steps": 2,
    }]
    coord = select_coordinate(failures, "view_only", 0)
    assert coord == "psi", "an answer unanchored from the model's own view " \
                           "must not be read as a skill-text failure"


def test_format_failures_shows_spatial_line():
    from viewcraft.evolution.propose import format_failures
    text = format_failures([{
        "outcome": "wrong", "question": "q", "answer": "0.1, 0.2",
        "observation_steps": 2, "n_images_sent": 2, "view_summary": [],
        "spatial": {"n_crops": 1, "answer_inside_own_last_crop": False},
        "requirements": {}, "grounding": None,
    }])
    assert "answer lands inside the model's own last zoom: False" in text


def test_answer_view_reference_in_free_text():
    """The executor also emits the view reference as free text ("| image 2"),
    which is what evolved thetas produce when they ask for a labelled
    verification trace -- measured on sspro_v2 viewcraft r2-r4, where the
    JSON-only reader made three candidates score ~0 regardless of merit."""
    crop = _view()
    text, note = _resolve_answer_point(
        "Copy icon | 0.5, 0.5 | image 1", None, [crop])
    assert text == "0.5000, 0.4000"
    # last reference wins over narration
    text2, _ = _resolve_answer_point(
        "looked at image 1, but the target is at 0.5, 0.5 on image 1",
        None, [crop])
    assert text2 == "0.5000, 0.4000"
    # the "image=2" dialect, measured on sspro_v3 round 3 (0.708 -> 0.000
    # when the space-only reader missed it)
    text3, _ = _resolve_answer_point(
        "NEIGHBOURS: a | b\nANSWER: image=1, x=0.5, y=0.5", None, [crop])
    assert text3 == "0.5000, 0.4000"


def test_answer_without_action_key_parses():
    """{"note": ..., "answer": ...} with NO "action" key is still an answer.
    Evolved thetas keep inducing this dialect (217 answers across 8 runs,
    ~2/3 previously scored as free text -- a viewcraft-only suppression)."""
    r = parse_model_turn(
        '```json\n{"note": "ledger: frame 1 shows A", "answer": "A"}\n```', 4)
    assert r.action == "answer" and r.answer == "A"
    r2 = parse_model_turn('{"note": "x", "answer": "0.4, 0.8", "image": 2}', 0)
    assert r2.action == "answer" and r2.answer_view == 1
    # but a note-only object is not an answer
    r3 = parse_model_turn('{"note": "just thinking"}', 0)
    assert r3.action == "answer" and r3.answer != ""  # free text fallback


def test_point_action_alias_parses():
    r = parse_model_turn(
        '{"action": "point", "point": [0.4, 0.8], "image": 1}', 0)
    assert r.action == "answer" and "0.4" in r.answer and r.answer_view == 0


def test_step0_default_is_uniform_not_tail():
    """An empty select_expr over a pool larger than the image cap must send a
    UNIFORM spread of the pool at step 0, not the last max_imgs units.
    Measured on docvqa2026 business_report_3_q2: the tail-only default sent
    the assurance/GRI pages of a ~100-page report and the model answered
    "Data not available" for data it could never have reached."""
    import io as _io
    from PIL import Image as _Image
    from viewcraft.runtime import run_static_item
    from viewcraft.schema import PublicItem
    from viewcraft.views.config import Coordinates

    class _Resp:
        class usage:
            input_tokens = 1
            output_tokens = 1
        content = '{"action": "answer", "answer": "A"}'
        request_id = "r1"

    class _Client:
        def call(self, messages, max_tokens=1024):
            return _Resp()

    paths = []
    for i in range(20):
        p = f"/tmp/vf_pool_{i}.png"
        _Image.new("RGB", (16, 16), (i * 12, 0, 0)).save(p)
        paths.append(p)
    item = PublicItem(item_id="t/1", domain="t", question="q?",
                      media_ids=tuple(paths), options=("A", "B"))
    traj = run_static_item(item, Coordinates(), _Client(),
                           {"observations": 2, "max_images_per_request": 6},
                           "ph", arm="baseline", condition="normal",
                           split="test", answer_style="letter")
    sent = [v.source_media_id for v in traj.views
            if v.kind in ("full", "overview") and v.actually_sent]
    assert len(sent) == 6
    idx = sorted(paths.index(s) for s in sent)
    assert idx != [14, 15, 16, 17, 18, 19], "tail-only default is back"
    assert idx[0] < 6 and idx[-1] > 13, f"not a uniform spread: {idx}"


def test_point_domain_refuses_composite_candidates():
    from viewcraft.evolution.chain import Chain
    from viewcraft.evolution.propose import Candidate, CandidateRejected
    from viewcraft.views.config import Coordinates, PsiConfig

    class _Spec:
        answer_style = "point"
    ch = Chain.__new__(Chain)
    ch.spec = _Spec()
    cur = Coordinates()
    ch.coords = cur
    cand = Candidate(coordinate="joint",
                     coords=Coordinates(psi=PsiConfig(
                         layout_expr="grid(views, cols=2)")),
                     rationale="", diff="psi.layout_expr: grid")
    try:
        ch._assert_candidate_keeps_point_frames(cand)
        raise AssertionError("composite candidate was not refused")
    except CandidateRejected:
        pass
    # a pixel-side candidate on the same domain is fine
    ok = Candidate(coordinate="phi",
                   coords=Coordinates(phi=PhiConfig(allow_upscale=True)),
                   rationale="", diff="phi.allow_upscale")
    ch._assert_candidate_keeps_point_frames(ok)
    # and the layout is fine on a letter domain
    class _Letter:
        answer_style = "letter"
    ch.spec = _Letter()
    ch._assert_candidate_keeps_point_frames(cand)
