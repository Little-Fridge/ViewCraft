"""Deterministic acceptance tests. No model calls.

These cover the failure modes that would silently invalidate results:
coordinate round-trips, images actually being attached, split disjointness,
answer/evidence leakage into the executor path, and ablations really
disabling their module.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from viewcraft.schema import (PrivateItem, PublicItem, Region, Sample,
                              TimeSpan, Trajectory, ViewRecord,
                              GroundingResult)
from viewcraft.views.config import (Coordinates, PhiConfig, PsiConfig,
                                    SkillConfig, clamp_region)
from viewcraft.views.render import (ViewConfigError, render_crop, render_full,
                                    render_grid, render_overview_with_boxes,
                                    source_point_to_view,
                                    view_point_to_source)
from viewcraft import splits
from viewcraft.evaluation import scorers


# ---------------------------------------------------------------------------
# fixtures


@pytest.fixture
def big_image(tmp_path) -> Path:
    """A 2400x1600 image with a distinctive 40x30 marker, mimicking V*Bench."""
    img = Image.new("RGB", (2400, 1600), "white")
    # marker at absolute (1800, 1200) size 40x30
    for x in range(1800, 1840):
        for y in range(1200, 1230):
            img.putpixel((x, y), (255, 0, 0))
    p = tmp_path / "big.png"
    img.save(p)
    return p


# ---------------------------------------------------------------------------
# geometry and coordinate round trips


def test_region_rejects_out_of_range():
    with pytest.raises(ValueError):
        Region(-0.1, 0, 0.5, 0.5)
    with pytest.raises(ValueError):
        Region(0, 0, 1.5, 0.5)


def test_region_rejects_empty_area():
    with pytest.raises(ValueError):
        Region(0.5, 0.5, 0.5, 0.9)


def test_pixel_roundtrip_is_stable():
    reg = Region.from_pixels((1800, 1200, 1840, 1230), 2400, 1600)
    assert reg.to_pixels(2400, 1600) == (1800, 1200, 1840, 1230)


def test_union_preserves_both_targets():
    """A relative-position question must keep both objects in one crop."""
    a = Region.from_pixels((1353, 1069, 1377, 1087), 2250, 1500)
    b = Region.from_pixels((1621, 1080, 1711, 1117), 2250, 1500)
    u = a.union(b)
    assert u.x0 <= a.x0 and u.x1 >= b.x1
    assert u.y0 <= min(a.y0, b.y0) and u.y1 >= max(a.y1, b.y1)


def test_view_point_maps_back_to_source(big_image):
    """A point identified in a crop must map back to the true source pixel."""
    reg = Region.from_pixels((1700, 1100, 1900, 1300), 2400, 1600)
    view = render_crop(big_image, "m", reg, 512, margin=0.0)
    rec = view.record
    # the marker's top-left in source coords
    vx, vy = source_point_to_view(rec, 1800, 1200)
    sx, sy = view_point_to_source(rec, vx, vy)
    assert abs(sx - 1800) < 1.5
    assert abs(sy - 1200) < 1.5


def test_crop_recovers_real_detail(big_image):
    """A tiny target unreadable in the overview becomes large in a crop."""
    overview = render_full(big_image, "m", 1024)
    # 40px marker in a 2400px image, downscaled to 1024 -> ~17px
    assert overview.record.scale < 0.5
    marker_px_in_overview = 40 * overview.record.scale
    assert marker_px_in_overview < 20

    reg = Region.from_pixels((1780, 1180, 1860, 1250), 2400, 1600)
    crop = render_crop(big_image, "m", reg, 1024, margin=0.0)
    # crop source span is 80px wide, fitted to 1024 => upscale is refused
    assert crop.record.upscaled is False
    assert crop.record.scale == 1.0
    # so the marker occupies 40 of ~80 px: half the frame instead of 1.7%
    assert crop.record.out_w <= 100


def test_no_upscale_unless_allowed(big_image):
    reg = Region.from_pixels((1800, 1200, 1840, 1230), 2400, 1600)
    plain = render_crop(big_image, "m", reg, 1024, allow_upscale=False)
    assert plain.record.upscaled is False
    assert plain.record.scale == 1.0

    up = render_crop(big_image, "m", reg, 1024, allow_upscale=True)
    assert up.record.upscaled is True
    assert up.record.out_w > plain.record.out_w


def test_grid_tiles_cover_frame_without_gaps(big_image):
    views = [render_grid(big_image, "m", 2, 2, 512, i) for i in range(4)]
    total = sum(v.record.region.width * v.record.region.height
                for v in views)
    assert abs(total - 1.0) < 1e-6


def test_grid_index_out_of_range(big_image):
    with pytest.raises(ViewConfigError):
        render_grid(big_image, "m", 2, 2, 512, 4)


def test_min_crop_frac_keeps_context_and_stays_in_bounds():
    phi = PhiConfig(min_crop_frac=0.2)
    tiny = Region(0.98, 0.98, 0.99, 0.99)
    out = clamp_region(tiny, phi)
    assert out.width >= 0.19 and out.height >= 0.19
    assert 0.0 <= out.x0 and out.x1 <= 1.0
    assert 0.0 <= out.y0 and out.y1 <= 1.0


def test_timespan_iou_and_multi_segment_max():
    gold = [(12.1, 17.1), (20.0, 23.5), (29.7, 33.2)]
    # a prediction matching the SECOND segment must not score zero
    assert scorers.temporal_iou((20.0, 23.5), gold) == pytest.approx(1.0)
    assert scorers.temporal_iou((0.0, 1.0), gold) == 0.0


# ---------------------------------------------------------------------------
# config execution


def test_unknown_phi_field_is_rejected():
    with pytest.raises(ViewConfigError):
        PhiConfig.from_dict({"crop_long_edge_px": 512, "nonexistent": 3})


def test_unknown_psi_field_is_rejected():
    with pytest.raises(ViewConfigError):
        PsiConfig.from_dict({"keep_overview": True, "made_up": 1})


def test_unsupported_psi_enum_is_rejected():
    with pytest.raises(ViewConfigError):
        PsiConfig(order="sideways")
    with pytest.raises(ViewConfigError):
        PsiConfig(evict_policy="random")


def test_config_hash_changes_with_content():
    a = Coordinates(theta=SkillConfig("x"), phi=PhiConfig(),
                    psi=PsiConfig())
    b = Coordinates(theta=SkillConfig("y"), phi=PhiConfig(),
                    psi=PsiConfig())
    c = Coordinates(theta=SkillConfig("x"), phi=PhiConfig(crop_margin=0.5),
                    psi=PsiConfig())
    assert a.config_hash != b.config_hash
    assert a.config_hash != c.config_hash


def test_snapshot_roundtrip(tmp_path):
    coords = Coordinates(theta=SkillConfig("# skill\nzoom first\n"),
                         phi=PhiConfig(crop_long_edge_px=768),
                         psi=PsiConfig(history_keep_last_n=2))
    h = coords.write_snapshot(tmp_path / "snap")
    back = Coordinates.read_snapshot(tmp_path / "snap")
    assert back.config_hash == h
    assert back.phi.crop_long_edge_px == 768
    assert back.psi.history_keep_last_n == 2
    assert "zoom first" in back.theta.text


def test_phi_change_actually_changes_pixels(big_image):
    """Changing phi must change the emitted image, not just the YAML."""
    small = render_full(big_image, "m", 512)
    large = render_full(big_image, "m", 1024)
    assert small.record.out_w != large.record.out_w
    assert small.png_bytes != large.png_bytes


# ---------------------------------------------------------------------------
# public / private separation


def test_public_item_rejects_answer_bearing_meta():
    with pytest.raises(ValueError):
        PublicItem(item_id="i", domain="vstar", question="q",
                   meta={"answer": "A"})
    with pytest.raises(ValueError):
        PublicItem(item_id="i", domain="vstar", question="q",
                   meta={"evidence_pages": [1]})


def test_public_item_has_no_answer_attribute():
    pub = PublicItem(item_id="i", domain="vstar", question="q")
    for banned in ("answer", "evidence_regions", "evidence_pages",
                   "evidence_spans"):
        assert not hasattr(pub, banned)


def test_sample_requires_matching_ids():
    pub = PublicItem(item_id="a", domain="vstar", question="q")
    priv = PrivateItem(item_id="b", answer="x")
    with pytest.raises(ValueError):
        Sample(public=pub, private=priv)


# ---------------------------------------------------------------------------
# splits


def test_split_assignment_is_disjoint_and_deterministic():
    groups = [f"g{i}" for i in range(191)]
    spec = {"mode": "by_image", "diagnostic": 24, "train": 48,
            "gate": 24, "test": 95}
    p1 = splits.build_plan("vstar", groups, spec)
    p2 = splits.build_plan("vstar", groups, spec)
    assert p1.plan_hash == p2.plan_hash
    assert p1.counts == {"diagnostic": 24, "train": 48, "gate": 24, "test": 95}
    splits.verify_disjoint(p1)
    # every group used exactly once
    assert len(p1.assignment) == 191


def test_fraction_split_uses_all_groups():
    groups = [f"d{i}" for i in range(135)]
    spec = {"mode": "by_document", "diagnostic_frac": 0.10,
            "train_frac": 0.40, "gate_frac": 0.20, "test_frac": 0.30}
    plan = splits.build_plan("mmlongdoc", groups, spec)
    assert sum(plan.counts.values()) == 135
    splits.verify_disjoint(plan)


def test_same_media_never_crosses_splits():
    """Two questions on one image must land in the same split."""
    groups = ["img1", "img1", "img2", "img2", "img3"]
    spec = {"diagnostic": 1, "train": 1, "gate": 1, "test": 0}
    plan = splits.build_plan("vstar", groups, spec)
    # dedup happened, so each image has exactly one split
    assert len(plan.assignment) == 3


def test_split_plan_persists(tmp_path):
    groups = [f"g{i}" for i in range(10)]
    plan = splits.build_plan("vstar", groups,
                             {"diagnostic": 2, "train": 4, "gate": 2, "test": 2})
    p = tmp_path / "splits.json"
    splits.save_plan(plan, p)
    back = splits.load_plan(p)
    assert back.plan_hash == plan.plan_hash


# ---------------------------------------------------------------------------
# scorers


@pytest.mark.parametrize("raw,expect", [
    ("Answer: B", 1.0),
    ("answer: b", 1.0),
    ("B", 1.0),
    ("Answer: A", 0.0),
    ("The color of the van is silver.", 1.0),   # exact option text
    ("", 0.0),
    ("I cannot tell", 0.0),
])
def test_multiple_choice_matching(raw, expect):
    options = ["The color of the van is red.",
               "The color of the van is silver.",
               "The color of the van is black.",
               "The color of the van is white."]
    assert scorers.match_multiple_choice(raw, options, 1) == expect


def test_multiple_choice_prefers_declared_answer_over_prose():
    options = ["yes", "no"]
    # 'no' appears in prose but B is declared
    assert scorers.match_multiple_choice(
        "there is no doubt. Answer: B", options, 1) == 1.0


def test_options_are_not_resorted():
    """Scoring must use the presented order, not a sorted one."""
    options = ["zebra", "apple"]
    # correct is index 0 ('zebra') as presented
    assert scorers.match_multiple_choice("A", options, 0) == 1.0
    assert scorers.match_multiple_choice("B", options, 0) == 0.0


@pytest.mark.parametrize("pred,gold,expect", [
    ("100", "100", 1.0),
    ("104", "100", 1.0),      # 4% relative -> within 5%
    ("106", "100", 0.0),      # 6% relative -> outside
    ("0.104", "0.1", 1.0),    # relative, not absolute 0.01
    ("Blue", "blue", 1.0),
    ("Red", "blue", 0.0),
    ("0", "0", 1.0),
])
def test_chartqa_relaxed_accuracy_is_relative(pred, gold, expect):
    assert scorers.relaxed_accuracy(pred, gold) == expect


def test_mra_is_threshold_hit_rate_not_one_minus_error():
    # 10% relative error: passes thresholds where 1-theta > 0.1, i.e. theta<0.9
    got = scorers.mra("11", "10")
    assert 0.0 < got < 1.0
    assert got != pytest.approx(0.9)      # not max(0, 1-rel_err)
    assert scorers.mra("10", "10") == 1.0
    assert scorers.mra("100", "10") == 0.0


def test_mmlongbench_formats():
    assert scorers.mmlongbench_match("42", "42", "Int") == 1.0
    assert scorers.mmlongbench_match("42.0", "42", "Int") == 1.0
    assert scorers.mmlongbench_match("43", "42", "Int") == 0.0
    assert scorers.mmlongbench_match("a, b", "b, a", "List") == 1.0
    assert scorers.mmlongbench_match(
        "This is not answerable from the document", "", "None") == 1.0
    assert scorers.mmlongbench_match("42", "", "None") == 0.0


def test_grounded_qa_requires_both():
    assert scorers.grounded_qa_accuracy(1.0, 0.7) == 1.0
    assert scorers.grounded_qa_accuracy(1.0, 0.3) == 0.0
    assert scorers.grounded_qa_accuracy(0.0, 0.9) == 0.0


def test_miniwob_outcomes_are_distinct():
    a = scorers.miniwob_outcome(1.0, True, False, False, False)
    assert a["status"] == "terminated" and a["success"] == 1.0
    b = scorers.miniwob_outcome(0.0, False, True, False, False)
    assert b["status"] == "truncated" and b["success"] == 0.0
    c = scorers.miniwob_outcome(0.0, False, False, True, False)
    assert c["status"] == "infra_error"
    d = scorers.miniwob_outcome(0.0, False, False, False, True)
    assert d["status"] == "wall_timeout"
    # a negative-reward normal termination is not a success
    e = scorers.miniwob_outcome(-1.0, True, False, False, False)
    assert e["status"] == "terminated" and e["success"] == 0.0


# ---------------------------------------------------------------------------
# grounding aggregation


def test_grounding_weights_and_unknown_rate():
    g = GroundingResult.from_labels({
        "r1": "supported", "r2": "partial",
        "r3": "unsupported", "r4": "unknown"})
    assert g.score == pytest.approx((1.0 + 0.5 + 0.0 + 0.0) / 4)
    assert g.unknown_rate == 0.25
    assert g.n_requirements == 4


def test_grounding_rejects_empty():
    with pytest.raises(ValueError):
        GroundingResult.from_labels({})


# ---------------------------------------------------------------------------
# trajectory bookkeeping


def _rec(sent: bool, w=100, h=100) -> ViewRecord:
    return ViewRecord(view_id=f"v{w}{h}{sent}", source_media_id="m",
                      kind="crop", long_edge_px=w, out_w=w, out_h=h,
                      src_w=1000, src_h=1000, region=Region(0, 0, .1, .1),
                      actually_sent=sent)


def test_only_sent_views_count_toward_pixels():
    t = Trajectory(item_id="i", domain="vstar", split="train", arm="baseline",
                   condition="normal", config_hash="c", protocol_hash="p")
    t.views = [_rec(True, 100, 100), _rec(False, 200, 200)]
    assert t.n_images_sent == 1
    assert t.sent_pixels == 100 * 100


def test_unit_key_distinguishes_conditions():
    base = dict(item_id="i", domain="vstar", split="train", arm="viewcraft",
                config_hash="c", protocol_hash="p")
    a = Trajectory(condition="normal", **base)
    b = Trajectory(condition="oracle", **base)
    assert a.unit_key() != b.unit_key()


def test_unit_key_is_stable_for_identical_conditions():
    base = dict(item_id="i", domain="vstar", split="train", arm="viewcraft",
                condition="normal", config_hash="c", protocol_hash="p")
    assert Trajectory(**base).unit_key() == Trajectory(**base).unit_key()


def test_unit_key_separates_protocol_versions():
    base = dict(item_id="i", domain="vstar", split="train", arm="viewcraft",
                condition="normal", config_hash="c")
    a = Trajectory(protocol_hash="p1", **base)
    b = Trajectory(protocol_hash="p2", **base)
    assert a.unit_key() != b.unit_key()


# ---------------------------------------------------------------------------
# MMLongBench-Doc adapter
#
# These cover the invariants that would silently invalidate a mmlongdoc
# number: the unanswerable half scoring by abstention, evidence pages never
# reaching the executor, and the released List gold being comparable at all.


def test_mmlongdoc_unanswerable_requires_abstention():
    """22% of the benchmark is unanswerable; those items score ABSTENTION.

    A model that always emits a concrete answer must score 0 here, and a
    model that declines must score 1. Getting this backwards would make the
    domain's headline number a measure of willingness to guess.
    """
    from viewcraft.adapters import mmlongdoc
    assert mmlongdoc.compare("The document does not state this", "", "None") == 1.0
    assert mmlongdoc.compare("Not answerable", "", "None") == 1.0
    assert mmlongdoc.compare("42", "", "None") == 0.0


def test_mmlongdoc_abstention_never_scores_on_answerable_items():
    """Declining must not earn credit outside the unanswerable half."""
    from viewcraft.adapters import mmlongdoc
    for fmt, gold in (("Int", "8"), ("Float", "21%"), ("Str", "Europe"),
                      ("List", "a, b")):
        assert mmlongdoc.compare("not answerable", gold, fmt) == 0.0


def test_mmlongdoc_list_gold_is_canonicalised_out_of_python_syntax():
    """The released List gold is a Python literal STRING, not a list.

    Comparing a model's "White, 10%" against the raw characters
    "['White', '10%']" fails on the brackets and quotes, so the adapter
    canonicalises the gold before storing it. Without this a correct list
    answer scores 0.
    """
    from viewcraft.adapters import mmlongdoc
    gold = mmlongdoc.canonical_gold("['White', '10%']", "List")
    assert gold == "White, 10%"
    assert mmlongdoc.compare("White, 10%", gold, "List") == 1.0
    # order-insensitive, but not membership-insensitive
    assert mmlongdoc.compare("10%, White", gold, "List") == 1.0
    assert mmlongdoc.compare("White", gold, "List") == 0.0


def test_mmlongdoc_float_uses_official_rounding_not_relative_tolerance():
    """Float is round-to-2 equality, NOT the 1% relative tolerance.

    scorers.mmlongbench_match would accept 18.11-18.47 against a gold of
    18.29%; the official comparison does not, and this domain's scorer must
    not route through that branch.
    """
    from viewcraft.adapters import mmlongdoc
    assert mmlongdoc.compare("18.29", "18.29%", "Float") == 1.0
    assert mmlongdoc.compare("18.29%", "18.29%", "Float") == 1.0
    assert mmlongdoc.compare("18.4", "18.29%", "Float") == 0.0
    # the deviating helper would have accepted it
    assert scorers.mmlongbench_match("18.4%", "18.29%", "Float") == 1.0


def test_mmlongdoc_percent_units_are_stripped_not_rescaled():
    """78 of 160 Float golds carry '%'. Dividing one side only would break all."""
    from viewcraft.adapters import mmlongdoc
    assert mmlongdoc.compare("51%", "51%", "Float") == 1.0
    assert mmlongdoc.compare("0.51", "51%", "Float") == 0.0


def test_mmlongdoc_extracts_trailing_number_for_numeric_formats():
    from viewcraft.adapters import mmlongdoc
    assert mmlongdoc.extract_short_answer(
        "I count 2 charts and 6 tables, so 8", "Int") == "8"
    assert mmlongdoc.extract_short_answer("Answer: 12", "Int") == "12"
    # an abstention must survive rather than be reduced to a stray digit
    got = mmlongdoc.extract_short_answer("This is not answerable", "Int")
    assert "not answerable" in got.lower()


def test_mmlongdoc_uniform_page_pool_spans_the_whole_document():
    """A capped pool must not be a prefix: the back of a long report stays
    reachable, and the scheme matches views.selection._uniform exactly."""
    from viewcraft.adapters import mmlongdoc
    from viewcraft.views.selection import _uniform
    idx = mmlongdoc.uniform_page_indices(468, 128)
    assert len(idx) == 128
    assert idx == sorted(set(idx))
    assert max(idx) > 400          # not a prefix of the document
    assert idx == _uniform(468, 128)
    # a short document is pooled whole
    assert mmlongdoc.uniform_page_indices(9, 128) == list(range(9))


def test_mmlongdoc_public_item_cannot_carry_evidence_pages():
    """Evidence pages are for post-hoc binning only.

    PublicItem bans the key outright, so a refactor that tried to pass
    evidence to the executor fails as a type error rather than leaking.
    """
    with pytest.raises(ValueError):
        PublicItem(item_id="i", domain="mmlongdoc", question="q",
                   meta={"n_pages": 30, "evidence_pages": [5]})


def test_mmlongdoc_aggregate_separates_the_two_subsets():
    """A bare mean blends an abstention test with a retrieval one."""
    from viewcraft.adapters import mmlongdoc
    scored = [
        (1.0, PrivateItem(item_id="a", answer="8",
                          meta={"answer_format": "Int", "answerable": True})),
        (0.0, PrivateItem(item_id="b", answer="9",
                          meta={"answer_format": "Int", "answerable": True})),
        (1.0, PrivateItem(item_id="c", answer="",
                          meta={"answer_format": "None", "answerable": False})),
    ]
    agg = mmlongdoc.aggregate(scored)
    assert agg["n"] == 3
    assert agg["answerable_accuracy"] == pytest.approx(0.5)
    assert agg["n_answerable"] == 2
    assert agg["unanswerable_abstention_rate"] == pytest.approx(1.0)
    assert agg["n_unanswerable"] == 1
    assert agg["per_answer_format"]["Int"]["n"] == 2


def test_mmlongdoc_is_registered_with_its_own_scorer_name():
    from viewcraft.adapters import registry
    spec = registry.get("mmlongdoc")
    assert spec.scorer_name == "mmlongdoc_generalized_accuracy_v1"
    assert spec.answer_style == "short_text"
    # page evidence exists, but per-page REGIONS do not, so the box-cropping
    # oracle conditions must not claim to be available here
    assert spec.has_evidence_regions is False


def test_mmlongdoc_answer_declaration_is_anchored_to_a_line():
    """"Answer:" inside prose is a mention, not a declaration.

    One released Str gold is a quoted prompt containing
    "...just output 'Answer: No entities found.'". An unanchored search
    extracted that fragment and scored the gold against itself as wrong.
    """
    from viewcraft.adapters import mmlongdoc
    gold = ("Identify the entities expressed by each sentence. If you do "
            "not find any entity, just output 'Answer: No entities found.'")
    assert mmlongdoc.extract_short_answer(gold, "Str") == gold
    assert mmlongdoc.compare(
        mmlongdoc.extract_short_answer(gold, "Str"), gold, "Str") == 1.0
    # a real declaration on its own line is still honoured, last one winning
    assert mmlongdoc.extract_short_answer(
        "Let me reconsider.\nAnswer: 7\nAnswer: 8", "Int") == "8"


def test_mmlongdoc_non_numeric_int_golds_are_retyped_not_left_unwinnable():
    """An Int-labelled row with a non-numeric gold can never be scored right.

    Three released rows are like this ("yes", "GPT-4", "14:04 CET"). Left as
    Int they are worth 0 to every arm regardless of the answer; re-typed to
    Str they are winnable.
    """
    from viewcraft.adapters import mmlongdoc
    # as released, the Int branch cannot even parse the gold
    assert mmlongdoc.compare("yes", "yes", "Int") == 0.0
    # re-typed, an exactly-correct answer scores
    assert mmlongdoc.compare("yes", "yes", "Str") == 1.0
    assert mmlongdoc.compare("GPT-4", "GPT-4", "Str") == 1.0
    # and a genuinely numeric Int gold is untouched by the retype rule
    assert mmlongdoc._clean_numeric("8") == 8.0
    assert mmlongdoc._clean_numeric("GPT-4") is None
