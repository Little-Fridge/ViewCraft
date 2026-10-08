"""Domain registry: the one place a domain's identity is declared.

Everything that used to be hardcoded to V*Bench lives here as data - the
loader, the scorer, the scorer's name, the answer shape the executor is asked
for, and the data root. The evolution chain, the runner and the analysis all
read this table, so adding a benchmark is a registry entry plus an adapter,
not a fork of the driver.

`answer_style` matters for validity, not just ergonomics: telling a free-form
domain like ChartQA to "give the option letter" made the executor invent an
A-D that the scorer then marked wrong, which looks exactly like a real error
and is not one.

`scorer_name` is recorded on every trajectory. It names the metric so an MRA
and an accuracy can never be silently averaged into one number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..paths import data_root

DATA = data_root()


@dataclass(frozen=True)
class DomainSpec:
    name: str
    data_root: Path
    answer_style: str            # "letter" | "short_text" | "yes_no" | "point"
    scorer_name: str
    loader: Callable
    scorer: Callable
    # A domain whose released annotation carries evidence LOCATIONS can run
    # the oracle / probe conditions; one without them cannot, and saying so
    # here stops a reference condition being faked from nothing.
    has_evidence_regions: bool = False
    media_kind: str = "image"    # "image" | "video_frames"
    # Extra keyword arguments the adapter needs. Kept here rather than in the
    # runner so the runner never has to know which domain it is running.
    loader_kwargs: dict = field(default_factory=dict)
    notes: str = ""

    def load(self, seed: int):
        """Load samples, passing only the kwargs this adapter accepts.

        Adapters were written independently and do not share a signature -
        the video ones take n_frames, the static ones do not, and one takes
        no seed at all. Filtering by the real signature means a mismatch is
        impossible rather than a TypeError at run start.

        Every returned sample is then stamped with THIS SPEC'S NAME. An
        adapter sets `domain` from its own module constant, which is wrong
        whenever two specs share one adapter: `blink_hard` reuses
        `adapters/blink.py`, whose `DOMAIN = "blink"`, so every trajectory of
        a blink_hard run recorded `domain="blink"`. Downstream that broke two
        things at once -- the figure looked up accepted rounds under
        "blink_hard", found none, and labelled three arms that had ALL
        accepted candidates as "still at start", while the viewcraft bar
        vanished entirely. The spec name is the identity the rest of the
        system keys on, so it is imposed here rather than trusted from below.
        """
        import inspect
        params = inspect.signature(self.loader).parameters
        kwargs = dict(self.loader_kwargs)
        if "seed" in params:
            kwargs["seed"] = seed
        kwargs = {k: v for k, v in kwargs.items() if k in params}
        samples = self.loader(self.data_root, **kwargs)
        for s in samples:
            if getattr(s.public, "domain", None) != self.name:
                object.__setattr__(s.public, "domain", self.name)
        return samples


def _vstar() -> DomainSpec:
    from ..adapters import vstar
    return DomainSpec(
        name="vstar", data_root=DATA / "vstar", answer_style="letter",
        scorer_name="vstar_accuracy_v1", loader=vstar.load,
        scorer=vstar.score, has_evidence_regions=True,
        notes="191 official questions; released bboxes enable oracle/probes")


def _screenspot_pro() -> DomainSpec:
    from ..adapters import screenspot_pro
    return DomainSpec(
        name="screenspot_pro", data_root=DATA / "screenspot_pro",
        answer_style="point", scorer_name="screenspot_point_in_box_v1",
        loader=screenspot_pro.load, scorer=screenspot_pro.score,
        has_evidence_regions=True,
        notes="GUI grounding, 1581 instructions over ~950 high-res "
              "screenshots. The one domain where zoom_gain > 1 on EVERY item: "
              "median target is 0.0355% of the screen and 24.5% of targets "
              "fall under a 10px long edge at a 1024px overview")


def _ocrbench() -> DomainSpec:
    from ..adapters import ocrbench
    return DomainSpec(
        name="ocrbench", data_root=DATA / "echo840_OCRBench",
        answer_style="short_text", scorer_name="ocrbench_containment_v1",
        loader=ocrbench.load, scorer=ocrbench.score,
        notes="VQA/KIE subsets only; pure crop-recognition subsets excluded "
              "because their images are already ~100px and hold no detail to "
              "recover")


def _mmstar() -> DomainSpec:
    from ..adapters import mmstar
    return DomainSpec(
        name="mmstar", data_root=DATA / "Lin-Chen_MMStar",
        answer_style="letter", scorer_name="mmstar_accuracy_v1",
        loader=mmstar.load, scorer=mmstar.score,
        notes="items were filtered by the authors to require the image, so a "
              "pixel change that moves accuracy moved something load-bearing")


def _chartqa() -> DomainSpec:
    from ..adapters import chartqa
    return DomainSpec(
        name="chartqa", data_root=DATA / "chartqa",
        answer_style="short_text", scorer_name="chartqa_relaxed_v1",
        loader=chartqa.load, scorer=chartqa.score,
        notes="official relaxed accuracy, 5% relative tolerance")


def _hallusionbench() -> DomainSpec:
    from ..adapters import hallusionbench
    return DomainSpec(
        name="hallusionbench", data_root=DATA / "hallusionbench",
        answer_style="yes_no", scorer_name="hallusionbench_yesno_v1",
        loader=hallusionbench.load, scorer=hallusionbench.score,
        notes="image subset only; the non_image split has no pixels and "
              "cannot exercise a view mechanism")


def _nextgqa() -> DomainSpec:
    from ..adapters import nextgqa
    return DomainSpec(
        name="nextgqa", data_root=DATA / "nextgqa", answer_style="letter",
        scorer_name="nextgqa_qa_accuracy_v1", loader=nextgqa.load,
        scorer=nextgqa.score, media_kind="video_frames",
        # The POOL is 32 frames; the SEND budget stays 8
        # (max_images_per_request). Those are different numbers and
        # conflating them is what made frame selection unsearchable: with a
        # pool of 8 and a budget of 8, "which frames" has exactly one
        # answer, so phi.select_expr could only ever reproduce the default.
        #
        # The earlier comment here said a larger N would mean frames are
        # "evicted before the model ever saw them". That is true of the
        # DEFAULT path and is exactly the point: selection chooses which 8
        # of 32 survive, and uniform(8) over 32 reproduces the old
        # behaviour exactly, so the control is preserved. The extra decode
        # cost is paid once and cached.
        loader_kwargs={"n_frames": 32},
        notes="plain QA accuracy, NOT the official Acc@GQA, which also "
              "requires temporal grounding IoU>=0.5")


def _vsi() -> DomainSpec:
    from ..adapters import vsi
    return DomainSpec(
        name="vsi", data_root=DATA / "vsibench", answer_style="short_text",
        scorer_name="vsi_mra_and_accuracy_v1", loader=vsi.load,
        scorer=vsi.score, media_kind="video_frames",
        # Pool 32, send budget 8. See the nextgqa note: the pool and the
        # per-request cap are different quantities, and holding the pool at
        # the cap is what left frame selection with a single legal answer.
        # The adapter's own default was already 32; it was narrowed to 8
        # here, which removed the choice rather than the cost.
        loader_kwargs={"n_frames": 32},
        notes="mixed metric: MRA on numeric types, accuracy on MC types; "
              "reported separately, never averaged into one headline")


def _mmlongdoc() -> DomainSpec:
    from ..adapters import mmlongdoc
    return DomainSpec(
        name="mmlongdoc", data_root=DATA / "mmlongbench",
        answer_style="short_text",
        scorer_name="mmlongdoc_generalized_accuracy_v1",
        loader=mmlongdoc.load, scorer=mmlongdoc.score,
        # Pages are stills, not sampled video frames: a page is addressable
        # and re-readable at full resolution, so crops recover real detail.
        media_kind="image",
        # Evidence PAGES are released and ride on PrivateItem, so post-hoc
        # "did it look at the right page" binning works. Evidence REGIONS
        # within a page are NOT released, so the oracle / probe conditions
        # that crop to a ground-truth box cannot run here; this stays False
        # rather than faking a box from a whole page.
        has_evidence_regions=False,
        # No loader_kwargs: the page pool is a property of each document
        # rather than a fixed N the runner picks, so there is nothing for the
        # registry to set. The pool cap (128 pages) and render size (2048 px)
        # are adapter constants, documented and justified there. Same
        # pool-vs-per-request-cap split the video domains now use: the pool
        # is wide so phi.select_expr has a real choice, and psi's 8-image cap
        # decides how many of it reach one request.
        notes="generalized accuracy over five answer formats; 22% of items "
              "are unanswerable and score ABSTENTION, so the headline mean "
              "is only meaningful with its answerable/unanswerable split. "
              "Deterministic answer extraction replaces the official LLM "
              "extractor, and Float uses the official round-to-2 comparison "
              "rather than scorers.mmlongbench_match's 1% relative "
              "tolerance (both declared in the adapter docstring)")


def _countbench() -> DomainSpec:
    from ..adapters import countbench
    return DomainSpec(
        name="countbench", data_root=DATA / "nielsr_countbench",
        answer_style="short_text", scorer_name="countbench_exact_count_v1",
        loader=countbench.load, scorer=countbench.score,
        notes="open-ended counting, exact integer match; the official "
              "CLIP-retrieval protocol needs an image-text similarity head "
              "and cannot run on a generative VLM, so the count is asked "
              "directly WITHOUT the 2-10 candidate set (strictly harder; "
              "chance 1/9). Counting stresses enumeration under a pixel "
              "budget, which no other domain here isolates")


def _syn_count() -> DomainSpec:
    from ..adapters import syn_count
    return DomainSpec(
        name="syn_count", data_root=DATA / "_synthetic" / "syn_count",
        answer_style="short_text", scorer_name="syn_count_exact_count_v1",
        loader=syn_count.load, scorer=syn_count.score,
        notes="SYNTHETIC hard counting (zero download, deterministic). Objects "
              "are plainly visible but numerous and similar, so the failure is "
              "enumeration working-memory, the regime the visual memory layer "
              "(mark-as-you-count) is built for. One image per item, so group "
              "== item and no image spans splits. N is dialled for the "
              "dose-response probe (gain vs memory load).")


def _syn_state() -> DomainSpec:
    from ..adapters import syn_state
    return DomainSpec(
        name="syn_state", data_root=DATA / "_synthetic" / "syn_state",
        answer_style="short_text", scorer_name="syn_state_exact_token_v1",
        loader=syn_state.load, scorer=syn_state.score,
        loader_kwargs={"n_slots": 10, "n_swaps": 30},
        notes="SYNTHETIC state-tracking (zero download, deterministic). NOT a "
              "vision task: K slots hold tokens, a text list of M swaps permutes "
              "them, the model reports the token in a target slot. The bottleneck "
              "is working memory, so it tests whether the visual canvas is a "
              "general external memory (jot the current token on each slot), "
              "not an image-task trick. Load (K x M) is dialled for dose-response.")


def _syn_math() -> DomainSpec:
    from ..adapters import syn_math
    return DomainSpec(
        name="syn_math", data_root=DATA / "_synthetic" / "syn_math",
        answer_style="short_text", scorer_name="syn_math_exact_cell_v1",
        loader=syn_math.load, scorer=syn_math.score,
        notes="SYNTHETIC math-as-visual-workspace (CL-Bench idea): a 2D "
              "recurrence/DP table whose natural solution is to WRITE cells on "
              "the grid. Non-visual task; tests whether the visual canvas is a "
              "general external memory for math, not an image trick. Zero download.")


def _babyvision() -> DomainSpec:
    from ..adapters import babyvision
    return DomainSpec(
        name="babyvision", data_root=DATA / "babyvision",
        answer_style="short_text", scorer_name="babyvision_norm_v1",
        loader=babyvision.load, scorer=babyvision.score,
        notes="visual tasks a 3-year-old solves but MLLMs fail (UnipatAI). "
              "Perception/working-memory failures, on-thesis. Only subtypes "
              "with >=15 items kept (tiny ones dropped per user instruction). "
              "Answers: grid coords / counts / choice, normalised by the scorer.")


def _clbench() -> DomainSpec:
    from ..adapters import clbench
    return DomainSpec(
        name="clbench", data_root=DATA / "clbench",
        answer_style="short_text", scorer_name="clbench_rubric_judge_v1",
        loader=clbench.load, scorer=clbench.score,
        notes="NON-visual long-context rule-application (tencent CL-bench, "
              "Rule System Application subset). Tests whether a visual "
              "workspace (blank reference sheet the model jots key rules onto) "
              "acts as external memory for a task that is not about images. "
              "Scored by an independent LLM judge against the release rubrics.")


def _arc() -> DomainSpec:
    from ..adapters import arc
    return DomainSpec(
        name="arc", data_root=DATA / "arc",
        answer_style="short_text", scorer_name="arc_exact_grid_v1",
        loader=arc.load, scorer=arc.score, loader_kwargs={"split": "training"},
        notes="ARC-AGI (fchollet). Genuinely hard for frontier models (real "
              "headroom), and the OUTPUT IS A GRID -- the model can sketch the "
              "transformation and draw the answer cell-by-cell on the canvas. "
              "The game / visual-workspace test. Exact grid match (official).")


def _muirbench() -> DomainSpec:
    from ..adapters import muirbench
    return DomainSpec(
        name="muirbench", data_root=DATA / "MUIRBENCH_MUIRBENCH",
        answer_style="letter", scorer_name="muirbench_accuracy_v1",
        loader=muirbench.load, scorer=muirbench.score,
        notes="MULTI-IMAGE reasoning (2-9 images/item, 12 task types incl. "
              "Difference Spotting / Counting / Attribute Similarity). The task "
              "type the suite lacked: relate N independent images -- composing, "
              "marking, set-of-mark, canvas tracking.")


def _scifibench() -> DomainSpec:
    from ..adapters import scifibench
    return DomainSpec(
        name="scifibench", data_root=DATA / "mm-eval_SciFIBench",
        answer_style="letter", scorer_name="scifibench_c2f_accuracy_v1",
        loader=scifibench.load, scorer=scifibench.score,
        notes="CROSS-FIGURE DISCRIMINATION: one arXiv caption against FIVE "
              "candidate figures from the same paper family (Caption2Figure "
              "half only, 1000 items). Every other perception domain here is "
              "'find the thing inside one picture' and is answered by zooming; "
              "this one is answered by COMPARING, under an image cap that will "
              "not hold five figures plus a crop. The option letter is "
              "POSITIONAL (option C is the third attached frame), so a psi "
              "that evicts or reorders a whole-image view shows up as a wrong "
              "answer, as in blink.")


def _hrbench() -> DomainSpec:
    from ..adapters import hrbench
    return DomainSpec(
        name="hrbench", data_root=DATA / "hrbench",
        answer_style="letter", scorer_name="hrbench_accuracy_v1",
        loader=hrbench.load, scorer=hrbench.score,
        notes="HIGH-RES fine perception (DreamMr HR-Bench 8K). Read illegible "
              "detail (numbers/text/counts) on high-res images -- zoom genuinely "
              "recovers it. The spatial-accumulation regime; distinct from video "
              "frame-selection and from grounding tasks (dropped ScreenSpot).")


def _mathvision() -> DomainSpec:
    from ..adapters import mathvision
    return DomainSpec(
        name="mathvision", data_root=DATA / "mathvision",
        answer_style="short_text", scorer_name="mathvision_norm_v1",
        loader=mathvision.load, scorer=mathvision.score,
        notes="math+visual reasoning (MathLLMs MathVision). Geometry subjects "
              "need DRAWING on the figure (auxiliary lines, marked values) -- "
              "the constructive-workspace regime. Hard (~46% baseline) with a "
              "level gradient. ViLoMem-inspired; distinct from saturated "
              "symbolic math.")


def _mmad() -> DomainSpec:
    from ..adapters import mmad
    return DomainSpec(
        name="mmad", data_root=DATA / "_dl" / "MMAD",
        answer_style="letter", scorer_name="mmad_accuracy_v1",
        loader=mmad.load, scorer=mmad.score,
        notes="MMAD (ICLR 2025) INDUSTRIAL ANOMALY DETECTION against a "
              "known-good reference unit. Seven official subtasks over four "
              "public datasets; every item attaches query + normal template. "
              "The bottleneck is CORRESPONDENCE and SCALE, not resolution: "
              "the paper's own result is that adding the reference image does "
              "not reliably help unless the comparison is organised, which is "
              "the phi/psi decision the single-image domains never pose. The "
              "defect mask ships with the release and is deliberately never "
              "attached -- it is the localisation ground truth.")


def _rsrcc() -> DomainSpec:
    from ..adapters import rsrcc
    return DomainSpec(
        name="rsrcc", data_root=DATA / "_dl" / "RSRCC",
        answer_style="letter", scorer_name="rsrcc_accuracy_v1",
        loader=rsrcc.load, scorer=rsrcc.score,
        notes="RSRCC (Google Research 2026) BEFORE/AFTER satellite change "
              "reading. Two aligned overhead captures of one place; the "
              "question names a sub-region and asks what changed there. The "
              "release is 512 px square and the overview budget is set to "
              "512, so render.py:_fit makes every crop return the pixels the "
              "whole-image view already showed: this is the one domain here "
              "with ARITHMETICALLY ZERO resolution headroom, added as a "
              "pre-registered test of whether the co-evolution gain tracks "
              "task headroom rather than pixels. What phi and psi still have "
              "to do is pair the two captures at matched scale and keep the "
              "pair co-resident. Positional hazard: view order carries "
              "before/after, so a psi that reorders inverts every change. The "
              "release's Yes/No half is DROPPED (1827/1891 answers are "
              "'Yes' -- constant-answer 0.966) and the 4-way options are "
              "permuted per item (release gold {A 751, B 996, C 86, D 2}, "
              "majority letter 0.543); both decisions were measured before "
              "any RSRCC inference. See adapters/rsrcc.py.")


def _spotdiff() -> DomainSpec:
    from ..adapters import spotdiff
    return DomainSpec(
        name="spotdiff", data_root=DATA / "spotdiff",
        answer_style="letter", scorer_name="spotdiff_accuracy_v1",
        loader=spotdiff.load, scorer=spotdiff.score,
        notes="CHANGE DETECTION between natural image pairs (Lancelot53). MCQ "
              "with 1 true + 3 cross-item distractor difference sentences, so "
              "caption priors do not answer. Compare-and-mark regime; distinct "
              "perception bottleneck from counting / video / multi-image.")


def _tallyqa() -> DomainSpec:
    from ..adapters import tallyqa
    return DomainSpec(
        name="tallyqa", data_root=DATA / "tallyqa",
        answer_style="short_text", scorer_name="tallyqa_exact_count_v1",
        loader=tallyqa.load, scorer=tallyqa.score,
        notes="TallyQA COMPLEX split: count RARE / occluded objects in real "
              "images (find-then-count under occlusion), distinct from "
              "syn_count mass-enumeration. Zoom-to-detail + mark regime.")


def _nabirds() -> DomainSpec:
    from ..adapters import nabirds
    return DomainSpec(
        name="nabirds", data_root=DATA / "nabirds",
        answer_style="letter", scorer_name="nabirds_accuracy_v1",
        loader=nabirds.load, scorer=nabirds.score,
        notes="NABirds fine-grained bird species (555 classes). Tiny "
              "discriminative detail -> zoom-into-part regime, distinct from "
              "counting / change-detection / video.")


def _funsd() -> DomainSpec:
    from ..adapters import funsd
    return DomainSpec(
        name="funsd", data_root=DATA / "funsd",
        answer_style="short_text", scorer_name="funsd_field_edit_similarity_v2",
        loader=funsd.load, scorer=funsd.score,
        notes="FUNSD dense-form field extraction: locate a field label in a "
              "noisy scanned form and read its value. Reading-order + fine-text "
              "regime; zoom into fields, track which are read.")


def _mmcomposition() -> DomainSpec:
    from ..adapters import mmcomposition
    return DomainSpec(
        name="mmcomposition", data_root=DATA / "mmcomposition",
        answer_style="letter", scorer_name="mmcomp_accuracy_v1",
        loader=mmcomposition.load, scorer=mmcomposition.score,
        notes="MMComposition validation split: single-image MCQ over 13 "
              "compositional capability classes x 4 difficulty levels. "
              "Multi-answer items scored as exact set match.")


def _reasonmap() -> DomainSpec:
    from ..adapters import reasonmap
    return DomainSpec(
        name="reasonmap", data_root=DATA / "reasonmap",
        answer_style="route", scorer_name="reasonmap_route_v1",
        loader=reasonmap.load, scorer=reasonmap.score,
        notes="ReasonMap transit-map route planning on VERY large maps (up to "
              "14k x 20k px): find station A, trace lines, transfer, reach B. "
              "Path-tracing regime -- the canvas's third strategy family "
              "after counting and colour isolation.")


def _docvqa2026() -> DomainSpec:
    from ..adapters import docvqa2026
    return DomainSpec(
        name="docvqa2026", data_root=DATA / "docvqa2026",
        answer_style="short_text", scorer_name="docvqa2026_judge_v1",
        loader=docvqa2026.load, scorer=docvqa2026.score,
        notes="ICDAR 2026 DocVQA (val split, public answers): multi-page "
              "documents over 8 domains incl. maps / comics / infographics / "
              "engineering drawings. Reasoning questions, LLM-judged "
              "(model_graded_fact).")


def _bv(subtype: str, style: str) -> DomainSpec:
    from ..adapters import bvsynth
    return DomainSpec(
        name=f"bv_{subtype}", data_root=DATA / "babyvision",
        answer_style=style, scorer_name=f"bv_{subtype}_acc_v1",
        loader=bvsynth.load, scorer=bvsynth.score,
        loader_kwargs={"subtype": subtype},
        notes="TRANSFER protocol: evolve on ~20 synthetic items (babygen, "
              "gold by construction), test the frozen config on the real "
              "BabyVision subset. Splits are forced by a pre-written plan "
              "(scripts/prep_bvsynth.py).")


def _screenspot_v2() -> DomainSpec:
    from ..adapters import screenspot_v2
    return DomainSpec(
        name="screenspot_v2", data_root=DATA / "screenspot_v2",
        answer_style="point", scorer_name="screenspot_v2_point_in_box_v1",
        loader=screenspot_v2.load, scorer=screenspot_v2.score,
        has_evidence_regions=True,
        notes="ScreenSpot v2 (960x540): the SIMPLER grounding sibling of "
              "ScreenSpot-Pro -- bigger targets, so the view channel rather "
              "than raw localisation precision is the constraint.")


def _bv_maze() -> DomainSpec:
    return _bv("maze", "letter")


def _bvgs(subtype: str) -> DomainSpec:
    """gpt-6-synthesised v2 transfer sets (~200 synthetic items/subtype).

    answer_style is short_text for every bvgs_ domain, including maze: the
    real Maze subtype is mixed (9 choice + 11 blank integer answers), and a
    letter-only instruction makes the executor invent A-D on blank items,
    which the scorer then marks wrong -- exactly the failure answer_style
    exists to prevent. The scorer still resolves a letter when the item is
    a choice item.
    """
    from ..adapters import bvgsynth
    return DomainSpec(
        name=f"bvgs_{subtype}", data_root=DATA / "babyvision",
        answer_style="short_text", scorer_name=f"bvgs_{subtype}_acc_v1",
        loader=bvgsynth.load, scorer=bvgsynth.score,
        loader_kwargs={"subtype": subtype},
        notes="TRANSFER protocol, v2 generators: ~200 synthetic items per "
              "subtype written by gpt-6-astra against real examples "
              "(scripts/bvgs_author.py), difficulty calibrated to raw "
              "executor 0.3-0.7 (scripts/bvgs_calibrate.py). Evolve on "
              "synthetic, test frozen config on the real BabyVision "
              "subtype. Splits forced by scripts/prep_bvgs.py.")


def _bv_maze_real() -> DomainSpec:
    return _bv("maze_real", "letter")


def _bv_count_clusters() -> DomainSpec:
    return _bv("count_clusters", "short_text")


def _bv_find_different() -> DomainSpec:
    return _bv("find_different", "short_text")


def _bv_connect_lines() -> DomainSpec:
    return _bv("connect_lines", "short_text")


def _bv_find_different_hard() -> DomainSpec:
    return _bv("find_different_hard", "short_text")


def _bv_find_different_hard2() -> DomainSpec:
    return _bv("find_different_hard2", "short_text")


def _blink() -> DomainSpec:
    from ..adapters import blink
    return DomainSpec(
        name="blink", data_root=DATA / "blink", answer_style="letter",
        scorer_name="blink_accuracy_v1", loader=blink.load,
        scorer=blink.score,
        notes="val split only (test answers are withheld for the "
              "leaderboard, so test-*.parquet has an empty answer column); "
              "1108 of 1901 items are multi-image and the questions refer to "
              "images POSITIONALLY, so this is the only domain here where a "
              "psi ordering bug shows up as a wrong answer rather than as a "
              "silent accuracy dip")


def _mpdocvqa() -> DomainSpec:
    from ..adapters import mpdocvqa
    return DomainSpec(
        name="mpdocvqa", data_root=DATA / "lmms-lab_MP-DocVQA",
        answer_style="short_text", scorer_name="mpdocvqa_anls_v1",
        loader=mpdocvqa.load, scorer=mpdocvqa.score,
        # Pages are released as images, not sampled from a video: a page is
        # addressable and re-readable at full resolution, so crops recover
        # real detail. Same media_kind as mmlongdoc, for the same reason.
        media_kind="image",
        # The released `answer_page_idx` names the evidence PAGE and rides
        # on PrivateItem for post-hoc binning. Evidence REGIONS within a page
        # are NOT released, so the oracle / probe conditions that crop to a
        # ground-truth box cannot run here; this stays False rather than
        # faking a box from a whole page.
        has_evidence_regions=False,
        # min_pages=2 drops the 31.5% of val questions that ship exactly one
        # page. Those items are single-image DocVQA: phi.select_expr has one
        # legal answer on them, so they cannot exercise the selection half of
        # the mechanism at all and would dilute the domain with what is
        # effectively a different benchmark. The filter is on the RELEASED
        # page count -- a property of the item, fixed before any score was
        # measured -- and the full official pool is recoverable with
        # min_pages=1.
        loader_kwargs={"min_pages": 2},
        notes="official DocVQA-family ANLS (threshold 0.5) over the full "
              "released answer list; val split only (test answers are "
              "withheld for the live RRC leaderboard). Pure single-hop "
              "retrieval over a median 6-page document: unlike mmlongdoc "
              "there is no unanswerable half, so a score change is a change "
              "in 'did the right page reach the model' with no abstention "
              "behaviour mixed in. Deterministic short-answer extraction "
              "replaces the official protocol's assumption of a "
              "pre-extracted span (declared in the adapter docstring)")


def _dude() -> DomainSpec:
    from ..adapters import dude
    return DomainSpec(
        name="dude", data_root=DATA / "dude",
        answer_style="short_text", scorer_name="dude_anls_v1",
        loader=dude.load, scorer=dude.score,
        # Pages are rasterised stills, not sampled video frames: a page is
        # addressable and re-readable at full resolution, so crops recover
        # real detail. Same reasoning as mmlongdoc and mpdocvqa.
        media_kind="image",
        # DUDE releases answer bounding boxes WITH a page number, so the
        # evidence PAGE is known and rides on PrivateItem for post-hoc
        # binning. The boxes are in raw page pixel coordinates rather than
        # the normalized Region the oracle conditions expect, and converting
        # them needs the per-page media box; until that conversion is
        # written and verified this stays False rather than feeding the
        # oracle a box in the wrong coordinate system.
        has_evidence_regions=False,
        # min_pages=2 drops the 214 single-page documents, which cannot
        # exercise selection at all (phi.select_expr has one legal answer)
        # and would dilute the domain with single-image DocVQA. Filter is on
        # the document's REAL page count, fixed before any score was read;
        # the full pool is recoverable with min_pages=1.
        loader_kwargs={"min_pages": 2},
        notes="official DocVQA-family ANLS (threshold 0.5) plus an "
              "ABSTENTION branch for the 13.2% of val questions released as "
              "not-answerable, so the headline mean is only meaningful with "
              "its answerable/unanswerable split. val only (test answers "
              "withheld for the live RRC leaderboard). Longest documents in "
              "the study (max 112 pages) and the only document domain where "
              "evidence for one question can span SEVERAL pages. "
              "Annotations are not in the HF repo -- they come from Zenodo "
              "record 7763635. Deterministic short-answer extraction "
              "replaces the official protocol's assumption of a "
              "pre-extracted span (declared in the adapter docstring)")


def _perceptiontest() -> DomainSpec:
    from ..adapters import perceptiontest
    return DomainSpec(
        name="perceptiontest",
        data_root=DATA,                 # the adapter resolves both roots
        answer_style="letter",
        scorer_name="perceptiontest_accuracy_v1",
        loader=perceptiontest.load, scorer=perceptiontest.score,
        media_kind="video_frames",
        # The release ships no temporal span for the evidence, so the
        # oracle / probe conditions cannot run. False rather than faking a
        # span from the whole clip.
        has_evidence_regions=False,
        loader_kwargs={"n_frames": 32},
        notes="Perception Test validation, multiple choice, scored by "
              "official accuracy. 4,892 of the 19,140 released questions "
              "resolve against the 792 clips MVBench's archive already put "
              "on disk, so this domain costs a 0.3 MB question file instead "
              "of the 110 GB video download. Two orthogonal released "
              "labels (area, reasoning) make it the cleanest place to test "
              "whether requirement groups want different sampling.")


# MVBENCH SPLIT BY WHAT THE QUESTION NEEDS FROM THE FRAMES.
#
# The 20 sub-tasks run from `action_prediction` at 0.957 baseline (1 wrong
# of 23) to `action_localization` at 0.200 (16 wrong of 20). Evolving ONE
# coordinate triple over all of them means the saturated tasks dilute every
# signal the search could get, and a candidate that helps counting can only
# be accepted if it survives tasks where nothing is left to win.
#
# These three domains are the same data under three task subsets, chosen by
# WHAT THE SAMPLING HAS TO DELIVER rather than by topic:
#
#   temporal  where in the clip did it happen / did a state change
#             -> needs coverage across the whole timeline, or ends paired
#   counting  how many times / how many objects
#             -> needs density; a missed instant is a missed count
#   causal    what caused what / which way is it moving
#             -> needs the decisive instants, not even coverage
#
# Verified before committing quota (per-task baseline from the existing
# mvbench run): each group has baseline 0.53-0.55 and 42-44 wrong items
# with >1 source unit. Both domains that produced a positive result in this
# study sit at 34-37 on that surface, and the domains that produced nothing
# sit at 0 (ChartQA, single image) or 10 (BLINK, MP-DocVQA). Selecting on
# the working surface rather than on "view headroom" is deliberate: headroom
# turned out to be ANTI-correlated with outcomes.
#
# Each carries its own scorer_name, because scorer_name is the ledger's
# metric identity and a subset score must never pool with the full-mix one.
MVBENCH_GROUPS = {
    "temporal": ("action_localization", "state_change", "action_sequence",
                 "unexpected_action"),
    "counting": ("action_count", "moving_count", "object_existence",
                 "moving_attribute"),
    "causal": ("counterfactual_inference", "object_interaction",
               "egocentric_navigation", "moving_direction"),
}


def _mvbench_group(group: str):
    def build() -> DomainSpec:
        from ..adapters import mvbench
        return DomainSpec(
            name=f"mvbench_{group}", data_root=DATA / "OpenGVLab_MVBench",
            answer_style="letter",
            scorer_name=f"mvbench_{group}_accuracy_v1",
            loader=mvbench.load, scorer=mvbench.score,
            media_kind="video_frames", has_evidence_regions=False,
            loader_kwargs={"n_frames": 32,
                           "tasks": MVBENCH_GROUPS[group]},
            notes=f"MVBench restricted to the {group} sub-tasks "
                  f"({', '.join(MVBENCH_GROUPS[group])}). Same data, same "
                  f"official accuracy, same 32-frame pool as `mvbench`; "
                  f"only the task subset differs.")
    return build


def _mvbench() -> DomainSpec:
    from ..adapters import mvbench
    return DomainSpec(
        name="mvbench", data_root=DATA / "OpenGVLab_MVBench",
        answer_style="letter", scorer_name="mvbench_accuracy_v1",
        loader=mvbench.load, scorer=mvbench.score,
        media_kind="video_frames",
        # MVBench releases no temporal span for the evidence on most tasks,
        # so the oracle / probe conditions cannot run. False rather than
        # faking a span from the whole clip.
        has_evidence_regions=False,
        # Pool 32, send budget 8 -- the same split nextgqa and vsi use, and
        # for the same reason: the pool and the per-request cap are
        # different quantities, and holding the pool at the cap is what
        # leaves frame selection a single legal answer. uniform(8) over 32
        # reproduces the naive behaviour exactly, so the control survives.
        loader_kwargs={"n_frames": 32},
        notes="official MVBench accuracy over the released candidates "
              "(the benchmark is 100% multiple choice). 20 temporal "
              "sub-tasks x 200 questions; questions are about change OVER "
              "TIME, so unlike VSI's static walkthroughs a bad frame choice "
              "is usually unrecoverable. CAVEAT, stated in the adapter "
              "docstring: clips are SHORT (5-35 s), so the 32-frame pool is "
              "an adapter choice rather than an hour of footage -- the "
              "genuine long-video sets were gated (LongVideoBench) or too "
              "large to land overnight (Video-MME 101 GB, MLVU 430 GB). "
              "DEVIATION: the official fixed 'Best option: (' prompt is not "
              "used, because the prompt IS the theta coordinate being "
              "evolved; replies resolve via match_multiple_choice instead")


def _charxiv() -> DomainSpec:
    from ..adapters import charxiv
    return DomainSpec(
        name="charxiv", data_root=DATA / "princeton-nlp_CharXiv",
        answer_style="short_text", scorer_name="charxiv_reasoning_match_v1",
        loader=charxiv.load, scorer=charxiv.score,
        # CharXiv releases no bounding box for the evidence within a figure,
        # so the oracle / probe conditions that crop to a ground-truth box
        # cannot run. False rather than faking a box from the whole figure.
        has_evidence_regions=False,
        notes="reasoning split only (1,000 val figures, one question each); "
              "the descriptive half is excluded because its questions ship "
              "as template ids rather than text and the paper reports the "
              "two halves separately. SINGLE IMAGE per question, so this "
              "domain exercises only the RENDERING coordinate -- selection "
              "and composition have one legal answer -- and it is here for "
              "published difficulty (~0.4 SOTA), not for mechanism surface. "
              "DEVIATION: the official protocol uses GPT-4o as a judge; the "
              "comparison here is deterministic and therefore STRICTER, so "
              "numbers are a lower bound on the published metric and are "
              "comparable across arms but not to the leaderboard")


def _realworldqa() -> DomainSpec:
    from ..adapters import realworldqa
    return DomainSpec(
        name="realworldqa", data_root=DATA / "lmms-lab_RealWorldQA",
        # Mixed format: 435 multiple-choice and 327 free-form items. The
        # SCORER branches per item on the release's own instruction line, but
        # the executor gets ONE answer style, so short_text is used --- asking
        # for an option letter on a free-form item made the executor invent an
        # A-D that the scorer then marked wrong, which is exactly the failure
        # this field exists to prevent. match_multiple_choice still resolves a
        # letter when the model gives one, so the MC half is unharmed.
        answer_style="short_text", scorer_name="realworldqa_accuracy_v1",
        loader=realworldqa.load, scorer=realworldqa.score,
        notes="mixed MC/free-form, scored per item by the released format; "
              "the free-form half uses EXACT match rather than containment "
              "because the gold vocabulary is yes/no plus bare digits, where "
              "a verbose reply would contain several contradictory golds")


def _blink_hard() -> DomainSpec:
    from ..adapters import blink
    return DomainSpec(
        name="blink_hard", data_root=DATA / "blink", answer_style="letter",
        # Same metric DEFINITION as `blink` (exact multiple choice over the
        # released choices, identical function), but a distinct name because
        # scorer_name is the ledger's metric identity and the two domains
        # share a data_root. A shared name would let a full-mix score and a
        # hard-subset score collide in the ledger or be averaged into one
        # headline, which is exactly what this field exists to prevent.
        scorer_name="blink_hard_accuracy_v1", loader=blink.load,
        scorer=blink.score,
        # The six sub-tasks BLINK's own paper reports as hardest for frontier
        # models (GPT-4V 28-55% on these, vs 70-95% on the rest). The full
        # 14-task mix measured raw 0.72 here, which is saturated for our
        # purpose; the difficulty is concentrated in these six, and the other
        # eight contribute mostly-solved items that dilute any effect.
        #
        # SELECTION RATIONALE, and why this is not cherry-picking: the six are
        # taken from the PUBLISHED per-task difficulty in the BLINK paper,
        # NOT from our own measurement. A 50-item stratified probe gives only
        # 3-4 items per sub-task, so selecting on that probe's own ranking
        # would be fitting the split to sampling noise and the resulting
        # baseline would be biased low by construction. Choosing on external,
        # pre-existing numbers keeps the subset independent of anything we
        # measured, and the baseline reported for it is then a real
        # out-of-sample reading.
        loader_kwargs={"tasks": ("IQ_Test", "Multi-view_Reasoning",
                                 "Functional_Correspondence",
                                 "Semantic_Correspondence",
                                 "Relative_Reflectance", "Art_Style")},
        notes="the 6 sub-tasks published as hardest (793 val items); the full "
              "14-task blink domain measured raw 0.72, which is too easy to "
              "leave an optimizer anything to repair. Subset chosen on the "
              "paper's numbers, not on ours, so the baseline stays "
              "out-of-sample")


def _videoholmes() -> DomainSpec:
    from ..adapters import videoholmes
    return DomainSpec(
        name="videoholmes", data_root=DATA / "videoholmes",
        answer_style="letter",
        scorer_name="videoholmes_accuracy_v1",
        loader=videoholmes.load, scorer=videoholmes.score,
        media_kind="video_frames", has_evidence_regions=False,
        loader_kwargs={"n_frames": 32},
        notes="270 suspense shorts (1-5 min) / 1837 MCQ, 7 question types; "
              "multi-clue cumulative reasoning -- clues span the whole film, "
              "so frame selection under the 8-image cap is load-bearing. "
              "Official plain MCQ accuracy via match_multiple_choice; no "
              "evidence spans released, oracle/probes cannot run")


def _openeqa() -> DomainSpec:
    from ..adapters import openeqa
    return DomainSpec(
        name="openeqa", data_root=DATA / "openeqa",
        answer_style="short_text", scorer_name="openeqa_llm_match_v1",
        loader=openeqa.load, scorer=openeqa.score,
        media_kind="video_frames",
        loader_kwargs={"n_frames": 32},
        notes="HM-EQA only (ScanNet half needs TOS application). Official "
              "LLM-Match: free-form answers judged correct/incorrect by the "
              "OPTIMIZER as independent judge (judge != executor). "
              "First-person episodic memory -- the canvas map-marking home "
              "ground. Pool 32 / send 8, as nextgqa/vsi.")


def _slidevqa() -> DomainSpec:
    from ..adapters import slidevqa
    return DomainSpec(
        name="slidevqa", data_root=DATA / "slidevqa",
        answer_style="short_text",
        scorer_name="slidevqa_anls_v1",
        loader=slidevqa.load, scorer=slidevqa.score,
        media_kind="image",
        has_evidence_regions=False,
        notes="1652 val questions over 300 decks, uniform 20 pages; 13% need "
              "2 evidence pages -- the one document domain measuring selection "
              "recall. Pages joined from slidevqa-corpus parquet on "
              "document_id. Deterministic extract_span replaces the official "
              "GPT judge")


def _mmiu() -> DomainSpec:
    from ..adapters import mmiu
    return DomainSpec(
        name="mmiu", data_root=DATA / "mmiu", answer_style="letter",
        scorer_name="mmiu_accuracy_v1", loader=mmiu.load,
        scorer=mmiu.score, media_kind="image",
        has_evidence_regions=False,
        notes="MULTI-IMAGE correspondence/difference detection (2-31 images/"
              "item, 60 subtasks; topk selection ~inert = composition/canvas "
              "control). 3 of 7 archives on disk (4125/11698 questions); "
              "missing subdomains skipped with counts.")


def _stibench() -> DomainSpec:
    from ..adapters import stibench
    return DomainSpec(
        name="stibench", data_root=DATA / "stibench",
        answer_style="letter", scorer_name="stibench_accuracy_v1",
        loader=stibench.load, scorer=stibench.score,
        media_kind="video_frames", has_evidence_regions=False,
        loader_kwargs={"n_frames": 32},
        notes="official STI-Bench MCQ accuracy; 8 task types split "
              "static vs dynamic understanding (meta['task_type']); "
              "whole-video uniform pool per official protocol. Spatial-"
              "temporal accumulation -- the canvas mark-the-path home ground.")


_BUILDERS = {
    "perceptiontest": _perceptiontest,
    # MVBench by requirement group -- same data and metric as `mvbench`,
    # three task subsets. Added last because the 20-task mix hid the
    # signal: baselines inside it run 0.200 to 0.957.
    "mvbench_temporal": _mvbench_group("temporal"),
    "mvbench_counting": _mvbench_group("counting"),
    "mvbench_causal": _mvbench_group("causal"),
    "vstar": _vstar,
    "screenspot_pro": _screenspot_pro,
    "ocrbench": _ocrbench,
    "mmstar": _mmstar,
    "chartqa": _chartqa,
    "hallusionbench": _hallusionbench,
    "nextgqa": _nextgqa,
    "vsi": _vsi,
    "mmlongdoc": _mmlongdoc,
    # Third wave, 2026-09-12: added because the existing image domains are
    # saturated (raw 0.77-0.91, only 4-23 wrong items per 100), which leaves
    # an optimizer nothing to repair. These three were selected for measured
    # difficulty, not variety.
    "countbench": _countbench,
    "blink": _blink,
    "blink_hard": _blink_hard,
    "realworldqa": _realworldqa,
    # Fourth wave, 2026-09-13: added because the study has only 4 usable
    # domains and needs 7, and because the surviving gap is MULTI-UNIT
    # sources specifically. The mechanism evolves which units are shown, how
    # each is rendered, and how they are composed; a single-image domain can
    # only exercise the middle one, and the visual half measured ~nothing
    # there. These are selected for many addressable units per question
    # (mpdocvqa: median 6 pages, up to 20) and for published difficulty,
    # with raw baselines MEASURED before any is used for evolution.
    "mpdocvqa": _mpdocvqa,
    "dude": _dude,
    "mvbench": _mvbench,
    # Single-image, added for published DIFFICULTY rather than for mechanism
    # surface: CharXiv reasoning is ~0.4 for frontier models where every
    # other image domain here is 0.76-0.96. It can only exercise the
    # rendering coordinate, which is stated in its adapter docstring so a
    # flat result is not misread as the mechanism failing.
    "charxiv": _charxiv,
    # Synthetic domains (zero download) for the visual-memory study. The
    # canonical canvas task: hard counting where the bottleneck is working
    # memory, not visibility.
    "syn_count": _syn_count,
    "syn_state": _syn_state,
    "syn_math": _syn_math,
    "babyvision": _babyvision,
    "clbench": _clbench,
    "arc": _arc,
    "muirbench": _muirbench,
    "scifibench": _scifibench,
    "hrbench": _hrbench,
    "mathvision": _mathvision,
    "mmad": _mmad,
    "rsrcc": _rsrcc,
    "spotdiff": _spotdiff,
    "tallyqa": _tallyqa,
    "nabirds": _nabirds,
    "funsd": _funsd,
    "mmcomposition": _mmcomposition,
    "reasonmap": _reasonmap,
    "docvqa2026": _docvqa2026,
    "bv_maze": _bv_maze,
    "bv_count_clusters": _bv_count_clusters,
    "bv_find_different": _bv_find_different,
    "bv_connect_lines": _bv_connect_lines,
    "bv_find_different_hard": _bv_find_different_hard,
    "bv_find_different_hard2": _bv_find_different_hard2,
    "screenspot_v2": _screenspot_v2,
    "bv_maze_real": _bv_maze_real,
    # gpt-6-synthesised v2 transfer sets; bv_* entries above are untouched
    # and keep pointing at the original 24-item sets.
    "bvgs_maze": lambda: _bvgs("maze"),
    "bvgs_count_clusters": lambda: _bvgs("count_clusters"),
    "bvgs_find_different": lambda: _bvgs("find_different"),
    "bvgs_connect_lines": lambda: _bvgs("connect_lines"),
    "bvgs_find_shadow": lambda: _bvgs("find_shadow"),
    "videoholmes": _videoholmes,
    "openeqa": _openeqa,
    "stibench": _stibench,
    "mmiu": _mmiu,
    "slidevqa": _slidevqa,
}


def get(domain: str) -> DomainSpec:
    """Resolve a domain spec, importing its adapter lazily.

    Lazy import is deliberate: a domain whose adapter is still being written,
    or whose data failed to download, must not break the runner for every
    other domain.
    """
    if domain not in _BUILDERS:
        raise KeyError(f"unknown domain {domain!r}; "
                       f"known: {sorted(_BUILDERS)}")
    return _BUILDERS[domain]()


def available() -> list[str]:
    """Domains whose adapter imports and whose data root exists."""
    out = []
    for name in sorted(_BUILDERS):
        try:
            spec = get(name)
        except Exception:                                   # noqa: BLE001
            continue
        if spec.data_root.exists():
            out.append(name)
    return out


def register(spec: DomainSpec) -> None:
    """Register a user task for this process; built-in names cannot be replaced."""
    if spec.name in _BUILDERS:
        raise ValueError(f"Domain already registered: {spec.name}")
    _BUILDERS[spec.name] = lambda: spec


def all_names() -> list[str]:
    return sorted(_BUILDERS)
