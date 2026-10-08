"""Evolution driver for one (domain, arm) chain.

Eight rounds, one candidate per round. Each round:
  1. run the current config on this round's training batch (reusing any
     already-valid single result for the same unit key)
  2. collect failures with their real on-screen images and consistency checks
  3. ask the optimizer for one change at the attributed coordinate
  4. evaluate the candidate on the frozen gate set
  5. apply the arm's acceptance rule; on rejection continue from the currently
     accepted snapshot

Baseline runs no chain: it is evaluated once.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from ..adapters import registry
from ..consistency.grounded import (RequirementCache, extract_requirements,
                                    verify_grounding)
from ..evolution.gate import ArmResults, GateDecision, decide
from ..evolution.propose import (CandidateRejected, propose,
                                 allowed_coordinates, describe_change,
                                 enforce_arm_scope, propose_expression,
                                 propose_joint, propose_operator,
                                 select_coordinate)
from ..evolution.strategy_book import (StrategyBook, directive_block,
                                       record_outcome, reflect_update)
from ..jobs.ledger import Ledger
from ..models import CFG, ModelClient, ModelFormatError, ModelUnavailable
from ..splits_cap import cap_by_groups
from ..runtime import run_static_item
from ..schema import Sample, Trajectory
from ..views.config import Coordinates
from .gate import projection_eligible

# Every arm in this design uses C_ground; kept as a named set so an arm that
# should not read evidence verification can be added without touching the loop.
# VIEWCRAFT_SKIP_GROUNDING=1 (env) skips the two per-trajectory C_ground
# calls (extract_requirements + verify_grounding). Added 2026-09-19 for the
# LOCAL-MODEL self-evolution lines: those calls double the per-unit cost
# through the same serialized model, and on a 27B/4B local server that priced
# the whole protocol at <1 unit/min. The API lines keep the full audit;
# C_ground remains in the gate decision for every API-line run.
GROUNDING_BLIND_ARMS: set[str] = set()
if os.environ.get("VIEWCRAFT_SKIP_GROUNDING") == "1":
    GROUNDING_BLIND_ARMS = {"viewcraft"}

# Fields whose effect is on the PROMPT, not on the emitted pixels. A candidate
# that moves only these produces identical geometry by construction, so the
# pixel-identity guard cannot say anything about whether it is inert -- the
# same reasoning that has always exempted theta. `psi.view_contract` belongs
# here on its own declaration (config.py: "str: how the attached images are
# DESCRIBED"); every other psi field changes which views exist, how many
# survive, or in what order they are sent, all of which the guard can see.
PURE_TEXT_FIELDS: frozenset[str] = frozenset(
    {"theta", "psi.view_contract", "psi.answer_contract", "psi.protocol_contract"})


def _view_summary(v) -> str:
    """One-line description of an emitted view, for the optimizer's context."""
    if v.region is None:
        reg = "none"
    else:
        reg = (f"({v.region.x0:.3f},{v.region.y0:.3f},"
               f"{v.region.x1:.3f},{v.region.y1:.3f})")
    return (f"{v.kind} region={reg} out={v.out_w}x{v.out_h} "
            f"scale={v.scale:.3f}")


def _trajectory_from_dict(d: dict) -> Trajectory:
    """Rebuild a Trajectory from its persisted form.

    Only the fields the gate and the optimizer read are restored; the raw
    record on disk stays the source of truth for auditing.
    """
    from ..schema import ExecResult, GroundingResult, Region, ViewRecord

    views = []
    for v in d.get("views", []):
        reg = v.get("region")
        views.append(ViewRecord(
            view_id=v["view_id"], source_media_id=v["source_media_id"],
            kind=v["kind"], long_edge_px=v["long_edge_px"],
            out_w=v["out_w"], out_h=v["out_h"],
            src_w=v["src_w"], src_h=v["src_h"],
            parent_view_id=v.get("parent_view_id"),
            region=(Region(**reg) if isinstance(reg, dict) else None),
            page_index=v.get("page_index"), seconds=v.get("seconds"),
            scale=v.get("scale", 1.0), upscaled=v.get("upscaled", False),
            ui_state_version=v.get("ui_state_version"),
            actually_sent=v.get("actually_sent", False),
            request_id=v.get("request_id"),
            bytes_len=v.get("bytes_len", 0),
            created_by=v.get("created_by", "")))

    t = Trajectory(
        item_id=d["item_id"], domain=d["domain"], split=d["split"],
        arm=d["arm"], condition=d["condition"],
        config_hash=d["config_hash"], protocol_hash=d["protocol_hash"],
        environment_seed=d.get("environment_seed"),
        answer_text=d.get("answer_text", ""),
        score=d.get("score"), scored_by=d.get("scored_by", ""),
        views=views, request_ids=list(d.get("request_ids") or []),
        observation_steps=d.get("observation_steps", 0),
        n_actions=d.get("n_actions", 0),
        input_tokens=d.get("input_tokens", 0),
        output_tokens=d.get("output_tokens", 0),
        wall_time_s=d.get("wall_time_s", 0.0),
        status=d.get("status", "ok"), error=d.get("error", ""),
        notes=d.get("notes", ""))

    er = d.get("exec_result")
    if isinstance(er, dict):
        t.exec_result = ExecResult(er.get("passed", False),
                                   tuple(er.get("violations") or ()))
    g = d.get("grounding")
    if isinstance(g, dict) and g.get("per_requirement"):
        t.grounding = GroundingResult(
            per_requirement=g["per_requirement"], score=g["score"],
            unknown_rate=g.get("unknown_rate", 0.0),
            n_requirements=g.get("n_requirements",
                                 len(g["per_requirement"])),
            verifier_request_ids=tuple(g.get("verifier_request_ids") or ()))
    return t


@dataclass
class RoundRecord:
    round_index: int
    coordinate: str
    diff: str
    rationale: str
    train_score: float
    gate_current: float
    gate_candidate: float
    ground_current: float | None
    ground_candidate: float | None
    decision: dict
    accepted_config_hash: str
    submissions: int = 0
    gate_evaluations: int = 0

    def to_dict(self) -> dict:
        return self.__dict__


class Chain:
    def __init__(self, domain: str, arm: str, run_dir: Path,
                 samples: list[Sample], budget: dict, protocol_hash: str,
                 executor: ModelClient, optimizer: ModelClient,
                 ledger: Ledger, initial: Coordinates,
                 train_batch_size: int = 64, rounds: int = 8,
                 seed: int = 20260911, concurrency: int = 12):
        self.domain = domain
        self.spec = registry.get(domain)
        self.arm = arm
        self.run_dir = Path(run_dir)
        self.budget = budget
        self.protocol_hash = protocol_hash
        self.executor = executor
        self.optimizer = optimizer
        self.ledger = ledger
        self.coords = initial
        self.rounds = rounds
        self.train_batch_size = train_batch_size
        self.seed = seed
        self.concurrency = concurrency

        self.train = [s for s in samples if s.split == "train"]
        if not self.train:
            raise ValueError("Evolution requires a nonempty train split")
        # THE GATE IS SIZED IN QUESTIONS, NOT GROUPS.
        #
        # `configs/experiment.yaml` specifies splits in GROUPS (documents,
        # videos, scenes) so that no group spans two splits. But gate COST
        # scales with questions, and a video carries ~5 of them: a
        # "gate: 80" for VSI realised 466 questions and NExT-GQA 390. The
        # full plan came to 27.7 hours against a 20-hour budget, and the
        # discrepancy was invisible because both numbers are called "gate".
        #
        # Capping here keeps the group-level disjointness (the cap is
        # applied to whole groups, in group order) while making the cost
        # predictable. It is a budget decision applied identically to every
        # arm and condition, and a SMALLER gate makes eps = 1/n larger, so
        # it raises the acceptance bar rather than lowering it.
        gate_all = [s for s in samples if s.split == "gate"]
        # Per-domain override (2026-09-19, the videoholmes_v1 lesson): the
        # effective n for STRATEGY SELECTION is the GROUP count, not the
        # question count -- 97 questions over 15 films taught a strategy
        # that gained +0.041 on-gate and lost 0.083 on the test's unseen
        # films. When the split plan deliberately declares a larger gate in
        # groups for such a domain, the global 96-question cost cap must not
        # silently shrink it back to 15 groups. Declared in experiment.yaml
        # as gate_question_cap_by_domain.
        _cap = int(CFG["evolution"].get(
            "gate_question_cap_by_domain", {}).get(self.domain, 0) or 0) \
            or int(CFG["evolution"].get("gate_question_cap", 0) or 0)
        gate_all = cap_by_groups(
            gate_all, _cap,
            label=f"[{arm}] gate",
            # Frozen per RUN, not per arm: every arm must be gated on the
            # same questions, and a resumed round must reuse the same set a
            # previous process chose.
            freeze_path=self.run_dir / "configs" / "gate_capped_items.json")
        self.gate = gate_all
        if not self.gate:
            raise ValueError("Evolution requires a nonempty gate split")
        # Resolution facts are a property of the data, measured once and
        # shared with the optimizer so it cannot propose a change that is
        # arithmetically incapable of altering an emitted pixel.
        self.resolution_stats = self._measure_resolution(
            self.train + self.gate)
        self.req_cache = RequirementCache(
            self.run_dir / "cache" / "requirements")

        self.arm_dir = self.run_dir / "arms" / arm
        (self.arm_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        # RESUME, not restart. A process killed mid-chain and relaunched with
        # the same run id used to begin from round 0 with an empty history,
        # which OVERWROTE history.json and destroyed the record of every
        # completed round. Measured: ocrbench/viewcraft went from
        # "rounds=3 accepted=1" back to "rounds=1 accepted=0", while its
        # round03 snapshot still sat on disk - so the accepted candidate
        # existed but the evidence for it had been deleted, and the arm
        # silently re-searched from its initial config.
        self.history: list[RoundRecord] = self._load_history()
        resumed = self._load_accepted_snapshot()
        if resumed is not None:
            self.coords = resumed
        # units that never produced a valid result, so coverage gaps are
        # reported rather than silently averaged away
        self.incomplete: list[dict] = []
        # diffs the gate already refused, so a round is not spent re-proposing
        # an identical change
        self.rejected_diffs: list[str] = []
        # phi/psi FIELDS found to emit byte-identical pixels on this domain.
        # Accumulated as they are discovered and fed back to the optimizer, so
        # a dead field is proposed at most once per chain instead of being
        # re-offered until the round's attempts run out.
        self.dead_fields: list[str] = []
        # Diffs already refused in earlier rounds of this same chain, restored
        # on resume so the optimizer is not handed a change the gate has
        # already rejected.
        for h in self.history:
            if not h.decision.get("accepted") and h.diff:
                self.rejected_diffs.append(h.diff)
        # Strategy induction layer (optional; config `evolution.strategy_book`,
        # overridable per-process with VIEWCRAFT_STRATEGY_BOOK=1 so a canvas-only
        # and a canvas+induction run can be A/B'd without editing the frozen
        # config a running process already read).
        import os as _os
        _env = _os.getenv("VIEWCRAFT_STRATEGY_BOOK")
        self.use_book = (_env == "1") if _env is not None else bool(
            (CFG.get("evolution") or {}).get("strategy_book"))
        self.book = StrategyBook.load(self.arm_dir / "strategy_book.json") \
            if self.use_book else StrategyBook()

    def _load_history(self) -> list[RoundRecord]:
        """Prior rounds for this arm, so a resume appends instead of erasing."""
        p = self.arm_dir / "history.json"
        if not p.exists():
            return []
        try:
            raw = json.loads(p.read_text())
        except json.JSONDecodeError:
            return []
        out = []
        for d in raw:
            try:
                out.append(RoundRecord(**d))
            except TypeError:
                continue          # shape changed; keep what parses
        return out

    def _load_accepted_snapshot(self) -> Coordinates | None:
        """The arm's latest ACCEPTED state, so a resume continues from it.

        A roundNN snapshot is written only on acceptance, so the highest
        numbered one is the arm's real current position. Restarting from the
        initialization would discard accepted work and re-explore ground the
        arm had already covered.
        """
        snaps = self.arm_dir / "snapshots"
        if not snaps.is_dir():
            return None
        best = None
        for d in snaps.iterdir():
            if not d.name.startswith("round") or d.name == "round00":
                continue
            if not (d / "CONFIG_HASH").exists():
                continue
            try:
                n = int(d.name[5:])
            except ValueError:
                continue
            if best is None or n > best[0]:
                best = (n, d)
        if best is None:
            return None
        try:
            return Coordinates.read_snapshot(best[1])
        except Exception:                                    # noqa: BLE001
            return None

    @staticmethod
    def _measure_resolution(samples: list[Sample], cap: int = 200) -> dict:
        """Source long-edge distribution over this domain's own splits."""
        from PIL import Image
        longs = []
        for s in sorted(samples, key=lambda x: x.public.item_id)[:cap]:
            for m in s.public.media_ids[:1]:
                try:
                    with Image.open(m) as im:
                        longs.append(max(im.size))
                except Exception:                            # noqa: BLE001
                    continue
        if not longs:
            return {}
        longs.sort()
        def q(f):
            return longs[min(len(longs) - 1, int(f * (len(longs) - 1)))]
        return {"src_long_edge_median": q(0.5), "src_long_edge_p90": q(0.9),
                "src_long_edge_max": longs[-1], "n_measured": len(longs)}

    # -- execution helpers -----------------------------------------------

    def _score(self, traj: Trajectory, s: Sample) -> float:
        return self.spec.scorer(traj.answer_text, s.private, s.public)

    def _scorer_label(self, s: Sample) -> str:
        """The metric name to record on this trajectory.

        Single-metric domains get the spec name unchanged. A domain whose
        adapter records a per-item `metric` gets it appended, so two
        incommensurable metrics can never land in one aggregate.
        """
        metric = (s.private.meta or {}).get("metric")
        if metric:
            return f"{self.spec.scorer_name}:{metric}"
        return self.spec.scorer_name

    def _run_one(self, s: Sample, coords: Coordinates, split: str,
                 want_grounding: bool) -> Trajectory:
        """Run one sample once, honouring the ledger's one-result rule."""
        traj = run_static_item(
            s.public, coords, self.executor, self.budget, self.protocol_hash,
            arm=self.arm, condition="normal", split=split,
            answer_style=self.spec.answer_style)
        if traj.error or not traj.exec_result or not traj.exec_result.passed:
            raise RuntimeError(traj.error or "Execution contract failed")
        traj.score = self._score(traj, s)
        # A domain with a MIXED metric must not stamp one blended label on
        # every trajectory: VSI-Bench scores numeric types with MRA and
        # multiple-choice types with accuracy, and a single `scored_by` made
        # the two indistinguishable downstream, so the summary averaged an
        # MRA against an accuracy. The per-item metric the adapter already
        # recorded is appended, which keeps them in separate buckets.
        traj.scored_by = self._scorer_label(s)

        if want_grounding:
            try:
                reqs = extract_requirements(s.public, self.executor,
                                            self.req_cache)
                urls = self._sent_urls(traj, s, coords)
                traj.grounding = verify_grounding(traj, reqs, self.executor,
                                                  urls)
                traj.notes += f" | reqs={len(reqs)}"
            except (ModelFormatError, ValueError) as e:
                traj.notes += f" | grounding_error={str(e)[:120]}"
        return traj

    def _sent_urls(self, traj: Trajectory, s: Sample,
                   coords: Coordinates) -> list[str]:
        """Re-render exactly the views that were sent, for the verifier.

        Rendering is deterministic given the recorded region and size, so this
        reproduces the pixels the executor saw rather than approximating them.

        CRITICAL: model-authored operators must be replayed here too. Without
        that, the executor answered from an enhanced image while C_ground
        judged evidence support on the ORIGINAL -- so every authored operator
        was scored against pixels nobody ever saw, and the resulting C_ground
        drop was a property of the measurement rather than of the candidate.
        Measured on chartqa_ops3 r1: C_ground -0.0471 on a candidate whose
        task score was exactly level, which is what refused it.
        """
        from ..views.operators import safe_context
        from ..views.render import (RenderedView, apply_context_ops,
                                    apply_view_ops, render_crop,
                                    render_full)
        from ..views.exprs import render_pixel_expr
        from ..views.layout import eval_layout_expr

        view_ops = list(getattr(coords.phi, "view_ops", []) or [])
        ctx_ops = list(getattr(coords.psi, "context_ops", []) or [])
        pixel_expr = str(getattr(coords.phi, "pixel_expr", "") or "").strip()
        layout_expr = str(getattr(coords.psi, "layout_expr", "") or "").strip()
        n_media = len(s.public.media_ids or ())

        views = []
        skipped_composite = 0
        for rec in traj.sent_views:
            # A composite's source_media_id is a comma-joined provenance
            # list, not a path, so it cannot be re-rendered directly. It is
            # REBUILT below from its sources by replaying layout_expr;
            # skipping it outright left C_ground judging whatever non-
            # composite views happened to remain.
            if "," in (rec.source_media_id or ""):
                # A composite records its sources as a comma-joined list.
                # Re-render each source at this view's budget and let the
                # layout replay below recompose them, so the verifier ends
                # up with the same single sheet the executor saw. Skipping
                # (or refusing) here left C_ground with nothing to judge.
                for m in [x for x in rec.source_media_id.split(",") if x]:
                    try:
                        views.append(render_full(m, m, rec.long_edge_px))
                    except Exception:                        # noqa: BLE001
                        skipped_composite += 1
                continue
            # Re-render from the view's OWN source. Using media_ids[0] for
            # every view silently showed the verifier frame 1 of a video
            # while the executor had been looking at frame 6, so C_ground
            # scored evidence the executor never received.
            media = rec.source_media_id
            if rec.kind in ("full", "overview"):
                v = render_full(media, media, rec.long_edge_px)
            else:
                v = render_crop(media, media, rec.region, rec.long_edge_px,
                                margin=0.0,
                                allow_upscale=rec.upscaled)
            if view_ops:
                ctx = safe_context(
                    s.public, n_media, (v.record.out_w, v.record.out_h),
                    long_edge=int(coords.phi.overview_long_edge_px))
                try:
                    v = apply_view_ops(v, view_ops, ctx)
                except Exception:                            # noqa: BLE001
                    # A replay failure must not silently fall back to the
                    # unprocessed image, which is the bug this method fixes.
                    raise
            views.append(v)

        # MODEL-AUTHORED OPERATORS MUST BE REPLAYED TOO.
        #
        # Without this, the executor answered from an enhanced / composed
        # image while C_ground judged evidence support on the ORIGINAL
        # pixels, so every authored operator was scored against something
        # nobody ever saw. Measured on blink_ev4 round 2: a joint
        # layout+contract+skill candidate RAISED task 0.812 -> 0.833 and was
        # refused because C_ground "fell" by 0.5278 -- a drop produced
        # entirely by the verifier seeing four separate images where the
        # executor saw one composed grid.
        if pixel_expr and views:
            import io as _io
            from PIL import Image as _Image
            from ..views.render import _encode
            out = []
            for v in views:
                im = _Image.open(_io.BytesIO(v.png_bytes)).convert("RGB")
                ctx = safe_context(
                    s.public, n_media, (v.record.out_w, v.record.out_h),
                    long_edge=int(coords.phi.overview_long_edge_px))
                try:
                    im2 = render_pixel_expr(im, pixel_expr, ctx)
                except Exception:                            # noqa: BLE001
                    raise
                rec2 = v.record
                png = _encode(im2)
                rec2.out_w, rec2.out_h = im2.size
                rec2.bytes_len = len(png)
                out.append(RenderedView(rec2, png))
            views = out

        # CANVAS replay. The model's accumulated notes were burned into the
        # views it actually read; C_ground must judge THOSE pixels, not the
        # bare frame. Replaying without the canvas would repeat the ops3 bug
        # (verifier scoring pixels nobody saw). Notes are stored in source
        # coords, so they are mapped into EACH view's own frame here -- an
        # overview shows all of them, a crop only the ones inside it, which
        # is exactly what the executor saw.
        canvas_expr = str(getattr(coords.phi, "canvas_expr", "") or "").strip()
        notes_all = [n for n in (getattr(traj, "canvas_notes", []) or [])
                     if isinstance(n, dict)]
        if canvas_expr and notes_all and views:
            import io as _io
            from PIL import Image as _Image
            from ..views.render import _encode
            from ..views.exprs import render_canvas
            from ..runtime import _map_point_to_view
            by_src: dict[str, list] = {}
            for n in notes_all:
                by_src.setdefault(n.get("source_media_id", ""), []).append(n)
            out = []
            for v in views:
                ns = by_src.get(v.record.source_media_id)
                if ns and "," not in (v.record.source_media_id or ""):
                    local = []
                    for n in ns:
                        uv = _map_point_to_view((n["x"], n["y"]), v.record)
                        if uv is not None:
                            local.append({"x": uv[0], "y": uv[1],
                                          "text": n.get("text", "")})
                    if local:
                        im = _Image.open(_io.BytesIO(v.png_bytes)).convert("RGB")
                        ctx = safe_context(
                            s.public, n_media, (v.record.out_w, v.record.out_h),
                            long_edge=int(coords.phi.overview_long_edge_px))
                        im2 = render_canvas(im, canvas_expr, local, ctx)
                        rec2 = v.record
                        png = _encode(im2)
                        rec2.out_w, rec2.out_h = im2.size
                        rec2.bytes_len = len(png)
                        out.append(RenderedView(rec2, png))
                        continue
                out.append(v)
            views = out

        if layout_expr and views:
            import io as _io
            from PIL import Image as _Image
            from ..views.render import _encode
            from ..schema import Region as _Region, ViewRecord as _VR, \
                new_view_id as _nid
            imgs = [_Image.open(_io.BytesIO(v.png_bytes)).convert("RGB")
                    for v in views]
            ctx = safe_context(
                s.public, n_media,
                (views[0].record.out_w, views[0].record.out_h),
                long_edge=int(coords.phi.overview_long_edge_px))
            ctx["crop_views"] = [im for im, v in zip(imgs, views)
                                 if v.record.kind == "crop"]
            ctx["crop_regions"] = [
                [v.record.region.x0, v.record.region.y0,
                 v.record.region.x1, v.record.region.y1]
                for v in views if v.record.kind == "crop" and v.record.region]
            composed = eval_layout_expr(layout_expr, imgs, ctx)
            srcs = ",".join(sorted({v.record.source_media_id
                                    for v in views}))[:400]
            rebuilt = []
            for im in composed:
                png = _encode(im)
                rebuilt.append(RenderedView(_VR(
                    view_id=_nid("lay"), source_media_id=srcs,
                    kind=views[0].record.kind,
                    long_edge_px=max(im.size), out_w=im.size[0],
                    out_h=im.size[1], src_w=views[0].record.src_w,
                    src_h=views[0].record.src_h,
                    region=_Region(0.0, 0.0, 1.0, 1.0), scale=1.0,
                    bytes_len=len(png), created_by="psi.layout_expr"), png))
            views = rebuilt

        if ctx_ops and views:
            ctx = safe_context(
                s.public, n_media,
                (views[0].record.out_w, views[0].record.out_h),
                long_edge=int(coords.phi.overview_long_edge_px))
            views = apply_context_ops(views, ctx_ops, ctx)

        if not views and (skipped_composite or traj.sent_views):
            raise RuntimeError(
                f"C_ground replay produced no views for {s.public.item_id}: "
                f"{skipped_composite} composite view(s) could not be rebuilt. "
                "Scoring evidence support on an empty view set would report a "
                "collapse that is an instrument failure, not a candidate "
                "property.")

        return [v.data_url for v in views]

    def _run_set(self, samples: list[Sample], coords: Coordinates,
                 split: str, want_grounding: bool) -> list[tuple[Sample,
                                                                 Trajectory]]:
        """Ensure every sample has exactly one valid result for this config.

        Results already present in the ledger are LOADED, not skipped: an
        aggregate must cover the whole set, otherwise the current version and
        a candidate get compared on different subsets and the paired gate
        becomes meaningless.

        A unit claimed by ANOTHER arm is waited for rather than skipped. Since
        the unit key is the measurement and excludes the arm, four arms
        starting from the identical config legitimately want the same units,
        and whichever arm loses the race must still end up with the result.
        Skipping produced exactly the coverage loss this method exists to
        prevent ("gate set returned 47/51").
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        out: list[tuple[Sample, Trajectory]] = []
        todo: list[tuple[Sample, str]] = []

        for s in samples:
            key = Trajectory(
                item_id=s.public.item_id, domain=self.domain, split=split,
                arm=self.arm, condition="normal",
                config_hash=coords.config_hash,
                protocol_hash=self.protocol_hash).unit_key()
            existing = self._load_existing(key, s, coords, split)
            if existing is not None:
                out.append((s, existing))
                continue
            todo.append((s, key))

        contended: list[tuple[Sample, str]] = []
        contended_lock = __import__("threading").Lock()

        def claim_then_run(s, key):
            """Claim at the moment work STARTS, not when it is queued.

            Claiming the whole set up front and then feeding it through a
            pool of `concurrency` workers meant a unit could sit claimed for
            hours before executing: measured 2026-09-12, vsi/nextgqa units
            waited 50-75 minutes between claim and completion, and 158 of 750
            leases had already expired while their work was still queued. An
            expired lease is re-claimable, so a second worker could begin the
            SAME measurement - the one-valid-result-per-unit rule was being
            held by luck rather than by construction.
            """
            meta = dict(domain=self.domain, split=split,
                        item_id=s.public.item_id, arm=self.arm,
                        condition="normal",
                        config_hash=coords.config_hash,
                        protocol_hash=self.protocol_hash)
            if not self.ledger.claim(key, meta):
                with contended_lock:
                    contended.append((s, key))
                return None
            return self._run_one(s, coords, split, want_grounding)

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futs = {}
            for s, key in todo:
                futs[ex.submit(claim_then_run, s, key)] = (s, key)

            for fut in as_completed(futs):
                s, key = futs[fut]
                try:
                    traj = fut.result()
                    if traj is None:
                        continue      # claimed by a peer; awaited below
                except (ModelUnavailable, ModelFormatError) as e:
                    self.ledger.mark(key, "failed", str(e)[:300])
                    self.incomplete.append({"item_id": s.public.item_id,
                                            "split": split,
                                            "reason": str(e)[:200]})
                    continue
                except Exception as e:                       # noqa: BLE001
                    self.ledger.mark(key, "needs_review",
                                     f"{type(e).__name__}: {str(e)[:280]}")
                    self.incomplete.append({"item_id": s.public.item_id,
                                            "split": split,
                                            "reason": f"needs_review: {e}"[:200]})
                    continue
                path = self._persist(traj, coords)
                self.ledger.commit_done(key, traj.score, traj.status,
                                        traj.request_ids, str(path))
                out.append((s, traj))

        if contended:
            out.extend(self._await_contended(contended, coords, split,
                                             want_grounding))

        missing = len(samples) - len(out)
        if missing:
            # Surface coverage loss loudly. An aggregate over a silently
            # shrunken set is not comparable to the version it is gated
            # against, so the caller must see this rather than infer it.
            print(f"[{self.arm}] WARNING: {split} set returned "
                  f"{len(out)}/{len(samples)} results ({missing} missing)")
            self.ledger.audit("", f"{self.domain}/{self.arm}",
                              "incomplete_set",
                              {"split": split, "have": len(out),
                               "want": len(samples),
                               "config_hash": coords.config_hash})
        return out

    def _await_contended(self, contended: list[tuple[Sample, str]],
                         coords: Coordinates, split: str,
                         want_grounding: bool, timeout_s: float = 2400.0
                         ) -> list[tuple[Sample, Trajectory]]:
        """Wait for units another arm is computing, then load their results.

        Polls rather than blocking on a lock, because the owner is a
        different PROCESS. If an owner dies its lease expires and the unit is
        reopened, at which point this arm claims and runs it itself - so a
        crashed peer costs time, never coverage.
        """
        import time as _time

        out: list[tuple[Sample, Trajectory]] = []
        pending = list(contended)
        deadline = _time.monotonic() + timeout_s
        delay = 2.0

        while pending and _time.monotonic() < deadline:
            still: list[tuple[Sample, str]] = []
            for s, key in pending:
                got = self._load_existing(key, s, coords, split)
                if got is not None:
                    out.append((s, got))
                    continue
                rec = self.ledger.get(key)
                state = (rec or {}).get("state")
                expires = (rec or {}).get("lease_expires") or 0
                # Same invariant as Ledger.claim: a lease is only live if
                # it is unexpired AND its holder is running. Judging by
                # expiry alone made four chains wait ~5 hours on units
                # owned by processes that had already exited.
                stale = (state != "leased"
                         or expires < _time.time()
                         or not self.ledger._owner_alive((rec or {}).get(
                             "owner")))
                if stale:
                    # Owner gone or unit reopened: take it over ourselves.
                    meta = dict(domain=self.domain, split=split,
                                item_id=s.public.item_id, arm=self.arm,
                                condition="normal",
                                config_hash=coords.config_hash,
                                protocol_hash=self.protocol_hash)
                    if self.ledger.claim(key, meta):
                        try:
                            traj = self._run_one(s, coords, split,
                                                 want_grounding)
                        except Exception as e:               # noqa: BLE001
                            self.ledger.mark(key, "failed", str(e)[:300])
                            self.incomplete.append(
                                {"item_id": s.public.item_id, "split": split,
                                 "reason": f"takeover failed: {e}"[:200]})
                            continue
                        path = self._persist(traj, coords)
                        self.ledger.commit_done(key, traj.score, traj.status,
                                                traj.request_ids, str(path))
                        out.append((s, traj))
                        continue
                still.append((s, key))
            pending = still
            if pending:
                _time.sleep(delay)
                delay = min(delay * 1.5, 20.0)

        for s, key in pending:
            self.incomplete.append({"item_id": s.public.item_id,
                                    "split": split,
                                    "reason": "peer arm never published a "
                                              "result within the wait window"})
        return out

    def _load_existing(self, key: str, s: Sample, coords: Coordinates,
                       split: str) -> Trajectory | None:
        """Rehydrate a previously completed single result from disk.

        If the ledger says 'done' but the trajectory file is gone (deleted by
        hand, or lost with a scratch directory), the unit is reopened for
        recomputation rather than counted as missing: a 'done' row we cannot
        read would otherwise vanish from the aggregate and silently shrink the
        denominator, which is how a whole arm once reported task=0.000.
        """
        rec = self.ledger.get(key)
        if not rec or rec["state"] != "done":
            return None
        path = rec["result_path"]
        if path and Path(path).exists():
            try:
                t = _trajectory_from_dict(json.loads(Path(path).read_text()))
            except json.JSONDecodeError:
                t = None
            if t is not None:
                # The unit key is the MEASUREMENT, which does not include the
                # arm, so a result first produced by another arm at the same
                # config is the same measurement and is reused rather than
                # re-sampled. Relabelling the in-memory copy keeps this arm's
                # own aggregates and failure reports attributed to it, while
                # the file on disk stays the single original record.
                t.arm = self.arm
                return t
        self.ledger.mark(key, "pending",
                         "result file missing or unreadable; recomputing")
        return None

    def _persist(self, traj: Trajectory, coords: Coordinates) -> Path:
        d = (self.arm_dir / "trajectories" /
             coords.config_hash[:12] / traj.split)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{traj.item_id.replace('/', '__')}.json"
        p.write_text(json.dumps(traj.to_dict(), ensure_ascii=False,
                                indent=2, default=str))
        return p

    # -- aggregation -----------------------------------------------------

    def _aggregate(self, pairs: list[tuple[Sample, Trajectory]],
                   want_grounding: bool) -> ArmResults:
        # An EMPTY result set is MISSING DATA, not a score of zero.
        # `max(1, len(pairs))` silently turned "nothing ran" into
        # "task=0.000", and the gate then read that as a measured collapse.
        # Observed on vsi_pilot r2: a layout candidate whose units never
        # executed was recorded as 0.700 -> 0.000 and refused on that basis.
        # An aggregate over nothing is undefined; return an explicitly
        # incomparable result so the gate's unequal-set rule refuses it for
        # the right reason.
        if not pairs:
            return ArmResults(task_score=float("nan"), ground_score=None,
                              n_items=0, exec_pass=False, resource_ok=False,
                              per_item_scores={})

        scores = {s.public.item_id: (t.score or 0.0) for s, t in pairs}
        n = len(pairs)
        task = sum(scores.values()) / n

        ground = None
        unknown = 0.0
        if want_grounding:
            gs = [t.grounding for _, t in pairs if t.grounding]
            if len(gs) == n:
                ground = sum(g.score for g in gs) / len(gs)
                unknown = sum(g.unknown_rate for g in gs) / len(gs)

        exec_pass = all((t.exec_result and t.exec_result.passed)
                        for _, t in pairs) if pairs else False
        resource_ok = all(
            t.observation_steps <= int(self.budget["observations"])
            for _, t in pairs)

        return ArmResults(task_score=task, ground_score=ground, n_items=n,
                          exec_pass=exec_pass, resource_ok=resource_ok,
                          unknown_rate=unknown, per_item_scores=scores)

    def _failures(self, pairs: list[tuple[Sample, Trajectory]]) -> list[dict]:
        """Failure evidence for the optimizer. No gold answers included.

        Wholly-wrong items come first, but items that were CORRECT yet
        poorly grounded are included behind them. Reason: on a strong
        executor a 16-item batch can contain a single outright failure --
        measured, ocrbench and nextgqa each had exactly 1 -- and one sample
        is not enough to form a hypothesis about visual presentation. An
        item the model got right while C_ground says the evidence was only
        partially visible is a genuine presentation weakness and is
        informative for exactly the question the optimizer is answering.
        They are labelled, so the optimizer is never told a correct answer
        was wrong.
        """
        out = []
        weak = []
        for s, t in pairs:
            if (t.score or 0.0) >= 1.0:
                g = t.grounding
                if g is not None and g.score < 0.75:
                    weak.append((s, t))
                continue
            reqs = {}
            try:
                for r in extract_requirements(s.public, self.executor,
                                              self.req_cache):
                    reqs[r.req_id] = r.text
            except (ModelFormatError, ValueError):
                pass
            # SPATIAL DIAGNOSTICS, computed without any gold: for point-form
            # answers, does the reported point fall inside the region the
            # model ITSELF last zoomed into? "Wrong region" and "right region,
            # imprecise report" look identical in a bare wrong-answer feed
            # ("answered: 0.412, 0.867 (incorrect)") but demand opposite
            # fixes -- measured on screenspot_pro, where the optimizer could
            # not tell them apart and invented a coordinate-conversion bug
            # that does not exist (0.4845 -> 0.1056 when applied).
            spatial = {}
            try:
                from ..evaluation.scorers import parse_point
                pt = parse_point(t.answer_text or "")
            except Exception:                                # noqa: BLE001
                pt = None
            if pt is not None and max(pt) <= 1.0:
                crops = [v for v in t.sent_views
                         if v.kind == "crop" and v.region is not None]
                spatial["n_crops"] = len(crops)
                if crops:
                    last = crops[-1].region
                    spatial["answer_inside_own_last_crop"] = bool(
                        last.x0 <= pt[0] <= last.x1
                        and last.y0 <= pt[1] <= last.y1)
            # The images the executor ACTUALLY saw. Without these the
            # optimizer was designing visual operators from a text summary
            # ("crop out=1024x192") and had never seen a single failing
            # image -- which is why its proposals were generic legibility
            # boilerplate. It is a vision model; give it the pixels.
            shown = []
            try:
                shown = self._sent_urls(t, s, self.coords)[:3]
            except Exception as e:                           # noqa: BLE001
                # A silent [] here is how the optimizer ended up writing
                # "No failure images were provided" and falling back to
                # generic legibility advice. Record the cause so a missing
                # image is visible in the audit instead of looking like the
                # trajectory simply had none.
                shown = []
                self.ledger.audit("", f"{self.domain}/{self.arm}",
                                  "failure_images_unavailable",
                                  {"item_id": s.public.item_id,
                                   "error": f"{type(e).__name__}: {e}"[:200]})
            out.append({
                "outcome": "wrong",
                "images": shown,
                "category": (s.public.meta.get("category")
                             or s.public.meta.get("question_type")
                             or s.public.meta.get("subset")
                             or s.public.meta.get("type")),
                "question": s.public.question,
                "answer": t.answer_text[:120],
                "note": t.notes[:300],
                "observation_steps": t.observation_steps,
                "n_images_sent": t.n_images_sent,
                "view_summary": [_view_summary(v) for v in t.sent_views],
                "spatial": spatial,
                "requirements": reqs,
                "grounding": (None if not t.grounding else {
                    "score": t.grounding.score,
                    "per_requirement": t.grounding.per_requirement}),
            })

        for s, t in weak[: max(0, 8 - len(out))]:
            shown = []
            try:
                shown = self._sent_urls(t, s, self.coords)[:2]
            except Exception:                                # noqa: BLE001
                shown = []
            out.append({
                "outcome": "correct_but_poorly_grounded",
                "images": shown,
                "category": (s.public.meta.get("category")
                             or s.public.meta.get("question_type")
                             or s.public.meta.get("type")),
                "question": s.public.question,
                "answer": t.answer_text[:120],
                "note": t.notes[:300],
                "observation_steps": t.observation_steps,
                "n_images_sent": t.n_images_sent,
                "view_summary": [_view_summary(v) for v in t.sent_views],
                "requirements": {},
                "grounding": (None if not t.grounding else {
                    "score": t.grounding.score,
                    "per_requirement": t.grounding.per_requirement}),
            })
        return out

    def _assert_candidate_keeps_point_frames(self, cand) -> None:
        """Refuse a composite-producing candidate on a point-answer domain.

        layout_expr / context_ops REPLACE the view set with composite sheets,
        which have no single source transform. On a domain whose answer is a
        point, the model then has no addressable view left to point in: the
        answer falls back to full-frame interpretation of coordinates the
        model meant on the sheet, and every zoomed answer is displaced by the
        tiling offset. Measured on sspro_v2 viewcraft round 1: a joint
        proposal bundling a layout collapsed the gate 0.708 -> 0.000, after
        which the gate had to be paid to learn something knowable in advance.

        This is a STRUCTURAL fact about the answer style, not a measured
        score: same class as the dead-field hints, supplied so the round is
        re-spent on a candidate that can work.
        """
        if self.spec.answer_style != "point":
            return
        changed = self._changed_fields(cand)
        toxic = changed & {"psi.layout_expr", "psi.context_ops"}
        if toxic:
            raise CandidateRejected(
                f"{sorted(toxic)} replaces the view set with composite "
                "sheets, which have no source transform to resolve a point "
                "against; on a point-answer domain that displaces every "
                "answer. Propose a change that keeps the model's views "
                "addressable (pixels, crops, contracts, gates, selection).")

    def _assert_expressions_parse(self, cand) -> None:
        """Parse every expression ONCE, before the candidate costs a gate.

        A malformed expression is a property of the CANDIDATE, so it must
        be caught once here -- not re-discovered per item by the executor.
        Measured on dude_H3: `select_expr` used `minimum(...)`, which the
        selection language lacked, and 93 of 149 units died one by one
        with `status='error'`. The candidate scored 0.000 and the round
        read as a catastrophic strategy when it was a missing alias.

        Runtime still refuses a bad selection rather than falling back to
        uniform -- a silent fallback would score the default under the
        candidate's config hash. This check simply means runtime never
        sees one.
        """
        from ..views.exprs import eval_pixel_expr, pixel_env, render_canvas
        from ..views.layout import eval_layout_expr
        from ..views.operators import OperatorRejected
        from ..views.selection import eval_selection_expr
        from ..views.zoomgate import (eval_zoom_gate, mentions_a_zoom_fact,
                                      zoom_facts)
        from ..schema import Region
        from PIL import Image

        probe = Image.new("RGB", (64, 64), (120, 110, 100))
        probe_note = [{"x": 0.5, "y": 0.5, "text": "1"}]

        # Fact sets spanning the two populations section 11 measured: a source
        # at or below the overview budget (a zoom there cannot magnify) and one
        # well above it (a zoom recovers real detail), each with a small and a
        # large requested region.
        gate_probes = [
            zoom_facts(src_w=w, src_h=int(w * 0.75), overview_px=1024,
                       crop_px=1024, region=Region(0.1, 0.1, 0.1 + f, 0.1 + f),
                       step=s, steps_left=3 - s, crops_so_far=s)
            for w in (500, 1000, 2048, 4096)
            for f in (0.05, 0.5)
            for s in (0, 2)]

        def _check_gate(e: str) -> None:
            # A gate reading none of the request facts is a constant on every
            # item of a domain -- it either blocks all zooming or allows all of
            # it, and neither is the conditional mechanism this field is for.
            # Same rule as "a canvas_expr that ignores the note arrays is a
            # no-op and is rejected".
            if not mentions_a_zoom_fact(e):
                raise OperatorRejected(
                    "gate reads no fact about the requested region, so it is "
                    "constant on every item of this domain; write a condition "
                    "over e.g. zoom_gain or region_frac")
            verdicts = [eval_zoom_gate(e, f) for f in gate_probes]
            if not any(verdicts):
                raise OperatorRejected(
                    "gate refuses every probed request, including a small "
                    "region on a 4096 px source; that disables zooming rather "
                    "than conditioning it")
            # There is deliberately NO symmetric "allows every probe" check.
            # `src_long_edge > 400` allows all 16 synthetic probes and is still
            # live on a domain with 300 px sources, so refusing it here would be
            # the "the probe's nature fails the candidate" defect. Whether a
            # gate is inert on THIS data is decided by `compare_emitted_pixels`
            # against real items, which is the only place that can know.

        checks = (
            ("phi.select_expr",
             getattr(cand.coords.phi, "select_expr", ""),
             lambda e: eval_selection_expr(e, 32, {}, 8)),
            ("phi.pixel_expr",
             getattr(cand.coords.phi, "pixel_expr", ""),
             lambda e: eval_pixel_expr(e, pixel_env(probe, {}))),
            # canvas_expr is validated WITH a probe note present, since it is
            # only meaningful where a note exists; an empty-note env would make
            # a valid marking expression look like a no-op.
            ("phi.canvas_expr",
             getattr(cand.coords.phi, "canvas_expr", ""),
             lambda e: render_canvas(probe, e, probe_note, {})),
            # zoom_gate_expr is checked for MORE than parsing, because unlike
            # the others a gate cannot be judged inert by pixels alone: a gate
            # that refuses everything and one that refuses nothing both parse
            # and both emit a well-formed run.
            ("phi.zoom_gate_expr",
             getattr(cand.coords.phi, "zoom_gate_expr", ""),
             _check_gate),
            ("psi.layout_expr",
             getattr(cand.coords.psi, "layout_expr", ""),
             lambda e: eval_layout_expr(e, [probe, probe], {"n_media": 2})),
        )
        for name, expr, run in checks:
            expr = str(expr or "").strip()
            if not expr:
                continue
            try:
                run(expr)
            except OperatorRejected as e:
                rej = CandidateRejected(
                    f"{name} is not valid: {str(e)[:200]}")
                # Carry the text that failed, so the ledger can print it
                # rather than only the complaint about it.
                rej.exprs = {name.split(".")[-1]: expr[:600]}
                raise rej from e
            except Exception:                                # noqa: BLE001
                # A numeric failure on a synthetic 64x64 probe is not
                # evidence about real images; only refuse on an explicit
                # rejection, and let the real render decide the rest.
                pass

    def _assert_candidate_can_change_pixels(self, cand) -> None:
        """Refuse a phi change that provably emits identical pixels.

        Raising overview_long_edge_px above the source long edge is a no-op,
        because the whole-image view is never upscaled. Evaluating such a
        candidate costs a full gate pass and returns task and C_ground
        unchanged to the digit - the exact signature of "config changed but
        images did not" that this codebase treats as a bug rather than a
        result. Caught here so the round can be re-spent on a real change.
        """
        src = (self.resolution_stats or {}).get("src_long_edge_max")
        if not src or cand.coordinate != "phi":
            return
        before = int(self.coords.phi.overview_long_edge_px)
        after = int(cand.coords.phi.overview_long_edge_px)
        if after != before and min(after, before) >= src:
            raise CandidateRejected(
                f"overview_long_edge_px {before} -> {after} cannot change any "
                f"pixel: the largest source image in this domain is {src}px "
                "and the whole-image view is never upscaled")

    def _changed_fields(self, cand) -> set[str]:
        """Which coordinate fields the candidate actually moves.

        Computed from the configs, never parsed back out of `cand.diff`: the
        diff string is a human label chosen by the proposer path that built it
        and carries a hash suffix, so anything derived from it is a guess.
        """
        cur, new = self.coords, cand.coords
        out: set[str] = set()
        if cur.theta.hash != new.theta.hash:
            out.add("theta")
        for coord in ("phi", "psi"):
            a = getattr(cur, coord).to_dict()
            b = getattr(new, coord).to_dict()
            for k in set(a) | set(b):
                if a.get(k) != b.get(k):
                    out.add(f"{coord}.{k}")
        return out

    def _assert_candidate_changes_emitted_pixels(self, cand,
                                                 pairs) -> None:
        """Refuse a phi/psi candidate that would send byte-identical pixels.

        The arithmetic guard above catches only one inert case. This one is
        empirical and general: it replays each item's RECORDED view requests
        under both configs and compares the geometry that would be emitted.

        Measured before this existed: chartqa view_only reached round 7 with
        49/49 gate items identical across five config hashes, and ocrbench
        round 6 with 51/51 identical. Those gate passes cost full inference
        and searched nothing, while reporting a clean "no change" result that
        is indistinguishable from a real null.

        PURE-TEXT candidates are exempt, not just theta ones. A field that
        changes how the images are DESCRIBED emits identical geometry by
        construction, so an empty pixel diff says nothing about whether it is
        inert -- which is the whole reason theta was exempted in the first
        place. Keying the exemption on `cand.coordinate == "theta"` missed
        `psi.view_contract` (config.py: "str: how the attached images are
        DESCRIBED") and so refused joint proposals whose every changed field
        was text: measured across 8 runs (countbench_p1, dude_s5, hrbench_p1
        x3, mathvision_s2, muirbench_s5, nabirds_s5, nextgqa_s5 x2,
        vstar_p1), each refusal a round the arm never got to spend.
        """
        if not pairs:
            return
        changed = self._changed_fields(cand)
        if changed and changed <= PURE_TEXT_FIELDS:
            return
        from ..views.operators import OperatorRejected
        from .pixel_identity import compare_emitted_pixels
        try:
            diff = compare_emitted_pixels(self.coords, cand.coords, pairs)
        except OperatorRejected as e:
            # The candidate's expression is invalid on this domain's real
            # images -- e.g. the grid guard refusing a sheet that would
            # shrink tiles below legibility. That is a rejected CANDIDATE,
            # not a broken chain. Letting it propagate killed the arm after
            # one round: mmstar_full1 wrote "1 round of 8" and exited
            # cleanly, which reads as "evolution finished" when it crashed.
            self.ledger.audit("", f"{self.domain}/{self.arm}",
                              "candidate_invalid_on_real_images",
                              # Record the EXPRESSIONS, not just the diff
                              # label. Without them a parse failure is
                              # undebuggable: the ledger said "does not
                              # parse: unexpected indent (line 3)" and the
                              # text that failed was nowhere on disk, so
                              # each gap had to be guessed at and
                              # reproduced by hand.
                              {"diff": cand.diff, "error": str(e)[:300],
                               "pixel_expr": str(getattr(
                                   cand.coords.phi, "pixel_expr", ""))[:600],
                               "select_expr": str(getattr(
                                   cand.coords.phi, "select_expr", ""))[:300],
                               "layout_expr": str(getattr(
                                   cand.coords.psi, "layout_expr", ""))[:300],
                               "view_contract": str(getattr(
                                   cand.coords.psi,
                                   "view_contract", ""))[:200]})
            raise CandidateRejected(
                f"{cand.diff} is invalid on this domain's images: "
                f"{str(e)[:200]}") from e
        if diff.n_probed and not diff.changes_pixels:
            self.ledger.audit("", f"{self.domain}/{self.arm}",
                              "inert_candidate_refused",
                              {"diff": cand.diff, "detail": diff.detail,
                               "n_probed": diff.n_probed,
                               "changed_fields": sorted(changed)})
            # Record the FIELD, computed from the configs. Reverse-parsing
            # `cand.diff` appended whole diff labels with their hashes --
            # "theta + phi.zoom_gate_expr + psi.view_contract #7de1a110" --
            # and `dead_fields` is injected into the proposer's prompt as a
            # list of field NAMES, so the proposer was being told that a name
            # which does not exist is dead.
            #
            # Only an UNAMBIGUOUS attribution is recorded: if the candidate
            # moved two non-text fields at once, the pair is inert but which
            # one is dead is unknown, and naming both would retire a field on
            # no evidence. The refusal is audited either way.
            suspects = sorted(changed - PURE_TEXT_FIELDS)
            if len(suspects) == 1 and suspects[0] not in self.dead_fields:
                self.dead_fields.append(suspects[0])
            raise CandidateRejected(
                f"{cand.diff} emits byte-identical pixels on this domain "
                f"({diff.detail}); the change cannot alter any measured "
                "quantity, so the round is not spent on it")

    # -- the chain -------------------------------------------------------

    def _visual_projection_fallback(self, cand, cur_gate, want_ground):
        """Joint-arm fallback: evaluate the visual part of a REJECTED joint
        candidate on its own.

        A joint proposal bundles a visual change with a text change and is
        accepted or refused as a whole, so a GOOD visual move can be vetoed by
        a bad coupled theta (measured on single-lever domains, where one
        visual lever dominates and the coupled text drags it down).
        Reverting theta to the current value and re-gating the
        SAME phi/psi lets the framework keep the visual win the coupled text
        had dragged down.

        This is NOT the forbidden "rejected visual becomes an accepted text
        edit" path: it is the same pixels, just unbundled from the text that
        was hurting them -- the acceptance is still on measured gate
        improvement, never manufactured. viewcraft is then "co-evolve where
        the coordinates are coupled, evolve the visual coordinate alone where
        one lever dominates": joint-first, but never worse than a
        visual-only move in expectation. Returns (projection, proj_gate,
        proj_dec) only when the projection is ACCEPTED, else None.
        """
        # Revert theta to current; keep the candidate's phi/psi. Raises
        # CandidateRejected when the candidate had no visual change at all
        # (theta-only), in which case there is nothing to salvage.
        self._projection_evaluated = False
        if not (self._changed_fields(cand) - PURE_TEXT_FIELDS):
            return None
        try:
            proj = enforce_arm_scope("view_only", self.coords, cand.coords)
        except CandidateRejected:
            return None
        # theta did not change in the candidate, so the candidate WAS already
        # visual-only and was rejected -- re-gating the identical config would
        # just re-measure the same refused condition.
        if proj.config_hash == cand.coords.config_hash:
            return None
        proj_diff = describe_change(self.coords, proj)
        if proj_diff in self.rejected_diffs:
            return None
        self.ledger.audit("", f"{self.domain}/{self.arm}", "projection_submitted",
                          {"config_hash": proj.config_hash, "diff": proj_diff})
        self._projection_evaluated = True
        proj_gate_pairs = self._run_set(self.gate, proj, "gate", want_ground)
        proj_gate = self._aggregate(proj_gate_pairs, want_ground)
        if len(proj_gate_pairs) != len(self.gate) or proj_gate.ground_score is None:
            raise RuntimeError("Projection gate evaluation is incomplete. Check model errors and resume.")
        proj_dec = decide(self.arm, cur_gate, proj_gate)
        if not proj_dec.accepted:
            # remember the refusal so a later round is not spent re-testing it
            self.rejected_diffs.append(proj_diff)
            self.ledger.audit("", f"{self.domain}/{self.arm}",
                              "visual_projection_refused",
                              {"round_diff": cand.diff[:120],
                               "proj_diff": proj_diff[:160]})
            return None
        return proj, proj_gate, proj_dec

    def batches(self) -> list[list[Sample]]:
        """Fixed, shared batch plan: sampling without replacement per round."""
        rng = random.Random(f"{self.seed}:{self.domain}:batches")
        pool = sorted(self.train, key=lambda s: s.public.item_id)
        rng.shuffle(pool)
        size = min(self.train_batch_size, len(pool))
        out = []
        for r in range(self.rounds):
            if size >= len(pool):
                out.append(list(pool))       # not enough train items: use all
            else:
                start = (r * size) % len(pool)
                sel = pool[start:start + size]
                if len(sel) < size:
                    sel = sel + pool[: size - len(sel)]
                out.append(sel)
        return out

    def run(self) -> dict:
        want_ground = self.arm not in GROUNDING_BLIND_ARMS
        plan = self.batches()

        if not (self.arm_dir / "snapshots" / "round00").exists():
            self.coords.write_snapshot(self.arm_dir / "snapshots" / "round00")
        cur_gate_pairs = self._run_set(self.gate, self.coords, "gate",
                                       want_ground)
        cur_gate = self._aggregate(cur_gate_pairs, want_ground)
        if len(cur_gate_pairs) != len(self.gate) or cur_gate.ground_score is None:
            raise RuntimeError("Initial gate evaluation is incomplete. Check model errors and resume.")
        print(f"[{self.arm}] round 0 gate baseline: "
              f"task={cur_gate.task_score:.3f} "
              f"ground={cur_gate.ground_score}")

        done_rounds = {h.round_index for h in self.history}
        for r in range(self.rounds):
            if (r + 1) in done_rounds:
                # already executed and recorded in a previous process; its
                # decision is on disk and must not be re-litigated
                continue
            t0 = time.monotonic()
            batch = plan[r]
            train_pairs = self._run_set(batch, self.coords, "train",
                                        want_ground)
            if len(train_pairs) != len(batch):
                raise RuntimeError("Training evaluation is incomplete. Check model errors and resume.")
            train_agg = self._aggregate(train_pairs, want_ground)
            failures = self._failures(train_pairs)

            # STRATEGY INDUCTION (optional). Diagnose this round's failures by
            # visual cause and fold them into the persistent book; the book's
            # directive then steers this round's proposal at the highest-value
            # unresolved failure mode (and bars re-rolling past failures).
            directive = ""
            if self.use_book:
                try:
                    self.book = reflect_update(self.book, failures,
                                               self.optimizer, r + 1)
                    directive = directive_block(self.book)
                    self.book.save(self.arm_dir / "strategy_book.json")
                except Exception as e:                       # noqa: BLE001
                    # The book is advisory; a reflection failure must never
                    # crash a chain or lose a round.
                    self.ledger.audit("", f"{self.domain}/{self.arm}",
                                      "strategy_book_error",
                                      {"round": r + 1, "error": str(e)[:200]})

            coord = select_coordinate(failures, self.arm, r)
            # CO-EVOLUTION. theta, phi and psi are treated as ONE coupled
            # object, so the default move is a joint proposal rather than a
            # rotation over single coordinates. A restricted scope keeps the
            # single-coordinate path.
            joint = len(allowed_coordinates(self.arm)) == 3
            # An inert candidate must not consume the round. Before the guard
            # existed, the optimizer's inert proposals were EVALUATED and the
            # round was lost to a full gate pass that searched nothing. With
            # the guard, refusing and moving on would lose the round just as
            # surely - so the refusal is fed back as a rejected diff and the
            # optimizer is asked again, up to a small bounded number of tries.
            # Bounded, because a domain with no live field left should end its
            # chain honestly rather than loop forever.
            cand = None
            proposal = None
            submission_limit = int(CFG["evolution"].get("max_submissions_per_round", 4))
            gate_limit = int(CFG["evolution"].get("max_gate_evaluations_per_round", 2))
            submissions = 0
            gate_evaluations = 0
            for _attempt in range(submission_limit):
                submissions += 1
                try:
                    stats = dict(self.resolution_stats)
                    if stats:
                        budget = int(self.coords.phi.overview_long_edge_px)
                        stats["headroom_frac"] = (
                            1.0 if stats["src_long_edge_median"] > budget
                            else 0.0)
                    # Alternate between AUTHORING a visual operator and
                    # tuning a scalar. Operators are where the open-ended
                    # search lives -- the scalars are a bounded handful and
                    # were measured 39-56% inert per domain -- so an operator
                    # gets the first attempt of every phi/psi round, with
                    # scalar tuning as the fallback.
                    # A joint proposal that was refused for a SYNTAX reason
                    # must be retried as a joint proposal, with the parser's
                    # complaint fed back. Falling through to the
                    # single-coordinate path instead turns a rejected VISUAL
                    # candidate into a theta-only edit, and if that edit is
                    # then accepted the run has manufactured evidence that
                    # "only text evolution works" out of a punctuation
                    # error. Observed exactly once, on vstar_joint1 round 1:
                    # `keyword arguments are not supported` -> fallback ->
                    # theta 270->705 chars -> gate 0.938->1.000 ACCEPT.
                    if joint:
                        # The full arm's search unit is (theta, phi, psi) and
                        # EVERY attempt stays joint. Falling through to a
                        # single-coordinate path after a syntax refusal
                        # silently converts a rejected VISUAL proposal into
                        # an accepted TEXT edit -- that is how a parser gap
                        # manufactures evidence for text-only evolution.
                        # Observed three times: vstar_joint1 r1 (kwargs),
                        # blink_ev5/mmlongdoc_ev5 r1 (subscript), blink_ev7
                        # r1-2 (`label=none`). In each case the visual half
                        # of the round was lost to punctuation and the
                        # fallback theta edit became the round's result.
                        # The refusal text is fed back so the optimizer can
                        # fix its own syntax.
                        proposal = propose_joint(
                            self.arm, self.coords, failures, self.optimizer,
                            r, resolution_stats=stats,
                            rejected=self.rejected_diffs,
                            directive=directive,
                            answer_style=self.spec.answer_style)
                    elif coord in ("phi", "psi") and _attempt <= 2:
                        # First attempt of a visual round: let the optimizer
                        # WRITE an expression, with the failure images
                        # attached. This is the open-ended path.
                        proposal = propose_expression(
                            self.arm, self.coords, failures, self.optimizer,
                            r, resolution_stats=stats,
                            rejected=self.rejected_diffs,
                            answer_style=self.spec.answer_style)
                    elif coord in ("phi", "psi") and _attempt == 3:
                        proposal = propose_operator(
                            self.arm, self.coords, failures, self.optimizer,
                            r, resolution_stats=stats,
                            rejected=self.rejected_diffs,
                            answer_style=self.spec.answer_style)
                    else:
                        proposal = propose(self.arm, self.coords, failures,
                                           self.optimizer, r,
                                           coordinate=coord,
                                           rejected=self.rejected_diffs,
                                           resolution_stats=stats,
                                           dead_fields=self.dead_fields,
                                           answer_style=self.spec.answer_style)
                    self._assert_expressions_parse(proposal)
                    self._assert_candidate_keeps_point_frames(proposal)
                    self._assert_candidate_can_change_pixels(proposal)
                    self._assert_candidate_changes_emitted_pixels(
                        proposal, train_pairs)
                except CandidateRejected as e:
                    # Remember the refusal so the optimizer is not handed the
                    # same dead field again, then retry this round.
                    # Feed back the REASON, not just the label. Only the
                    # diff label was recorded, so an optimizer that made a
                    # syntax slip -- a miscounted paren in a six-level
                    # nested expression -- was told "already tried" and
                    # never learned what was wrong. It cannot fix an error
                    # it is not shown.
                    diff = getattr(proposal, "diff", "")
                    note = f"{diff} -> REFUSED: {str(e)[:220]}" if diff \
                        else f"REFUSED: {str(e)[:220]}"
                    if note not in self.rejected_diffs:
                        self.rejected_diffs.append(note)
                    proposal = None
                    print(f"[{self.arm}] round {r+1} attempt {_attempt+1}: "
                          f"refused ({str(e)[:110]})")
                    # Record the failing EXPRESSION TEXT, exactly as the
                    # `candidate_invalid_on_real_images` path already does.
                    # Without it chartqa's two round-1 refusals showed only
                    # "unknown name 'height'" and the expression that used
                    # `height` was nowhere on disk, so the gap had to be
                    # guessed at. A rejection the ledger cannot explain is a
                    # rejection nobody can fix. The text rides on the
                    # exception, so it is available however far the raise is
                    # from here.
                    entry = {"round": r + 1, "attempt": _attempt + 1,
                             "error": str(e)[:300]}
                    entry.update(getattr(e, "exprs", {}) or {})
                    self.ledger.audit("", f"{self.domain}/{self.arm}",
                                      "candidate_rejected", entry)
                    continue
                except (ModelFormatError, ModelUnavailable) as e:
                    print(f"[{self.arm}] round {r+1}: proposer unavailable "
                          f"({type(e).__name__}: {str(e)[:110]})")
                    self.ledger.audit("", f"{self.domain}/{self.arm}",
                                      "proposer_failed",
                                      {"round": r + 1, "error": str(e)[:300]})
                    raise
                cand = proposal
                break

            if cand is None:
                print(f"[{self.arm}] round {r+1}: no live candidate found in "
                      f"{submission_limit} attempts; leaving the config unchanged")
                self.ledger.audit("", f"{self.domain}/{self.arm}",
                                  "round_exhausted_no_live_candidate",
                                  {"round": r + 1,
                                   "rejected_so_far": self.rejected_diffs[-8:]})
                self.history.append(RoundRecord(
                    round_index=r + 1, coordinate="none", diff="no valid candidate",
                    rationale="All proposal attempts were rejected before gate evaluation",
                    train_score=train_agg.task_score,
                    gate_current=cur_gate.task_score, gate_candidate=cur_gate.task_score,
                    ground_current=cur_gate.ground_score, ground_candidate=cur_gate.ground_score,
                    decision={"accepted": False, "reason": "no valid candidate"},
                    accepted_config_hash=self.coords.config_hash,
                    submissions=submissions, gate_evaluations=0))
                (self.arm_dir / "history.json").write_text(json.dumps(
                    [h.to_dict() for h in self.history], indent=2, default=str))
                continue

            gate_evaluations += 1
            cand_gate_pairs = self._run_set(self.gate, cand.coords, "gate",
                                            want_ground)
            cand_gate = self._aggregate(cand_gate_pairs, want_ground)
            if len(cand_gate_pairs) != len(self.gate) or cand_gate.ground_score is None:
                raise RuntimeError("Candidate gate evaluation is incomplete. Check model errors and resume.")

            dec = decide(self.arm, cur_gate, cand_gate)

            # Write the gate's verdict back onto the strategy this candidate
            # targeted, so the book becomes an empirical bandit over this
            # domain's visual strategies.
            if self.use_book:
                try:
                    record_outcome(self.book, cand.diff, cand.rationale,
                                   dec.accepted, r + 1)
                    self.book.save(self.arm_dir / "strategy_book.json")
                except Exception:                            # noqa: BLE001
                    pass

            # VISUAL-PROJECTION FALLBACK (joint arm only). A rejected joint
            # candidate may carry a good visual change vetoed by a bad coupled
            # theta. Evaluate the SAME visual change with theta reverted; if
            # it clears the gate on its own, accept it. Joint-first, but never
            # worse than a visual-only move in expectation. See
            # _visual_projection_fallback.
            if (len(allowed_coordinates(self.arm)) == 3
                    and CFG["evolution"].get("viewcraft_visual_fallback", True)
                    and projection_eligible(
                        dec, submissions, submission_limit,
                        gate_evaluations, gate_limit)):
                _fb = self._visual_projection_fallback(cand, cur_gate,
                                                       want_ground)
                if self._projection_evaluated:
                    submissions += 1
                    gate_evaluations += 1
                if _fb is not None:
                    _proj, _proj_gate, _proj_dec = _fb
                    print(f"[{self.arm}] round {r+1} joint rejected but its "
                          f"VISUAL PROJECTION accepted: gate "
                          f"{cur_gate.task_score:.3f} -> "
                          f"{_proj_gate.task_score:.3f}")
                    from .propose import Candidate as _Cand
                    cand = _Cand(
                        coordinate="joint_visual_projection", coords=_proj,
                        rationale=(cand.rationale +
                                   "  [visual projection of a rejected joint "
                                   "candidate; theta reverted to current]"),
                        diff=describe_change(self.coords, _proj),
                        proposer_request_id=cand.proposer_request_id)
                    cand_gate = _proj_gate
                    dec = _proj_dec

            rec = RoundRecord(
                round_index=r + 1, coordinate=cand.coordinate,
                diff=cand.diff, rationale=cand.rationale,
                train_score=train_agg.task_score,
                gate_current=cur_gate.task_score,
                gate_candidate=cand_gate.task_score,
                ground_current=cur_gate.ground_score,
                ground_candidate=cand_gate.ground_score,
                decision=dec.to_dict(),
                accepted_config_hash=(cand.coords.config_hash
                                       if dec.accepted
                                       else self.coords.config_hash),
                submissions=submissions, gate_evaluations=gate_evaluations)
            self.history.append(rec)
            # Mirror the round decision into the ledger's audit table. The
            # audit only recorded REFUSALS, so when history.json was
            # overwritten by a resume there was no second copy of an ACCEPTED
            # round anywhere - the accepted snapshot existed on disk with no
            # record of why. An append-only audit row makes the round record
            # reconstructible even if the JSON is lost.
            self.ledger.audit("", f"{self.domain}/{self.arm}",
                              "round_decided", rec.to_dict())

            print(f"[{self.arm}] round {r+1} {cand.coordinate}: "
                  f"{cand.diff[:60]} | gate {cur_gate.task_score:.3f} -> "
                  f"{cand_gate.task_score:.3f} | "
                  f"{'ACCEPT' if dec.accepted else 'reject'} ({dec.reason}) "
                  f"[{time.monotonic()-t0:.0f}s]")

            if dec.accepted:
                self.coords = cand.coords
                cur_gate = cand_gate
                self.coords.write_snapshot(
                    self.arm_dir / "snapshots" / f"round{r+1:02d}")
            else:
                self.rejected_diffs.append(cand.diff)

            (self.arm_dir / "history.json").write_text(json.dumps(
                [h.to_dict() for h in self.history], indent=2, default=str))

        final = self.arm_dir / "snapshots" / "final"
        self.coords.write_snapshot(final)
        return {
            "arm": self.arm,
            "domain": self.domain,
            # Fields that could not change a pixel here. A null evolution
            # result is only interpretable alongside this list: it separates
            # "the method did not help" from "there was nothing to try".
            "dead_fields_on_this_domain": list(self.dead_fields),
            "incomplete_units": list(self.incomplete),
            "final_config_hash": self.coords.config_hash,
            "rounds": [h.to_dict() for h in self.history],
            "final_gate_task": cur_gate.task_score,
            "final_gate_ground": cur_gate.ground_score,
        }
