"""ScreenSpot-Pro adapter: GUI grounding on high-resolution professional software.

THE TASK. One screenshot of a professional application (CAD, Premiere, VS Code,
Origin, ...) plus one natural-language instruction naming a control ("adjust
bottom crop for the selected track"). The answer is a CLICK POINT, scored by
the benchmark's own point-in-box metric: correct iff the predicted point falls
inside the ground-truth bounding box. No IoU, no distance tolerance, no partial
credit.

WHY THIS DOMAIN IS WORTH THE QUOTA, and it is a different reason from every
other domain in this corpus.

  Everywhere else, zooming is mostly futile. §11 measured it: over 1808
  recorded crops the median magnification is 1.000x and 73.5% deliver <=1.05x,
  because `render.py:_fit` never upscales and most sources already fit the
  overview budget. That is why `zoom_gate_expr` exists.

  ScreenSpot-Pro is the opposite regime, and arithmetically so. Measured on all
  1581 rows before any inference:

    * screens are large: 2560x1440 (514 rows), 3840x2160 (270), 3840x1080
      (181), 3456x2234 (161), ... every row exceeds a 1024 px overview budget,
      so `zoom_gain > 1` on every item rather than on a minority
    * targets are tiny: median box area is 0.0355% of the screen, p10 is
      0.0072%, median long edge 63 px
    * consequently 388/1581 = 24.5% of targets have a long edge under 10 px
      in a 1024 px overview -- not small, but unresolvable

  So this is the first domain in the corpus where the view budget, not the
  reasoning, is the binding constraint on a majority of items, and where a crop
  returns genuinely new pixels rather than the same pixels one round later.
  It tests the mechanism's core premise where that premise is live.

EXTERNAL CORROBORATION OF THE ACTION SPACE, not of our numbers. GUI-Lens
(arXiv 2608.03270v1, Fu et al., 4 Aug 2026) reports on this same benchmark that
MODEL-SELECTED region and scale beats CLICK-CENTERED zooming at every nonzero
round budget (86.0/87.0/88.7/90.3 vs 81.3/79.0/84.0/84.7 at 1/2/3/4 rounds,
GPT-5.5), and that removing cropping costs their framework 10.4 to 41.3 points
depending on backbone. That is an independent group, a different harness, and
the premise this action space was built on. Their absolute numbers are NOT
comparable to ours: they add an OCR engine and a UI-component detector as
coordinate references, run a separate verification call per proposal, and allow
up to 8 cropping rounds with 6 verification retries. We add no external tools
and no extra model calls.

COORDINATE FRAME -- the one place this domain could silently misscore. The
answer is normalised [0,1] on the FULL ORIGINAL SCREENSHOT, never on an
attached crop. This harness makes that safe rather than merely stated:
`_resolve_source` always crops from the original file, and the zoom protocol
already tells the model its region coordinates refer to the full image, so
there is exactly one frame of reference in the whole trajectory. GUI-Lens must
compose its Phi_r mapping back over the crop chain for precisely this reason;
here there is nothing to compose. `ANSWER_STYLES["point"]` carries the wording.

WHAT IS *NOT* GIVEN TO THE MODEL. No OCR text, no detector boxes, no
coordinate references, no element list. The instruction and the pixels only.
This is a deliberate scope choice, not an oversight: coordinate priming is a
separate mechanism (an external tool in the prompt) from view selection (what
this paper studies), and GUI-Lens's own ablation puts it at -1.0 / -2.0 points
on the two strong backbones. Mixing it in would make a view-selection result
unattributable.

GROUP KEY IS THE PICTURE, AND THAT IS NOT THE FILENAME. Measured on the
extracted files: the release ships 1581 rows under 1581 distinct filenames, but
only 1551 distinct images -- 23 pictures appear under two names each (two
captures seconds apart that are byte-identical, e.g. in SolidWorks and MATLAB).
Grouping on `img_filename` would therefore put the same pixels in both gate and
test for 30 rows, which is exactly the leak `mode: by_image` exists to prevent.
`group_key` is the sha of the file bytes (`img_sha`, added by
scripts/add_screenspot_pro_sha.py), so the unit is the picture.

PIXEL BUDGET WARNING, stated because it bounds the protocol rather than being
a detail: these are the largest sources in the corpus (up to 3840x2160). The
per-image token cap measured in §12.9 applies, and the overview is downscaled
hard. That is the point -- it is what makes the crop load-bearing -- but it
also means `overview_long_edge_px` is doing more work here than anywhere else,
and it is left at the corpus default rather than tuned for this domain.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Region, Sample

DOMAIN = "screenspot_pro"

PREAMBLE = (
    "The attached image is a screenshot of a professional desktop "
    "application. Find the single UI control described below and give the "
    "point you would click to operate it.\n\n"
    "Instruction: "
)


def load(root: str | Path, seed: int = 20260911,
         n_items: int = 0) -> list[Sample]:
    """Load ScreenSpot-Pro rows as public/private Sample pairs.

    `n_items = 0` keeps the whole release (1581 rows); a positive value takes a
    deterministic, group-preserving subsample for a cheaper run. The subsample
    is keyed on a hash of the PICTURE, not the row and not the filename, so a
    sampled picture contributes all of its instructions and the group_key
    invariant survives.
    """
    root = Path(root)
    meta = root / "screenspot_pro_meta.json"
    if not meta.exists():
        raise FileNotFoundError(
            f"{meta} missing -- run scripts/prep_screenspot_pro.py first")
    rows = json.loads(meta.read_text())
    if rows and "img_sha" not in rows[0]:
        # Refused rather than silently falling back to the filename: the
        # fallback would look like it worked and would leak 30 rows across the
        # gate/test boundary.
        raise ValueError(
            f"{meta} has no img_sha -- run scripts/add_screenspot_pro_sha.py; "
            "grouping on img_filename leaks byte-identical screenshots across "
            "the split")

    if n_items and n_items < len(rows):
        # Rank PICTURES by hash, then keep whole pictures until the row budget
        # is met. Sampling rows directly would split a picture across the
        # train/test boundary via group_key.
        by_img: dict[str, list] = {}
        for r in rows:
            by_img.setdefault(r["img_sha"], []).append(r)
        order = sorted(by_img, key=lambda k: hashlib.sha256(
            f"sspro|{k}".encode()).hexdigest())
        keep, taken = [], 0
        for k in order:
            if taken >= n_items:
                break
            keep.extend(by_img[k])
            taken += len(by_img[k])
        rows = keep

    samples: list[Sample] = []
    dropped_media = dropped_box = 0
    for r in rows:
        img = root / "images" / r["img_filename"]
        # Checked here, not at render time: every crop re-reads the original
        # file, so a path that resolves only at load would fail mid-trajectory.
        if not img.exists():
            dropped_media += 1
            continue
        # Trust the FILE's dimensions over the declared ones -- the boxes are
        # in source pixels and a mismatch would displace every one of them.
        W, H = int(r["actual_w"]), int(r["actual_h"])
        x0, y0, x1, y1 = (int(r["x0"]), int(r["y0"]),
                          int(r["x1"]), int(r["y1"]))
        if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
            dropped_box += 1
            continue

        box = Region.from_pixels((x0, y0, x1, y1), W, H)
        item_id = f"{DOMAIN}/{r['id']}"
        long_edge = max(x1 - x0, y1 - y0)
        public = PublicItem(
            item_id=item_id, domain=DOMAIN,
            question=PREAMBLE + str(r["instruction"]).strip(),
            media_ids=(str(img),), options=(),
            meta={
                "question_kind": "point",
                "category": r["group"],
                "application": r["application"],
                "platform": r["platform"],
                "ui_type": r["ui_type"],
                "image_width": W, "image_height": H,
                "n_images": 1,
                # Reported, never used for scoring or splitting: it is the
                # quantity that makes this domain different from the rest of
                # the corpus, so the per-size breakdown must be auditable.
                "target_long_edge_px": long_edge,
                "target_area_frac": ((x1 - x0) * (y1 - y0)) / float(W * H),
            },
        )
        private = PrivateItem(
            item_id=item_id,
            # The gold ANSWER is a box, and point-in-box is the metric, so the
            # stored answer text is the box centre purely for logging. Scoring
            # goes through `evidence_regions`, never through this string.
            answer=f"{(x0 + x1) / 2 / W:.4f}, {(y0 + y1) / 2 / H:.4f}",
            answer_type="point",
            evidence_regions=(box,),
            evidence_source="official",
            evidence_verified=True,
            meta={"category": r["group"], "ui_type": r["ui_type"],
                  "box_px": [x0, y0, x1, y1],
                  "image_width": W, "image_height": H,
                  "target_long_edge_px": long_edge},
        )
        # THE PICTURE is what must not straddle a split, and two filenames can
        # carry the same picture (23 of them do), so the key is the sha.
        samples.append(Sample(public=public, private=private,
                              group_key=r["img_sha"]))

    n_icon = sum(1 for s in samples if s.public.meta["ui_type"] == "icon")
    tiny = sum(1 for s in samples
               if s.public.meta["target_long_edge_px"]
               * (1024 / max(s.public.meta["image_width"],
                             s.public.meta["image_height"])) < 10)
    print(f"[{DOMAIN}] kept {len(samples)} instructions over "
          f"{len({s.group_key for s in samples})} distinct pictures "
          f"({dropped_media} missing media, {dropped_box} bad boxes); "
          f"{n_icon} icon / {len(samples)-n_icon} text; "
          f"{tiny} targets under 10px long edge at a 1024px overview")
    return samples


def score(answer_text: str, private: PrivateItem,
          public: PublicItem) -> float:
    """ScreenSpot's point-in-box accuracy, against the released box."""
    from ..evaluation.scorers import point_in_box
    return point_in_box(
        raw=answer_text,
        boxes=private.evidence_regions,
        img_w=int(private.meta["image_width"]),
        img_h=int(private.meta["image_height"]),
    )
