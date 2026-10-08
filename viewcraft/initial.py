"""Initial theta / phi / psi for each domain.

theta starts deliberately plain: a competent but generic instruction, so that
evolution has somewhere to go and the baseline is not crippled on purpose.
phi/psi start at the ordinary native-tool behaviour: whole image at the budget
long edge, model-chosen crops, short visual history.
"""
from __future__ import annotations

import copy

from .views.config import Coordinates, PhiConfig, PsiConfig, SkillConfig

VSTAR_THETA = """# Visual question answering

Answer the question about the image using the option letter.

- Look at the image before answering.
- If the region you need is too small to read, request a zoomed view of it.
- Give your final answer as the letter of the option you choose.
"""

# Deliberately plain, like every other starting theta: it states the task and
# the protocol and NOTHING about strategy. Writing "scan the toolbars first" or
# "zoom before answering" here would hand the search its answer and make any
# gain the starting text rather than the optimisation. The one non-obvious
# thing it must say is the coordinate frame, because that is a protocol fact
# the model cannot infer and getting it wrong misscores every item.
SCREENSPOT_PRO_THETA = """# GUI grounding

Find the UI control the instruction names and give the point to click.

- Look at the screenshot before answering.
- If the control is too small to identify, request a zoomed view of it. You
  may zoom INSIDE a zoom to go progressively finer.
- Coordinates are fractions of the image you name with "image": within a FULL
  frame that is the whole screenshot, within a ZOOM that is the close-up's
  own frame. The system maps them back exactly.
- Give your final answer as the two numbers x, y, and the image they refer to.
"""

CHARTQA_THETA = """# Chart question answering
Answer the question about the chart.

- Read the axis labels, legend and values before answering.
- If a label or value is unreadable, request a zoomed view of it.
- Answer with the value or short phrase the question asks for.
"""

MMLONGDOC_THETA = """# Document question answering

Answer the question using the document.

- The document has many pages; locate the pages that carry the evidence.
- If a table or figure is unreadable, request a zoomed view of it.
- If the document does not contain the answer, say it is not answerable.
"""

NEXTGQA_THETA = """# Video question answering

Answer the question about the video using the option letter.

- Frames are sampled from the whole video; the evidence may be brief.
- If you need a different moment, request frames from that time range.
- Give your final answer as the letter of the option you choose.
"""

VSI_THETA = """# Spatial video question answering

Answer the question about the scene shown in the video.

- Frames are sampled across the walkthrough.
- Track objects across frames so the same object is not counted twice.
- Answer with the number or option the question asks for.
"""

MINIWOB_THETA = """# Web interface task

Complete the task described in the instruction using the screenshot.

- Act on what the current screenshot shows.
- After each action the screen changes; check the new screenshot.
- Use pixel coordinates from the screenshot you were given.
"""

OCRBENCH_THETA = """# Reading text in images

Answer the question using the text visible in the image.

- Find the text the question refers to before answering.
- If the text is too small or blurred to read, request a zoomed view of it.
- Answer with the text or value itself, not a sentence about it.
"""

MMSTAR_THETA = """# Multimodal question answering

Answer the question about the image using the option letter.

- Look at the image before answering.
- If the detail you need is too small to resolve, request a zoomed view of it.
- Give your final answer as the letter of the option you choose.
"""

MMCOMPOSITION_THETA = """# Compositional visual question answering

Answer the question about the image using the option letter.

- The question tests how objects, attributes and relations COMPOSE, so
  verify each candidate option against the pixels rather than picking the
  one that sounds most natural.
- If the detail that separates the options is too small to resolve, request
  a zoomed view of it. You may zoom inside a zoom.
- Give your final answer as the letter of the option you choose.
"""

BABYVISION_THETA = """# Visual perception task

Answer the question about the image.

- Work methodically: scan the whole image first, then verify the details
  that decide the answer.
- If the detail you need is too small to resolve, request a zoomed view of
  it. You may zoom inside a zoom, and you may drop notes on the image (on
  the full frame or inside any zoom) to mark things you have already
  checked.
- Give your final answer in the form the question asks for.
"""

REASONMAP_THETA = """# Route planning on a transit map

Find a route from the departure station to the arrival station on the map.

- The map is far larger than the view you are given, so station names are
  only legible in zoomed views. Locate both stations first, then trace the
  line(s) between them.
- You may zoom inside a zoom, and you may drop notes on the map (on the full
  frame or inside any zoom) to mark stations, transfer points or lines you
  have already traced.
- Coordinates are fractions of the image you name with "image". The system
  maps them back exactly.
- Give your final answer in the route template the question shows.
"""

DOCVQA2026_THETA = """# Document question answering

Answer the question about the document.

- The document has many pages; only some carry the evidence. Decide which
  pages are worth reading before reading closely.
- If the text or figure detail you need is too small to read, request a
  zoomed view of it.
- Give your final answer as the value or short phrase the question asks for.
"""

HALLUSIONBENCH_THETA = """# Visual verification

Answer the yes/no question about the image.

- Decide from what the image actually shows, not from what is usually true.
- If the detail that settles the question is unclear, request a zoomed view.
- Answer exactly yes or no.
"""

COUNTBENCH_THETA = """# Counting objects in an image

Answer how many of the named object are in the image.

- Count every instance, including partly hidden and overlapping ones.
- If the objects are small or crowded, request a zoomed view and count in
  sections so the same object is not counted twice.
- Answer with the number only.
"""

# SYN_COUNT (synthetic hard counting). Deliberately NEUTRAL about the note
# affordance: the interface (build_task_prompt) already tells every arm that a
# `note` action exists, so the starting skill must not pre-load the winning
# strategy -- otherwise an accepted candidate could be credited for a strategy
# the hand-written text already contained. Evolution's job is to DISCOVER
# mark-as-you-count; the start only says to count carefully.
SYN_COUNT_THETA = """# Counting objects in an image

Answer how many of the named object are in the image.

- Count every instance, including partly hidden and overlapping ones.
- Answer with the number only.
"""

# SYN_STATE (synthetic state-tracking). Neutral about the canvas, for the same
# reason as syn_count: the interface already offers the `note` action, so the
# starting skill must not pre-load "jot the state on the slots" -- that is the
# strategy evolution is meant to discover.
SYN_STATE_THETA = """# Tracking state through a sequence of operations

Slots each hold a token. A list of swaps is applied in order. Answer which
token ends in the asked slot.

- Apply each swap to the current state, in the order given.
- Answer with the token letter only.
"""

# SYN_MATH (math as visual workspace). Neutral about the canvas: the interface
# already offers `note`, so the start must not pre-load "write each cell on the
# grid" -- that is the strategy evolution is meant to discover.
SYN_MATH_THETA = """# Filling a table

The top row and left column are given. Every other cell equals the cell above
it plus the cell to its left. Work out the bottom-right cell.

- Compute each cell from the cells it depends on.
- Answer with the number only.
"""

BABYVISION_THETA = """# Visual perception puzzle

Solve the visual puzzle in the image.

- Look carefully at every part of the image before answering.
- If a detail is too small, request a zoomed view of that region.
- Answer in exactly the format the question asks for.
"""

# CLBENCH (long-context rule application). Neutral about the workspace: the
# interface already offers `note`, so the start must not pre-load "write the
# key rules on the sheet" -- that is the strategy evolution is meant to find.
CLBENCH_THETA = """# Applying a long specification

You are given a long specification or rule book and a task that requires
applying it precisely.

- Identify the rules, priorities, and output formats the task depends on.
- Apply them exactly; the answer must satisfy every requirement of the task.
"""

ARC_THETA = """# ARC grid reasoning

Work out the transformation rule from the example input->output pairs, apply
it to the test input, and produce the output grid.

- Compare each input with its output to find the rule.
- Give the final grid as a JSON list of lists of colour indices (0-9).
"""

MUIRBENCH_THETA = """# Multi-image reasoning

Answer the question about the set of images using the option letter.

- The images belong to one question; compare them against each other before
  answering.
- If a detail that distinguishes the images is too small, request a zoomed
  view of that region.
- Give your final answer as the option letter.
"""

HRBENCH_THETA = """# High-resolution image question

Answer the question about the high-resolution image using the option letter.

- The detail that answers the question may be too small to read at first
  glance; request a zoomed view of the region you need.
- Give your final answer as the option letter.
"""

MATHVISION_THETA = """# Math problem with a figure

Solve the math problem using the figure.

- Read the values and relationships shown in the figure carefully.
- If the figure is small or detailed, request a zoomed view of the part you
  need.
- Answer with the number or short value the question asks for.
"""

SPOTDIFF_THETA = """# Spot the difference between two images

Two images are shown. Decide which statement is a REAL difference between them.

- Compare the two images region by region before deciding.
- If a detail is too small to verify, request a zoomed view of that region.
- Answer with the option letter.
"""

TALLYQA_THETA = """# Counting objects in an image

Answer how many of the named object are in the image.

- The objects may be few, small, or partly hidden; look carefully before
  counting.
- If a region is hard to see, request a zoomed view of it.
- Answer with the number only.
"""

NABIRDS_THETA = """# Bird species identification

Identify the bird species in the image using the option letter.

- Fine-grained marks (wing bars, eye ring, bill shape) distinguish the species;
  look at them before deciding.
- If a mark is too small to read, request a zoomed view of that part.
- Answer with the option letter.
"""

FUNSD_THETA = """# Reading a value from a form

Find the named field in the form and read its value.

- The form is dense; locate the field label first, then read the value beside
  or below it.
- If the text is too small to read, request a zoomed view of that area.
- Answer with the value text only.
"""

BLINK_THETA = """# Visual perception

Answer the question about the image or images using the option letter.

- You may be given several images. They appear in the order the question
  refers to them, so "the first image" means the first one you were shown.
- Compare the images against each other before answering.
- If the detail that settles the question is too small to resolve, request a
  zoomed view of it.
- Give your final answer as the letter of the option you choose.
"""

SCIFIBENCH_THETA = """# Matching a caption to one of several figures

Choose which of the attached candidate figures the caption describes.

- The candidates are attached as whole frames in the order the options list
  them: "candidate figure 1" is the first frame you were shown.
- The candidates come from the same body of work, so they often share subject,
  axis style and colour scheme. What separates them is usually a specific
  detail -- an axis label, a legend entry, the number of plotted series, the
  quantity on an axis.
- Read the caption for the details it commits to, then check each candidate
  against them rather than picking the one that looks broadly right.
- If a label or legend is too small to read, request a zoomed view of it.
- Give your final answer as the letter of the option you choose.
"""

MMAD_THETA = """# Inspecting a product against a known-good reference

Answer the question about the product unit in the first image.

- The second image is a known-good unit of the same product. Use it to learn
  what this product looks like when nothing is wrong; the question is never
  about it.
- Line the two up part by part rather than comparing them as a whole. A real
  defect is a difference at one place; lighting, pose and ordinary
  manufacturing variation differ everywhere.
- If a suspected difference is too small to judge, request a zoomed view of
  that region.
- Give your final answer as the letter of the option you choose.
"""

RSRCC_THETA = """# Reading change between two overhead captures

Answer the question about what changed between the two images.

- The first image is the BEFORE capture and the second is the AFTER capture.
  They cover the same area, so every place in one has a counterpart in the
  other; the direction of a change depends on which is which.
- Find the place the question names in BOTH images before deciding what
  happened there. A difference you can only see in one of them is not yet a
  change.
- If the place in question is too small to judge, request a zoomed view of it.
- Give your final answer as the letter of the option you choose.
"""

REALWORLDQA_THETA = """# Real-world scene question answering

Answer the question about the photographed scene.

- The object the question asks about may be small or far away; find it before
  answering.
- If it is too small to resolve, request a zoomed view of it.
- Some questions list lettered options and some do not. If options are
  listed, answer with the option letter; otherwise answer with the single
  word or number the question asks for.
"""

MPDOCVQA_THETA = """# Multi-page document question answering

Answer the question using the document pages.

- The pages come from one document; the answer is on a single page and you
  are not told which, so find that page before answering.
- If a table, form field or figure is unreadable, request a zoomed view of it.
- Answer with the value or short phrase the question asks for, copied from
  the document rather than paraphrased.
"""

CHARXIV_THETA = """# Scientific figure reasoning

Answer the question about the figure.

- The figure is from a research paper and may have several panels; work out
  which panel the question refers to before answering.
- Read the axis labels, tick values and legend rather than judging by shape
  alone. If any of them is unreadable, request a zoomed view of it.
- The answer usually has to be derived -- a comparison, a count, a
  difference -- not just read off. Answer with the value or short label the
  question asks for.
"""

DUDE_THETA = """# Long document question answering

Answer the question using the document pages.

- The document can be long and the pages come from one file; locate the
  pages that carry the evidence before answering. The evidence for one
  question is sometimes spread over more than one page.
- The pages vary in quality -- some are scans, some are forms, some are
  slides. If a field, table or figure is unreadable, request a zoomed view.
- If the document does not contain the answer, say it is not answerable.
- Otherwise answer with the value or short phrase the question asks for,
  copied from the document rather than paraphrased.
"""

MVBENCH_THETA = """# Temporal video question answering

Answer the question about the video using the option letter.

- The frames are sampled in time order across the clip. The question is
  about what CHANGES between them, so compare frames against each other
  rather than judging from any single one.
- If the moment that settles the question falls between the frames you were
  given, request frames from that part of the clip.
- Give your final answer as the letter of the option you choose.
"""

_THETA = {
    "vstar": VSTAR_THETA,
    "screenspot_pro": SCREENSPOT_PRO_THETA,
    "screenspot_v2": SCREENSPOT_PRO_THETA,
    "mmcomposition": MMCOMPOSITION_THETA,
    "reasonmap": REASONMAP_THETA,
    "docvqa2026": DOCVQA2026_THETA,
    "bv_maze": BABYVISION_THETA,
    "bv_count_clusters": BABYVISION_THETA,
    "bv_find_different": BABYVISION_THETA,
    "bv_connect_lines": BABYVISION_THETA,
    "bv_find_different_hard": BABYVISION_THETA,
    "bv_find_different_hard2": BABYVISION_THETA,
    "bv_maze_real": BABYVISION_THETA,
    # bvgs_* (gpt-6 v2 transfer sets): same task family, same starting
    # point -- a different hand-written theta would make the bv/bvgs
    # comparison a confound of starting text rather than of training data.
    "bvgs_maze": BABYVISION_THETA,
    "bvgs_count_clusters": BABYVISION_THETA,
    "bvgs_find_different": BABYVISION_THETA,
    "bvgs_connect_lines": BABYVISION_THETA,
    "bvgs_find_shadow": BABYVISION_THETA,
    "chartqa": CHARTQA_THETA,
    "mmlongdoc": MMLONGDOC_THETA,
    "nextgqa": NEXTGQA_THETA,
    "vsi": VSI_THETA,
    "miniwob": MINIWOB_THETA,
    "ocrbench": OCRBENCH_THETA,
    "mmstar": MMSTAR_THETA,
    "hallusionbench": HALLUSIONBENCH_THETA,
    "countbench": COUNTBENCH_THETA,
    "syn_count": SYN_COUNT_THETA,
    "syn_state": SYN_STATE_THETA,
    "syn_math": SYN_MATH_THETA,
    "babyvision": BABYVISION_THETA,
    "clbench": CLBENCH_THETA,
    "arc": ARC_THETA,
    "muirbench": MUIRBENCH_THETA,
    "hrbench": HRBENCH_THETA,
    "mathvision": MATHVISION_THETA,
    "spotdiff": SPOTDIFF_THETA,
    "tallyqa": TALLYQA_THETA,
    "nabirds": NABIRDS_THETA,
    "funsd": FUNSD_THETA,
    "blink": BLINK_THETA,
    "blink_hard": BLINK_THETA,
    "realworldqa": REALWORLDQA_THETA,
    "scifibench": SCIFIBENCH_THETA,
    "mmad": MMAD_THETA,
    "rsrcc": RSRCC_THETA,
    # Fourth wave: multi-unit domains, added because the mechanism's
    # selection half needs many addressable source units per question.
    "mpdocvqa": MPDOCVQA_THETA,
    "dude": DUDE_THETA,
    "mvbench": MVBENCH_THETA,
    # Perception Test is the same shape of task as MVBench -- a
    # 32-frame pool, multiple choice, questions about change over
    # time -- so it starts from the same hand-written theta. A
    # different starting text would make the two incomparable.
    "perceptiontest": MVBENCH_THETA,
    # Same 32-frame-pool video shape as MVBench/PerceptionTest, so the same
    # starting point: a different hand-written theta would make the
    # long-horizon rows incomparable with the short-clip ones.
    "videoholmes": MVBENCH_THETA,
    "openeqa": MVBENCH_THETA,
    "stibench": MVBENCH_THETA,
    "mmiu": MMCOMPOSITION_THETA,
    "slidevqa": MPDOCVQA_THETA,
    "charxiv": CHARXIV_THETA,
}

# phi/psi defaults per domain: the ordinary native behaviour.
_PHI = {
    "vstar": PhiConfig(overview_long_edge_px=1024, crop_long_edge_px=1024,
                       crop_margin=0.02, max_crops_per_step=2,
                       allow_upscale=False, union_crop_for_multi_target=True),
    # Left at the corpus default 1024 rather than raised for this domain, even
    # though the sources are the largest here (up to 3840x2160). Raising it
    # would buy accuracy by widening the overview instead of by selecting
    # views, which is the mechanism under test -- and it would make this
    # domain's number incomparable with every other row in the main table.
    # The consequence is deliberate and is the point: at 1024 px, 24.5% of
    # targets have a long edge under 10 px, so the crop has to do the work.
    # `union_crop_for_multi_target` is off because the instruction names ONE
    # control; there is no second target to keep in frame.
    "screenspot_pro": PhiConfig(overview_long_edge_px=1024,
                                crop_long_edge_px=1024, crop_margin=0.02,
                                max_crops_per_step=2, allow_upscale=False,
                                union_crop_for_multi_target=False),
    "screenspot_v2": PhiConfig(overview_long_edge_px=1024,
                               crop_long_edge_px=1024, crop_margin=0.02,
                               max_crops_per_step=2, allow_upscale=False,
                               union_crop_for_multi_target=False),
    # The ordinary native starting point, same as every static single-image
    # domain: whole image at 1024, model-chosen crops at 1024, no upscaling.
    "mmcomposition": PhiConfig(overview_long_edge_px=1024,
                               crop_long_edge_px=1024, crop_margin=0.02,
                               max_crops_per_step=2),
    # Same native starting point even though these are the largest sources in
    # the corpus: starting any domain closer to its own optimum would hand
    # that domain's arms a head start the baseline does not get.
    "reasonmap": PhiConfig(overview_long_edge_px=1024,
                           crop_long_edge_px=1024, crop_margin=0.02,
                           max_crops_per_step=2),
    "docvqa2026": PhiConfig(overview_long_edge_px=1024,
                            crop_long_edge_px=1024, crop_margin=0.02,
                            max_crops_per_step=2),
    "bv_maze": PhiConfig(overview_long_edge_px=1024, crop_long_edge_px=1024,
                         crop_margin=0.02, max_crops_per_step=2),
    "bv_count_clusters": PhiConfig(overview_long_edge_px=1024,
                                   crop_long_edge_px=1024, crop_margin=0.02,
                                   max_crops_per_step=2),
    "bv_find_different": PhiConfig(overview_long_edge_px=1024,
                                   crop_long_edge_px=1024, crop_margin=0.02,
                                   max_crops_per_step=2),
    "bv_connect_lines": PhiConfig(overview_long_edge_px=1024,
                                  crop_long_edge_px=1024, crop_margin=0.02,
                                  max_crops_per_step=2),
    "bv_find_different_hard": PhiConfig(overview_long_edge_px=1024,
                                        crop_long_edge_px=1024,
                                        crop_margin=0.02,
                                        max_crops_per_step=2),
    "bv_find_different_hard2": PhiConfig(overview_long_edge_px=1024,
                                         crop_long_edge_px=1024,
                                         crop_margin=0.02,
                                         max_crops_per_step=2),
    "bv_maze_real": PhiConfig(overview_long_edge_px=1024,
                              crop_long_edge_px=1024, crop_margin=0.02,
                              max_crops_per_step=2),
    # bvgs_* start where bv_* start, same confound argument as _THETA.
    "bvgs_maze": PhiConfig(overview_long_edge_px=1024, crop_long_edge_px=1024,
                           crop_margin=0.02, max_crops_per_step=2),
    "bvgs_count_clusters": PhiConfig(overview_long_edge_px=1024,
                                     crop_long_edge_px=1024, crop_margin=0.02,
                                     max_crops_per_step=2),
    "bvgs_find_different": PhiConfig(overview_long_edge_px=1024,
                                     crop_long_edge_px=1024, crop_margin=0.02,
                                     max_crops_per_step=2),
    "bvgs_connect_lines": PhiConfig(overview_long_edge_px=1024,
                                    crop_long_edge_px=1024, crop_margin=0.02,
                                    max_crops_per_step=2),
    "bvgs_find_shadow": PhiConfig(overview_long_edge_px=1024,
                                  crop_long_edge_px=1024, crop_margin=0.02,
                                  max_crops_per_step=2),
    "chartqa": PhiConfig(overview_long_edge_px=1024, crop_long_edge_px=1024,
                         crop_margin=0.02, max_crops_per_step=2),
    "mmlongdoc": PhiConfig(overview_long_edge_px=1024, crop_long_edge_px=1024,
                           crop_margin=0.02, max_crops_per_step=2),
    "nextgqa": PhiConfig(overview_long_edge_px=768, crop_long_edge_px=768,
                         crop_margin=0.02, max_crops_per_step=2),
    "vsi": PhiConfig(overview_long_edge_px=768, crop_long_edge_px=768,
                     crop_margin=0.02, max_crops_per_step=2),
    "miniwob": PhiConfig(overview_long_edge_px=1024, crop_long_edge_px=1024,
                         crop_margin=0.0, max_crops_per_step=1),
    # Second wave. Same ordinary native starting point as every other domain:
    # whole image at 1024, model-chosen crops at 1024, no upscaling. Starting
    # any domain closer to its own optimum would hand that domain's arms a
    # head start that the baseline does not get.
    "ocrbench": PhiConfig(overview_long_edge_px=1024, crop_long_edge_px=1024,
                          crop_margin=0.02, max_crops_per_step=2),
    "mmstar": PhiConfig(overview_long_edge_px=1024, crop_long_edge_px=1024,
                        crop_margin=0.02, max_crops_per_step=2),
    "hallusionbench": PhiConfig(overview_long_edge_px=1024,
                                crop_long_edge_px=1024, crop_margin=0.02,
                                max_crops_per_step=2),
    # Third wave. Same ordinary native starting point again: whole image at
    # 1024, model-chosen crops at 1024, no upscaling. BLINK gets it too even
    # though it is multi-image -- the per-request image cap is a psi concern,
    # not a phi one, so there is nothing domain-specific to set here.
    "countbench": PhiConfig(overview_long_edge_px=1024,
                            crop_long_edge_px=1024, crop_margin=0.02,
                            max_crops_per_step=2),
    # syn_count: single 1024px image, so the whole image already fits the
    # budget -- the task is enumeration memory, and the coordinate under test
    # is the CANVAS (canvas_expr), not resolution. max_crops_per_step=4 lets
    # one note action jot several counted dots in a single step.
    "syn_count": PhiConfig(overview_long_edge_px=1024,
                           crop_long_edge_px=1024, crop_margin=0.02,
                           max_crops_per_step=4),
    "syn_state": PhiConfig(overview_long_edge_px=1024,
                           crop_long_edge_px=1024, crop_margin=0.02,
                           max_crops_per_step=7),
    "syn_math": PhiConfig(overview_long_edge_px=1024,
                          crop_long_edge_px=1024, crop_margin=0.02,
                          max_crops_per_step=8),
    "babyvision": PhiConfig(overview_long_edge_px=1024,
                             crop_long_edge_px=1024, crop_margin=0.02,
                             max_crops_per_step=4),
    "clbench": PhiConfig(overview_long_edge_px=1024,
                          crop_long_edge_px=1024, crop_margin=0.02,
                          max_crops_per_step=6),
    "arc": PhiConfig(overview_long_edge_px=1400,
                      crop_long_edge_px=1024, crop_margin=0.02,
                      max_crops_per_step=8),
    "muirbench": PhiConfig(overview_long_edge_px=1024,
                            crop_long_edge_px=1024, crop_margin=0.02,
                            max_crops_per_step=4),
    "hrbench": PhiConfig(overview_long_edge_px=1400,
                          crop_long_edge_px=1400, crop_margin=0.02,
                          max_crops_per_step=4),
    "mathvision": PhiConfig(overview_long_edge_px=1024,
                             crop_long_edge_px=1024, crop_margin=0.02,
                             max_crops_per_step=6),
    "spotdiff": PhiConfig(overview_long_edge_px=1024,
                           crop_long_edge_px=1024, crop_margin=0.02,
                           max_crops_per_step=4),
    "tallyqa": PhiConfig(overview_long_edge_px=1024,
                          crop_long_edge_px=1024, crop_margin=0.02,
                          max_crops_per_step=4),
    "nabirds": PhiConfig(overview_long_edge_px=1024,
                          crop_long_edge_px=1024, crop_margin=0.02,
                          max_crops_per_step=4),
    "funsd": PhiConfig(overview_long_edge_px=1400,
                        crop_long_edge_px=1400, crop_margin=0.02,
                        max_crops_per_step=4),
    "blink": PhiConfig(overview_long_edge_px=1024, crop_long_edge_px=1024,
                       crop_margin=0.02, max_crops_per_step=2),
    "blink_hard": PhiConfig(overview_long_edge_px=1024,
                            crop_long_edge_px=1024, crop_margin=0.02,
                            max_crops_per_step=2),
    "realworldqa": PhiConfig(overview_long_edge_px=1024,
                             crop_long_edge_px=1024, crop_margin=0.02,
                             max_crops_per_step=2),
    # SciFIBench: five source figures per item, median long edge 1704 px, so
    # 1024 is the same ordinary starting point every other domain gets and a
    # crop still recovers real detail on most candidates. max_crops_per_step
    # 2 matches blink for the same reason -- with five sources already
    # occupying slots, a step that adds more than two derived views spends
    # the whole cap on one candidate.
    "scifibench": PhiConfig(overview_long_edge_px=1024,
                            crop_long_edge_px=1024, crop_margin=0.02,
                            max_crops_per_step=2),
    # MMAD. Ordinary native starting point, as everywhere else: whole frame
    # at 1024, model-chosen crops at 1024, no upscaling. The industrial
    # sources are 1024-2000 px on the long edge, so a crop of a suspected
    # region genuinely recovers detail rather than re-sending the same
    # pixels -- but nothing here pre-decides that the answer is to zoom,
    # since the domain's real question is which regions to PAIR.
    "mmad": PhiConfig(overview_long_edge_px=1024, crop_long_edge_px=1024,
                      crop_margin=0.02, max_crops_per_step=2),
    # RSRCC. The release is 512 px square throughout, and the overview budget
    # is set to 512 rather than the usual 1024 SO THAT the whole-image view is
    # the native file. `render.py:_fit` refuses to upscale, so with this
    # setting every crop returns exactly the pixels the overview already
    # showed: resolution headroom is zero arithmetically, not merely small.
    # That is the point -- see adapters/rsrcc.py for the pre-registered
    # prediction this domain exists to test. Raising the budget to 1024 would
    # NOT add detail (the file has none) but would silently upscale on the
    # crop path and muddy the claim. crop_long_edge_px matches, and 2 crops
    # per step is what it takes to cut the same region out of both captures.
    "rsrcc": PhiConfig(overview_long_edge_px=512, crop_long_edge_px=512,
                       crop_margin=0.02, max_crops_per_step=2),
    # Fourth wave. Same ordinary native starting point as every other
    # domain -- whole page at 1024, model-chosen crops at 1024, no
    # upscaling -- so no domain starts closer to its own optimum than the
    # others. Matches mmlongdoc exactly, which is the same shape of task:
    # many document pages, one of which carries the evidence.
    "mpdocvqa": PhiConfig(overview_long_edge_px=1024, crop_long_edge_px=1024,
                          crop_margin=0.02, max_crops_per_step=2),
    # Same ordinary starting point as the other document domains. DUDE's
    # pages are heterogeneous (scans, forms, slides) and some of them would
    # plainly benefit from a higher overview, which is exactly why that is
    # not pre-set: the rendering choice is what the mechanism is supposed to
    # search for, and starting this domain closer to its own optimum would
    # hand its arms a head start the baseline does not get.
    "dude": PhiConfig(overview_long_edge_px=1024, crop_long_edge_px=1024,
                      crop_margin=0.02, max_crops_per_step=2),
    # 768 px, matching the other two video domains rather than the 1024 the
    # document domains use: a sampled frame carries far less recoverable
    # fine detail than a rasterised page, and the per-request budget is
    # spent on MORE frames instead of sharper ones.
    "perceptiontest": PhiConfig(overview_long_edge_px=768,
                                crop_long_edge_px=768,
                                max_crops_per_step=2),
    "mvbench": PhiConfig(overview_long_edge_px=768, crop_long_edge_px=768,
                         crop_margin=0.02, max_crops_per_step=2),
    # Same starting render as mvbench: the frame pool is the same size, and
    # a sharper starting view would pre-empt the rendering choice the
    # mechanism is meant to search.
    "videoholmes": PhiConfig(overview_long_edge_px=768, crop_long_edge_px=768,
                             crop_margin=0.02, max_crops_per_step=2),
    "openeqa": PhiConfig(overview_long_edge_px=768, crop_long_edge_px=768,
                         crop_margin=0.02, max_crops_per_step=2),
    "stibench": PhiConfig(overview_long_edge_px=768, crop_long_edge_px=768,
                          crop_margin=0.02, max_crops_per_step=2),
    "mmiu": PhiConfig(overview_long_edge_px=1024, crop_long_edge_px=1024,
                      crop_margin=0.02, max_crops_per_step=2),
    "slidevqa": PhiConfig(overview_long_edge_px=1024, crop_long_edge_px=1024,
                          crop_margin=0.02, max_crops_per_step=2),
    # CharXiv is single-image, so it takes the same ordinary starting point
    # as every other static domain. A multi-panel arXiv figure is exactly
    # the case where a higher overview would help, which is precisely why it
    # is NOT set here: handing this domain a better starting render than the
    # others would pre-empt the rendering choice the mechanism is meant to
    # search for itself.
    "charxiv": PhiConfig(overview_long_edge_px=1024, crop_long_edge_px=1024,
                         crop_margin=0.02, max_crops_per_step=2),
}

_PSI = {
    "vstar": PsiConfig(keep_overview=True, annotate_overview_boxes=False,
                       history_keep_last_n=3, order="overview_first",
                       evict_policy="fifo", max_images_per_request=6),
    # `keep_overview=True` is not a tuning choice here, it is a correctness
    # requirement: the answer is a point on the FULL screenshot, so the model
    # must be able to see the full screenshot when it answers. Evicting the
    # overview would leave it estimating a global coordinate from a crop.
    "screenspot_pro": PsiConfig(keep_overview=True,
                                annotate_overview_boxes=False,
                                history_keep_last_n=3, order="overview_first",
                                evict_policy="fifo",
                                max_images_per_request=6),
    "screenspot_v2": PsiConfig(keep_overview=True,
                               annotate_overview_boxes=False,
                               history_keep_last_n=3, order="overview_first",
                               evict_policy="fifo",
                               max_images_per_request=6),
    "chartqa": PsiConfig(history_keep_last_n=3, max_images_per_request=6),
    "mmcomposition": PsiConfig(history_keep_last_n=3,
                               max_images_per_request=6),
    "reasonmap": PsiConfig(history_keep_last_n=4,
                           max_images_per_request=6),
    "docvqa2026": PsiConfig(history_keep_last_n=4,
                            max_images_per_request=8),
    "bv_maze": PsiConfig(history_keep_last_n=3, max_images_per_request=6),
    "bv_count_clusters": PsiConfig(history_keep_last_n=3,
                                   max_images_per_request=6),
    "bv_find_different": PsiConfig(history_keep_last_n=3,
                                   max_images_per_request=6),
    "bv_connect_lines": PsiConfig(history_keep_last_n=3,
                                  max_images_per_request=6),
    "bv_find_different_hard": PsiConfig(history_keep_last_n=3,
                                        max_images_per_request=6),
    "bv_find_different_hard2": PsiConfig(history_keep_last_n=3,
                                         max_images_per_request=6),
    "bv_maze_real": PsiConfig(history_keep_last_n=3,
                              max_images_per_request=6),
    "bvgs_maze": PsiConfig(history_keep_last_n=3, max_images_per_request=6),
    "bvgs_count_clusters": PsiConfig(history_keep_last_n=3,
                                     max_images_per_request=6),
    "bvgs_find_different": PsiConfig(history_keep_last_n=3,
                                     max_images_per_request=6),
    "bvgs_connect_lines": PsiConfig(history_keep_last_n=3,
                                    max_images_per_request=6),
    "bvgs_find_shadow": PsiConfig(history_keep_last_n=3,
                                  max_images_per_request=6),
    "mmlongdoc": PsiConfig(history_keep_last_n=4, max_images_per_request=8),
    "nextgqa": PsiConfig(history_keep_last_n=4, max_images_per_request=8),
    "vsi": PsiConfig(history_keep_last_n=4, max_images_per_request=8),
    "miniwob": PsiConfig(keep_overview=False, history_keep_last_n=2,
                         max_images_per_request=4),
    "ocrbench": PsiConfig(history_keep_last_n=3, max_images_per_request=6),
    "mmstar": PsiConfig(history_keep_last_n=3, max_images_per_request=6),
    "hallusionbench": PsiConfig(history_keep_last_n=3,
                                max_images_per_request=6),
    "countbench": PsiConfig(history_keep_last_n=3, max_images_per_request=6),
    # syn_count: the canvas lives on the overview, so the overview must stay in
    # context (keep_overview) and not be evicted mid-count.
    "syn_count": PsiConfig(history_keep_last_n=3, max_images_per_request=6,
                           keep_overview=True),
    "syn_state": PsiConfig(history_keep_last_n=3, max_images_per_request=6,
                           keep_overview=True),
    "syn_math": PsiConfig(history_keep_last_n=3, max_images_per_request=6,
                          keep_overview=True),
    "babyvision": PsiConfig(history_keep_last_n=3, max_images_per_request=6,
                            keep_overview=True),
    "clbench": PsiConfig(history_keep_last_n=4, max_images_per_request=6,
                          keep_overview=True),
    "arc": PsiConfig(history_keep_last_n=4, max_images_per_request=8,
                      keep_overview=True),
    "muirbench": PsiConfig(history_keep_last_n=6, max_images_per_request=9,
                            keep_overview=True),
    "hrbench": PsiConfig(history_keep_last_n=4, max_images_per_request=8,
                          keep_overview=True),
    "mathvision": PsiConfig(history_keep_last_n=4, max_images_per_request=6,
                             keep_overview=True),
    "spotdiff": PsiConfig(history_keep_last_n=4, max_images_per_request=6,
                           keep_overview=True),
    "tallyqa": PsiConfig(history_keep_last_n=4, max_images_per_request=6,
                          keep_overview=True),
    "nabirds": PsiConfig(history_keep_last_n=4, max_images_per_request=6,
                          keep_overview=True),
    "funsd": PsiConfig(history_keep_last_n=4, max_images_per_request=6,
                        keep_overview=True),
    # BLINK is the only static domain whose SOURCE is multi-image: an item
    # ships up to 4 images before the mechanism adds a single crop. At the
    # 6-image cap every other domain uses, a 4-image item has room for only
    # two derived views and the source images themselves start competing with
    # crops for the same slots -- eviction would then drop an image the
    # question refers to positionally ("the first image"), turning a layout
    # choice into a wrong-picture error. 8 leaves 4 slots for views on the
    # worst case while keeping the cap a real constraint the optimizer must
    # work within. Matches the video domains, which face the same
    # many-sources-per-item pressure.
    "blink": PsiConfig(history_keep_last_n=4, max_images_per_request=8),
    "blink_hard": PsiConfig(history_keep_last_n=4, max_images_per_request=8),
    # Fourth wave. 8 images per request, matching mmlongdoc rather than the
    # single-image domains' 6: step 0 spends the whole image cap on pooled
    # pages, so a 6-image cap against a median 6-page (max 20) document
    # would leave almost no slots for the derived crops the locate-then-read
    # chain needs. 8 of 20 pages also keeps the cap a REAL constraint -- the
    # model sees at most 40% of the largest documents per request, which is
    # the selection pressure this domain exists to create.
    "mpdocvqa": PsiConfig(history_keep_last_n=4, max_images_per_request=8),
    # 8 images, matching mmlongdoc and mpdocvqa. Against a 112-page
    # document that is 7% visible per request, which is the strongest
    # selection pressure in the study -- and the reason the cap is not
    # raised: a larger cap would dissolve the decision being measured.
    "dude": PsiConfig(history_keep_last_n=4, max_images_per_request=8),
    # Same 8-image cap as the other video domains: step 0 spends it on
    # sampled frames, so 8 of a 32-frame pool is the selection pressure the
    # domain exists to create.
    "mvbench": PsiConfig(history_keep_last_n=4, max_images_per_request=8),
    "videoholmes": PsiConfig(history_keep_last_n=4, max_images_per_request=8),
    "openeqa": PsiConfig(history_keep_last_n=4, max_images_per_request=8),
    "stibench": PsiConfig(history_keep_last_n=4, max_images_per_request=8),
    "mmiu": PsiConfig(history_keep_last_n=4, max_images_per_request=8),
    "slidevqa": PsiConfig(history_keep_last_n=4, max_images_per_request=8),
    "perceptiontest": PsiConfig(history_keep_last_n=4,
                                max_images_per_request=8),
    # Single image per item, so the 6-image cap the other static domains use
    # is right here too: the source costs one slot and the remaining five
    # are for derived crops of individual panels.
    "charxiv": PsiConfig(history_keep_last_n=3, max_images_per_request=6),
    "realworldqa": PsiConfig(history_keep_last_n=3, max_images_per_request=6),
    # SciFIBench ships FIVE source figures, one more than BLINK's worst case,
    # and the option letter is positional over them, so the BLINK note above
    # applies with less slack: at the 6-image cap a five-candidate item has
    # room for exactly one derived view, and any eviction drops a candidate
    # the options name by position. 8 leaves three slots while keeping the
    # cap a real constraint -- "keep all five and zoom one" and "drop a ruled-
    # out candidate to inspect another" are both reachable, which is the
    # decision this domain was added to pose.
    "scifibench": PsiConfig(history_keep_last_n=4, max_images_per_request=8),
    # MMAD ships only two sources, so the cap is about how many DERIVED
    # views may sit beside them: at 6 the query, the reference and two
    # matched crop pairs coexist, and evicting one becomes a real choice
    # rather than a forced one.
    "mmad": PsiConfig(history_keep_last_n=4, max_images_per_request=6),
    # RSRCC ships two sources per item, as MMAD does, and for the same reason
    # the cap is about the DERIVED views beside them: at 6 the before, the
    # after and two matched crop pairs coexist. Below that the mechanism
    # cannot hold a matched pair at all, which would decide the experiment by
    # the budget rather than by the search.
    "rsrcc": PsiConfig(history_keep_last_n=4, max_images_per_request=6),
}


def initial_coordinates(domain: str) -> Coordinates:
    """Fresh, independent coordinates for one arm.

    The returned configs MUST be deep copies: the tables above are module
    level, and handing out the same PhiConfig/PsiConfig instance to every arm
    let one arm's change mutate the shared object and silently apply to every
    other arm.

    A `<parent>_<group>` domain inherits `<parent>`'s starting point exactly.
    The MVBench requirement groups are the same data and the same metric
    under a task subset, so they must start where `mvbench` started: a
    different hand-written theta would make their results incomparable with
    the full-mix number they exist to refine, and any gain could then be the
    new starting text rather than the search.
    """
    if domain not in _THETA:
        parent = domain.rsplit("_", 1)[0]
        if parent in _THETA:
            domain = parent
        else:
            raise KeyError(f"no initial coordinates for domain {domain!r}")
    return Coordinates(theta=SkillConfig(_THETA[domain]),
                       phi=copy.deepcopy(_PHI[domain]),
                       psi=copy.deepcopy(_PSI[domain]))
