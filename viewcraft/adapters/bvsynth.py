"""BabyVision TRANSFER adapter: evolve on SYNTHETIC items, test on REAL ones.

The protocol, per the user's design: for each BabyVision subtype, a generator
(babygen/, written with gpt-6-astra assistance against real examples) produces
~20 synthetic training items whose gold answers are known by construction.
Evolution runs on those; the FINAL config is then evaluated on the real
BabyVision subset, which the search never sees. This tests whether an evolved
visual strategy transfers across a distribution shift, not just to held-out
items of the same generator.

Splits are FORCED by a pre-written plan (scripts/prep_bvsynth.py), not by the
hashed assignment: synthetic groups land in train/gate/diagnostic, real
groups in test. Both item kinds pass through the same scorer.

Scoring mirrors babyvision.py: choice -> letter; blank -> integer tuple
(coords or count); the line-mapping answer ("1-C,2-F,...") -> normalised
mapping match.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..paths import data_root
from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "bvsynth"

BABYGEN = Path(__file__).resolve().parents[2] / "babygen" / "out"
REAL_ROOT = str(data_root() / "babyvision")

SUBTYPES = {
    "maze": {"syn_dir": "maze", "real_subtype": "Maze"},
    "maze_real": {"syn_dir": "maze_real", "real_subtype": "Maze"},
    "count_clusters": {"syn_dir": "count_clusters",
                       "real_subtype": "Count Clusters"},
    "find_different": {"syn_dir": "find_different",
                       "real_subtype": "Find the different"},
    "find_different_hard": {"syn_dir": "find_different_hard",
                            "real_subtype": "Find the different"},
    "find_different_hard2": {"syn_dir": "find_different_hard2",
                             "real_subtype": "Find the different"},
    "connect_lines": {"syn_dir": "connect_lines",
                      "real_subtype": "Connect the lines"},
}


def _norm_numbers(s: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"-?\d+", str(s) or ""))


def _norm_mapping(s: str) -> str:
    pairs = re.findall(r"(\d+)\s*[-=:]\s*([A-Za-z])", str(s) or "")
    return ",".join(f"{a}-{b.upper()}" for a, b in sorted(pairs))


def _norm_text(s: str) -> str:
    """Case/whitespace-insensitive text equality, for comparison operators
    ('<', '>'), colour mappings ('A - Blue, ...') and word sequences."""
    return re.sub(r"\s+", "", str(s) or "").lower().strip(".,;:")


def load(root: str | Path, seed: int = 20260911,
         subtype: str = "maze") -> list[Sample]:
    cfg = SUBTYPES[subtype]
    samples: list[Sample] = []

    # --- synthetic items (evolution material) ---
    items_file = BABYGEN / cfg["syn_dir"] / "items.jsonl"
    if not items_file.exists():
        raise FileNotFoundError(
            f"{items_file} missing -- run babygen/gen_{cfg['syn_dir']}.py")
    for i, line in enumerate(items_file.read_text().splitlines()):
        r = json.loads(line)
        img = BABYGEN / cfg["syn_dir"] / r["image_relpath"]
        if not img.exists():
            continue
        split_tag = (r.get("meta") or {}).get("split", "train")
        item_id = f"{DOMAIN}/{subtype}_syn_{i:03d}"
        public = PublicItem(
            item_id=item_id, domain=DOMAIN, question=str(r["question"]),
            media_ids=(str(img),),
            options=tuple((r.get("meta") or {}).get("options", ())),
            meta={"subtype": subtype, "origin": "synthetic",
                  "question_type": subtype})
        private = PrivateItem(
            item_id=item_id, answer=str(r["answer"]), answer_type="string",
            evidence_source="constructed",
            meta={"origin": "synthetic", "split_tag": split_tag})
        samples.append(Sample(public=public, private=private,
                              group_key=f"syn/{subtype}/{i:03d}"))

    # --- real BabyVision items (held-out test) ---
    import pyarrow.parquet as pq
    rows = []
    for p in sorted(Path(REAL_ROOT).rglob("*.parquet")):
        rows.extend(pq.read_table(p).to_pylist())
    n_real = 0
    for i, r in enumerate(rows):
        if r.get("subtype") != cfg["real_subtype"]:
            continue
        blob = r.get("image")
        blob = blob.get("bytes") if isinstance(blob, dict) else blob
        if not blob:
            continue
        media = write_media(DOMAIN, blob, f"{subtype}_real_{i}")
        ans_type = str(r.get("ansType") or "blank")
        options: tuple = ()
        gold_letter = ""
        if ans_type == "choice":
            import ast
            try:
                opts = ast.literal_eval(str(r.get("options") or "[]"))
            except (ValueError, SyntaxError):
                opts = []
            options = tuple(str(o) for o in opts)
            try:
                idx = int(r.get("choiceAns"))
                gold_letter = (chr(ord("A") + idx)
                               if 0 <= idx < len(options) else "")
            except (TypeError, ValueError):
                gold_letter = ""
        ans = gold_letter if ans_type == "choice" else str(r.get("blankAns") or "")
        item_id = f"{DOMAIN}/{subtype}_real_{i:03d}"
        public = PublicItem(
            item_id=item_id, domain=DOMAIN, question=str(r.get("question")),
            media_ids=(media,), options=options,
            meta={"subtype": subtype, "origin": "real",
                  "question_type": subtype})
        private = PrivateItem(
            item_id=item_id, answer=ans, answer_type="string",
            evidence_source="official",
            meta={"origin": "real",
                  "choice_letter": gold_letter,
                  "blank_ans": str(r.get("blankAns") or "")})
        samples.append(Sample(public=public, private=private,
                              group_key=f"real/{subtype}/{i:03d}"))
        n_real += 1
    print(f"[{DOMAIN}/{subtype}] {len(samples) - n_real} synthetic + "
          f"{n_real} real items")
    return samples


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    ans = (answer_text or "").strip()
    meta = private.meta or {}
    if meta.get("origin") == "real":
        if meta.get("choice_letter"):
            m = re.search(r"\b([A-Z])\b", ans.upper())
            return 1.0 if m and m.group(1) == meta["choice_letter"] else 0.0
        want = _norm_numbers(meta.get("blank_ans", ""))
        got = _norm_numbers(ans)
        if want and got == want:
            return 1.0
        # a real mapping answer ("1-C,2-F") is compared as a mapping
        want_map = _norm_mapping(meta.get("blank_ans", ""))
        if want_map and _norm_mapping(ans) == want_map:
            return 1.0
        # comparison operators / colour mappings / word sequences
        return 1.0 if _norm_text(ans) and \
            _norm_text(ans) == _norm_text(meta.get("blank_ans", "")) else 0.0
    # synthetic: letter / integer-tuple / mapping, by what the gold looks like
    gold = str(private.answer).strip()
    if re.fullmatch(r"[A-Za-z]", gold):
        m = re.search(r"\b([A-Z])\b", ans.upper())
        return 1.0 if m and m.group(1) == gold.upper() else 0.0
    if re.search(r"\d\s*[-=:]\s*[A-Za-z]", gold):
        return 1.0 if _norm_mapping(ans) == _norm_mapping(gold) else 0.0
    want = _norm_numbers(gold)
    return 1.0 if want and _norm_numbers(ans) == want else 0.0
