"""Synthetic LOGIC-GRID (zebra) puzzle domain (syn_zebra) -- the user's
challenge: a NON-visual task that a single reasoning pass cannot solve
perfectly, so the visual workspace (filling the grid) is load-bearing.

Why this is the right test, and how it avoids the earlier traps. syn_state /
syn_math / clbench were all solvable in ONE reasoning pass (read it, answer
it), so the workspace never engaged. A logic-grid puzzle is different: N
houses x several categories, and the only way to keep the cross-constraints
straight is to WRITE the grid down -- humans cannot do it in their head either.
The grid IS the natural workspace, and it maps directly onto the canvas: the
model marks which value goes in which cell via notes.

The image is an empty grid (categories x houses). The clues are text. The
model must fill the grid (notes) to track its deductions, then answer. A
brute-force DFS solver guarantees each generated puzzle has a UNIQUE solution.
Difficulty is dialled by house count. One image per item; zero download.
"""
from __future__ import annotations

import hashlib
from itertools import permutations
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample

DOMAIN = "syn_zebra"

_CATS = {
    "color": ["red", "green", "blue", "yellow", "white", "black"],
    "pet": ["cat", "dog", "bird", "fish", "horse", "fox"],
    "drink": ["tea", "coffee", "milk", "water", "juice", "soda"],
    "job": ["chef", "nurse", "pilot", "artist", "judge", "guard"],
}


def _rng_for(item_index: int, seed: int):
    import random
    h = int(hashlib.sha256(f"{seed}:zebra:{item_index}".encode()).hexdigest(), 16)
    return random.Random(h % (2 ** 32))


def _solutions(assign_clues, n, cat_names):
    """All assignments consistent with the clues, via a PRUNING DFS.

    The naive itertools.product over every category's permutations is 120**3
    for 5 houses -- unusable. Assigning one category at a time and pruning on
    any clue that only references already-fixed categories cuts the search by
    orders of magnitude. Stops at 2 solutions (enough to test uniqueness).
    """
    def consistent(clue, assign):
        # a clue only constrains categories that are all assigned
        refs = {clue[1], clue[3]} if clue[0] in ("same", "left", "notsame") \
            else {clue[1]}
        if not refs.issubset(assign):
            return True                     # cannot judge yet
        kind = clue[0]
        if kind == "pos":
            _, c, v, i = clue
            return assign[c][i] == v
        _, ca, va, cb, vb = clue
        ha = assign[ca].index(va)
        hb = assign[cb].index(vb)
        if kind == "same":
            return ha == hb
        if kind == "left":
            return hb - ha == 1
        return ha != hb                     # notsame

    sols = []
    cat_list = list(cat_names)

    def dfs(k, assign):
        if len(sols) > 1:
            return
        if k == len(cat_list):
            if all(consistent(cl, assign) for cl in assign_clues):
                sols.append(dict(assign))
            return
        c = cat_list[k]
        for perm in permutations(_CATS[c][:n]):
            assign[c] = list(perm)
            # prune: any clue whose categories are now all assigned must hold
            if all(consistent(cl, assign) for cl in assign_clues):
                dfs(k + 1, assign)
            del assign[c]

    dfs(0, {})
    return sols


def _gen_puzzle(rng, n):
    """Generate a uniquely-solvable zebra puzzle; returns (solution, clues).

    Work BACKWARD from the solution: build a pool of true clues (weighted
    toward relational ones, so the model must chain rather than read off), then
    add them one at a time until the solver finds exactly one solution. This
    guarantees a unique answer and a genuinely deductive clue set -- unlike
    scattering a few random clues, which almost never pins down 120**3
    assignments for 5 houses x 3 categories.
    """
    cat_names = list(_CATS)
    for _ in range(60):
        solution = {c: list(rng.sample(_CATS[c][:n], n)) for c in cat_names}
        pool = []
        for ca in cat_names:
            for cb in cat_names:
                if ca >= cb:
                    continue
                for h in range(n):
                    pool.append(("same", ca, solution[ca][h], cb,
                                 solution[cb][h]))
                    if h + 1 < n:
                        pool.append(("left", ca, solution[ca][h], cb,
                                     solution[cb][h + 1]))
        rng.shuffle(pool)
        # a couple of direct positional anchors keep the puzzle readable
        clues = []
        for _ in range(rng.randint(1, 2)):
            c = rng.choice(cat_names)
            i = rng.randrange(n)
            clues.append(("pos", c, solution[c][i], i))
        rest = list(pool)
        rng.shuffle(rest)
        for clue in rest:
            clues.append(clue)
            if len(_solutions(clues, n, cat_names)) == 1:
                return solution, clues
    return None, None


def _render(path: Path, n: int, cat_names: list[str]) -> None:
    from PIL import Image, ImageDraw
    cw, ch = 96, 56
    w, h = 130 + cw * n, 40 + ch * (len(cat_names) + 1)
    img = Image.new("RGB", (w, h), (252, 252, 250))
    d = ImageDraw.Draw(img)
    for i in range(n):
        d.text((150 + i * cw + cw // 2, 14), f"House {i + 1}", fill=(60, 60, 60))
    for r, cat in enumerate(cat_names):
        d.text((16, 46 + r * ch + ch // 2), cat, fill=(60, 60, 60))
        for i in range(n):
            x0, y0 = 140 + i * cw, 40 + r * ch
            d.rectangle([x0, y0, x0 + cw - 8, y0 + ch - 8],
                        outline=(70, 70, 70), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def load(root: str | Path, seed: int = 20260911, n_items: int = 240,
         n_houses: int = 5) -> list[Sample]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cat_names = list(_CATS)
    samples: list[Sample] = []
    for i in range(n_items):
        rng = _rng_for(i, seed)
        solution, clues = _gen_puzzle(rng, n_houses)
        if solution is None:
            continue
        # ask about a non-trivially-derivable fact: which house has a given value
        qc = rng.choice(cat_names)
        hv = rng.randrange(n_houses)
        qval = solution[qc][hv]
        answer = f"House {hv + 1}"
        path = root / f"zebra_{i:04d}_n{n_houses}.png"
        if not path.exists():
            _render(path, n_houses, cat_names)
        clue_lines = []
        for cl in clues:
            if cl[0] == "pos":
                clue_lines.append(f"- The {cl[1]} '{cl[2]}' is in house {cl[3] + 1}.")
            elif cl[0] == "same":
                clue_lines.append(
                    f"- The house with {cl[1]} '{cl[2]}' also has {cl[3]} '{cl[4]}'.")
            elif cl[0] == "left":
                clue_lines.append(
                    f"- The house with {cl[1]} '{cl[2]}' is immediately to the "
                    f"left of the house with {cl[3]} '{cl[4]}'.")
            else:
                clue_lines.append(
                    f"- The house with {cl[1]} '{cl[2]}' is NOT the house with "
                    f"{cl[3]} '{cl[4]}'.")
        item_id = f"{DOMAIN}/{i:04d}"
        question = (
            f"There are {n_houses} houses in a row, numbered 1 to {n_houses} "
            f"from left to right. Each house has a color, a pet and a drink, "
            f"and within a category no two houses share a value. The grid image "
            f"has one row per category and one column per house.\n"
            f"Clues:\n" + "\n".join(clue_lines) +
            f"\n\nWhich house has the {qc} '{qval}'? Answer with the house "
            f"number only.")
        public = PublicItem(item_id=item_id, domain=DOMAIN, question=question,
                            media_ids=(str(path),),
                            meta={"n_houses": n_houses})
        private = PrivateItem(item_id=item_id, answer=str(hv + 1),
                              answer_type="integer", evidence_source="none",
                              meta={"solution": {c: solution[c] for c in cat_names},
                                    "clues": [str(c) for c in clues],
                                    "answer_house": hv + 1})
        samples.append(Sample(public=public, private=private,
                              group_key=path.stem))
    print(f"[zebra] generated {len(samples)} unique-solution puzzles "
          f"(n_houses={n_houses})")
    return samples


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    import re
    m = re.search(r"(\d+)", answer_text or "")
    return 1.0 if m and int(m.group(1)) == int(private.meta["answer_house"]) else 0.0
