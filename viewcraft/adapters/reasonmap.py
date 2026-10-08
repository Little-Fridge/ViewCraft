"""ReasonMap adapter (FSCCS/ReasonMap) -- route planning on transit maps.

THE TASK. One VERY large transit map (oslo 1682x1551 up to kl 14032x19843 --
the overview budget cannot make station labels legible, so zooming is
load-bearing on every item) plus "how do I get from station A to station B".
The answer is a multi-leg route in the benchmark's own template:

    Route Name: Line 1
    Departure Stop: xx Station
    Arrival Stop: yy Station
    --
    Route Name: Line 5
    ...

SCORING follows the official main.py, two components, no LLM judge:
  * acc (0/1): the route is VALID -- first leg departs station_1, last leg
    arrives station_2, every named line exists in the city topology, both
    endpoints of a leg lie on that line, and consecutive legs connect.
    Topology comes from the parquet's own `json` column (the same data as
    the repo's stations/<country>/<city>.json), never shown to the model.
  * map_api_score: 1 pt for correct endpoints, then per leg 2 pts line name
    + 1 departure + 1 arrival, zipped against the gold `routes`, capped at
    10. The dataset ships exactly one gold plan per item.
  The official short-task formula (min(10, map_api) + 10*acc)/20 has a
  per-item ceiling that varies with leg count (a 1-leg gold scores 0.8),
  which would make per-item scores incommensurable. The score here is
  therefore NORMALISED by the gold answer's own official score, computed at
  load time with the identical code path: gold = 1.0 on every item, and an
  invalid route keeps at most its partial map_api credit.

GROUP KEY IS THE CITY. Each city is exactly one map (verified byte-identical
per city), shared by ~10-40 questions; the same map must never straddle
splits, or a strategy could tune on a map it is later measured on.

ONLY question_short is used. question_long additionally asks for via-stop
counts/lists; its scoring (per-stop credit, China-only via-list IoU) is a
second metric, not a harder version of the same one, and mixing them in one
aggregate is the VSI mistake the `_scorer_label` split exists to prevent.

PIL NOTE: kl's map is 278M pixels -- past Pillow's decompression-bomb limit,
which is raised at import here because rendering goes through Image.open in
the same process. The official repo does the same.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None     # kl 14032x19843 = 278M px

DOMAIN = "reasonmap"

TRANSFER_SUFFIX = re.compile(r"（[^）]*）")


def _clean(s: str) -> str:
    """The official clean_string, plus station-suffix tolerance.

    The parquet mixes '石河北海站' (question spelling) with '石河北海' (gold
    route and topology spelling); English rows mix 'X Station' with 'X'.
    Dropping the suffix is what makes a verbatim gold answer score 1.0.
    """
    s = str(s or "").lower()
    s = re.sub(r"metro", "", s)
    s = re.sub(r"[,*]|（.*?）|\(.*?\)", "", s)
    s = re.sub(r"[\s\-]+", "", s)
    s = re.sub(r"(站|station)$", "", s)
    return s


def _line_key(name: str) -> str:
    """Normalise a line name to its distinguishing token.

    '地铁5号线(后关--虎滩新区)' -> 'line5', 'Metro A Line' -> 'linea',
    'Line 1' -> 'line1', 'MRed2' -> 'red2' -> 'linered2' fallback below.
    """
    s = str(name or "").strip()
    m = re.search(r"(\d+)", s)
    if m:
        return f"line{m.group(1)}"
    m = re.search(r"\b([A-Za-z])\b", s)
    if m:
        return f"line{m.group(1).lower()}"
    return "line" + _clean(s)


def _parse_segments(answer: str) -> list[dict]:
    """Collect legs line-by-line; a new "Route Name:" starts a new leg.

    Line-by-line, not "--" splitting: Chinese line names CONTAIN "--"
    ("地铁13号线(普兰店振兴街--九里)"), so splitting on the segment separator
    breaks inside the name itself and every leg vanishes.
    """
    legs: list[dict] = []
    cur: dict | None = None
    for line in str(answer or "").splitlines():
        m = re.match(r"\s*Route Name:\s*(.+)", line)
        if m:
            cur = {"route": m.group(1).strip(), "dep": "", "arr": ""}
            legs.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"\s*Departure Stop:\s*(.+)", line)
        if m:
            cur["dep"] = m.group(1).strip()
            continue
        m = re.match(r"\s*Arrival Stop:\s*(.+)", line)
        if m:
            cur["arr"] = m.group(1).strip()
    return [l for l in legs if l["dep"] and l["arr"]]


def _topology(json_blob: str) -> dict:
    """line-key -> set of cleaned station names, from the parquet json column."""
    try:
        topo = json.loads(json_blob)
    except (TypeError, json.JSONDecodeError):
        return {}
    out = {}
    for line, stations in topo.items():
        out[_line_key(line)] = {_clean(TRANSFER_SUFFIX.sub("", str(st)))
                                for st in stations}
    return out


def _official_score(answer_text: str, station_1: str, station_2: str,
                    gold_routes: list, topo: dict) -> float:
    legs = _parse_segments(answer_text)
    if not legs:
        return 0.0
    s1, s2 = _clean(station_1), _clean(station_2)

    # acc: validity against the topology (no gold comparison here)
    acc = 1.0
    if _clean(legs[0]["dep"]) != s1 or _clean(legs[-1]["arr"]) != s2:
        acc = 0.0
    for a, b in zip(legs, legs[1:]):
        if _clean(a["arr"]) != _clean(b["dep"]):
            acc = 0.0
    for leg in legs:
        stops = topo.get(_line_key(leg["route"]))
        if stops is None or _clean(leg["dep"]) not in stops \
                or _clean(leg["arr"]) not in stops:
            acc = 0.0

    # map_api_score against the gold plans (exactly one per item in this
    # release, but the column is a dict of plans, so keep the loop honest)
    best = 0.0
    for plan in gold_routes:
        sc = 0.0
        if legs and plan:
            if _clean(legs[0]["dep"]) == _clean(plan[0].get("departure_stop")):
                sc += 1
            if _clean(legs[-1]["arr"]) == _clean(plan[-1].get("arrival_stop")):
                sc += 1
        for leg, g in zip(legs, plan):
            if _line_key(leg["route"]) == _line_key(g.get("route_name")):
                sc += 2
            if _clean(leg["dep"]) == _clean(g.get("departure_stop")):
                sc += 1
            if _clean(leg["arr"]) == _clean(g.get("arrival_stop")):
                sc += 1
        best = max(best, sc)
    return (min(10.0, best) + 10.0 * acc) / 20.0


def load(root: str | Path, seed: int = 20260911) -> list[Sample]:
    import pyarrow.parquet as pq
    root = Path(root)
    pqs = sorted(root.rglob("*.parquet"))
    if not pqs:
        raise FileNotFoundError(f"no parquet under {root}")
    samples: list[Sample] = []
    for p in pqs:
        for i, r in enumerate(pq.read_table(p).to_pylist()):
            fig = r.get("figure")
            if not fig:
                continue
            media = write_media(DOMAIN, base64.b64decode(fig),
                                f"{r['country']}_{r['city']}")
            routes = r.get("routes") or {}
            plans = [[dict(g) for g in plan]
                     for plan in routes.values() if plan]
            item_id = f"{DOMAIN}/{r['country']}_{r['city']}_{i:03d}"
            group = f"{r['country']}/{r['city']}"
            public = PublicItem(
                item_id=item_id, domain=DOMAIN,
                question=str(r["question_short"]).strip(),
                media_ids=(media,),
                meta={"question_type": "route_planning",
                      "category": f"{r['country']}/{r['city']}",
                      "difficulty_question": r.get("difficulty_question"),
                      "difficulty_city": r.get("difficulty_city"),
                      "question_transfer_count":
                          r.get("question_transfer_count")})
            private = PrivateItem(
                item_id=item_id,
                answer=json.dumps(plans, ensure_ascii=False),
                answer_type="route",
                evidence_source="official",
                meta={"station_1": r["station_1"], "station_2": r["station_2"],
                      "topology_json": r.get("json") or "",
                      "gold_routes": plans,
                      # gold's own raw score under the identical code path;
                      # the per-item normaliser (see module docstring)
                      "gold_raw": _official_score(
                          "\n--\n".join(
                              f"Route Name: {g['route_name']}\n"
                              f"Departure Stop: {g['departure_stop']}\n"
                              f"Arrival Stop: {g['arrival_stop']}"
                              for g in (plans[0] if plans else [])),
                          r["station_1"], r["station_2"], plans,
                          _topology(r.get("json") or "")),
                      "difficulty_question": r.get("difficulty_question")})
            samples.append(Sample(public=public, private=private,
                                  group_key=group))
    cities = sorted({s.group_key for s in samples})
    print(f"[{DOMAIN}] kept {len(samples)} questions over "
          f"{len(cities)} city maps")
    return samples


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    meta = private.meta or {}
    topo = _topology(meta.get("topology_json", ""))
    raw = _official_score(answer_text, meta.get("station_1", ""),
                          meta.get("station_2", ""),
                          meta.get("gold_routes") or [], topo)
    gold_raw = float(meta.get("gold_raw") or 0.0)
    if gold_raw <= 0.0:
        return 0.0
    return min(1.0, raw / gold_raw)
