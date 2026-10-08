"""FUNSD adapter (nielsr/funsd) -- DENSE FORM perception: many small labeled
fields in a noisy scanned form. The model must LOCATE a field label among
clutter and READ its value -- a reading-order + fine-text regime (zoom into the
field, track which fields are read), distinct from counting / change-detection.

FUNSD tags: 1/2 = header, 3/4 = question (the field label), 5/6 = answer (the
value), 0 = other. Each item asks for the value of one field; the answer is the
linked answer entity (the ANSWER-tagged words nearest the field label).
"""
from __future__ import annotations

import ast
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "funsd"


def _group_entities(words, tags, bboxes):
    """Group consecutive same-entity words into (text, tag, bbox) entities."""
    ents = []
    cur, cur_tag, boxes = [], None, []
    for w, t, b in zip(words, tags, bboxes):
        kind = ("q" if t in (3, 4) else "a" if t in (5, 6) else "h" if t in (1, 2)
                else "o")
        if kind == cur_tag and t not in (1, 3, 5):
            cur.append(w); boxes.append(b)
        else:
            if cur_tag and cur:
                ents.append((cur_tag, " ".join(cur), boxes))
            cur, cur_tag, boxes = ([w] if kind in ("q", "a") else []), \
                                  (kind if kind in ("q", "a") else None), \
                                  ([b] if kind in ("q", "a") else [])
    if cur_tag and cur:
        ents.append((cur_tag, " ".join(cur), boxes))
    return ents


def load(root: str | Path, seed: int = 20260911) -> list[Sample]:
    import pyarrow.parquet as pq
    root = Path(root)
    pqs = sorted(root.rglob("*.parquet"))
    if not pqs:
        raise FileNotFoundError(f"no parquet under {root}")
    samples: list[Sample] = []
    for p in pqs:
        # Row indices repeat across parquet files, so namespace them by file
        # role: the official FUNSD test parquet gets "t"-prefixed group keys,
        # the train parquet keeps bare indices (frozen plans and ledgers that
        # predate the test parquet stay valid).
        pfx = "t" if "test" in p.stem else ""
        for i, r in enumerate(pq.read_table(p).to_pylist()):
            words = ast.literal_eval(r["words"]) if isinstance(r["words"], str) else r["words"]
            tags = ast.literal_eval(r["ner_tags"]) if isinstance(r["ner_tags"], str) else r["ner_tags"]
            bboxes = ast.literal_eval(r["bboxes"]) if isinstance(r["bboxes"], str) else r["bboxes"]
            blob = r.get("image")
            blob = blob.get("bytes") if isinstance(blob, dict) else blob
            if not (words and tags and bboxes and blob):
                continue
            ents = _group_entities(words, tags, bboxes)
            qs = [e for e in ents if e[0] == "q"]
            ans = [e for e in ents if e[0] == "a"]
            # link each field to the nearest answer entity by bbox y-overlap
            def cy(e):
                return sum(b[1] for b in e[2]) / len(e[2])
            for qi, (kind, label, qboxes) in enumerate(qs):
                if not label.strip() or not ans:
                    continue
                # nearest answer entity below/beside the label
                best = min(ans, key=lambda a: abs(cy(a) - cy((kind, label, qboxes))))
                value = best[1].strip()
                if not value or value == label:
                    continue
                media = write_media(DOMAIN, blob, f"{pfx}{i}")
                item_id = f"{DOMAIN}/{pfx}{i}_{qi}"
                public = PublicItem(
                    item_id=item_id, domain=DOMAIN,
                    question=(f"In this form, what is the value of the field "
                              f"\"{label}\"? Answer with the text only."),
                    media_ids=(media,), meta={})
                private = PrivateItem(item_id=item_id, answer=value,
                                      answer_type="string",
                                      evidence_source="none", meta={})
                samples.append(Sample(public=public, private=private,
                                      group_key=f"{pfx}{i}"))
    print(f"[funsd] kept {len(samples)} field-value items")
    return samples


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    """Lowercased field-value normalized edit similarity (paper Appendix A)."""
    from ..custom import normalized_edit_similarity
    return normalized_edit_similarity(answer_text, str(private.answer))
