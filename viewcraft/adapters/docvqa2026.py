"""DocVQA-2026 adapter (VLR-CVC/DocVQA-2026, val split, public answers).

ICDAR 2026 competition benchmark: reasoning questions over multi-page
documents across 8 domains (business reports, scientific papers, slides,
posters, maps, comics, infographics, engineering drawings). The val split is
25 documents / 80 questions (<1000, so the full set is run; the test split
has withheld answers and is never touched here).

SCORING: the competition's `model_graded_fact` -- an LLM judge decides
whether the model's answer is factually correct given the question and the
gold answer. The judge here is the OPTIMIZER client (never the executor),
same independence rule as clbench: the judge reads no pixels and no
trajectory, only (question, gold, answer).

PAGE POOL: pages are materialised once via write_media; documents over 128
pages are uniformly subsampled across the whole document (the mmlongdoc
scheme), so a capped pool still spans front to back. group_key is the
document: all questions of a document share its pages and must never
straddle splits.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..schema import PrivateItem, PublicItem, Sample
from .media import write_media

DOMAIN = "docvqa2026"
MAX_POOL_PAGES = 128


def uniform_page_indices(n_pages: int, n_sample: int) -> list[int]:
    if n_pages <= n_sample:
        return list(range(n_pages))
    return sorted({round(i * (n_pages - 1) / (n_sample - 1))
                   for i in range(n_sample)})


_JUDGE_CLIENT = None


def _judge():
    global _JUDGE_CLIENT
    if _JUDGE_CLIENT is None:
        from ..models import ModelClient
        _JUDGE_CLIENT = ModelClient("optimizer")
    return _JUDGE_CLIENT


def load(root: str | Path, seed: int = 20260911) -> list[Sample]:
    import pyarrow.parquet as pq
    root = Path(root)
    pqs = sorted(root.rglob("*.parquet"))
    if not pqs:
        raise FileNotFoundError(f"no parquet under {root}")
    samples: list[Sample] = []
    for p in pqs:
        for r in pq.read_table(p).to_pylist():
            doc_id = str(r["doc_id"])
            pages = r.get("document") or []
            keep = uniform_page_indices(len(pages), MAX_POOL_PAGES)
            paths = []
            for pi in keep:
                blob = pages[pi]
                blob = blob.get("bytes") if isinstance(blob, dict) else blob
                if blob:
                    paths.append(write_media(DOMAIN, blob,
                                             f"{doc_id}_p{pi:04d}"))
            if not paths:
                continue
            qs = r.get("questions") or {}
            ans = r.get("answers") or {}
            q_ids = qs.get("question_id") or []
            a_by_id = dict(zip(ans.get("question_id") or [],
                               ans.get("answer") or []))
            for qid, q in zip(q_ids, qs.get("question") or []):
                gold = str(a_by_id.get(qid, "")).strip()
                if not gold:
                    continue
                item_id = f"{DOMAIN}/{qid}"
                public = PublicItem(
                    item_id=item_id, domain=DOMAIN,
                    question=str(q).strip(), media_ids=tuple(paths),
                    meta={"question_type": "document_reasoning",
                          "category": r.get("doc_category"),
                          "n_pages": len(pages),
                          "n_pages_pooled": len(paths)})
                private = PrivateItem(
                    item_id=item_id, answer=gold, answer_type="string",
                    evidence_source="official",
                    meta={"category": r.get("doc_category")})
                samples.append(Sample(public=public, private=private,
                                      group_key=doc_id))
    print(f"[{DOMAIN}] kept {len(samples)} questions over "
          f"{len({s.group_key for s in samples})} documents")
    return samples


def score(answer_text: str, private: PrivateItem, public: PublicItem) -> float:
    """model_graded_fact: an independent judge marks the answer correct or not."""
    ans = (answer_text or "").strip()
    gold = str(private.answer).strip()
    if not ans or not gold:
        return 0.0
    q = public.question
    prompt = (
        "You are grading an answer to a document question. Decide whether "
        "the model's answer is FACTUALLY CORRECT given the gold answer. "
        "It is correct if it states the same fact, number or entity, even "
        "with different wording; it is incorrect if it is missing, "
        "contradicts, or adds a different fact.\n\n"
        f"QUESTION: {q[:800]}\nGOLD ANSWER: {gold[:800]}\n"
        f"MODEL ANSWER: {ans[:2000]}\n\n"
        'Reply with a JSON object: {"correct": true} or {"correct": false}')
    try:
        from ..models import user_message
        resp = _judge().call([user_message(prompt)], max_tokens=256,
                             use_cache=False)
        m = re.search(r"\{.*\}", resp.content or "", re.DOTALL)
        obj = json.loads(m.group(0)) if m else {}
        return 1.0 if obj.get("correct") is True else 0.0
    except Exception:                                       # noqa: BLE001
        return 0.0
