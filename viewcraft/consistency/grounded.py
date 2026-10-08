"""A0(q): requirement extraction, and C_ground: evidence support estimation.

A0 is produced once per question by a frozen prompt, from the question and the
allowed inputs only — never from the answer or from the evolving skill. It is
cached and shared, so no arm gets a hidden extra hint.

C_ground asks the executor model, in a separate request, whether each
requirement is supported by the images that were ACTUALLY sent. The verifier
never sees the final answer or the gold answer, so it cannot score by
agreement. Support is binary (1 for supported, 0 otherwise); unknown rate is
reported separately. Each requirement is verified exactly once — no repeated
voting.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..models import CFG, ModelClient, ModelFormatError, user_message
from ..schema import (GroundingResult, PublicItem, Requirement, Trajectory,
                      sha)

# A reasoning executor spends its completion allowance on hidden reasoning
# tokens before emitting any text; 512-1024 left nothing for the answer and
# produced empty replies. Sized from the config so a non-reasoning model does
# not pay for headroom it never uses.
ANSWER_TOKENS = int(
    (CFG.get("models", {}).get("executor", {}) or {}).get(
        "answer_max_tokens", 1024))

REQUIREMENTS_PROMPT = """\
You are analysing what visual evidence a question needs, before anyone answers it.

List at most three atomic pieces of information that must be VISIBLE in the
image(s) for this question to be answerable. Use only the question. Do not
answer it and do not pad the list.

Question: {question}
{options_block}
Reply with a JSON array of objects, each with:
  "text": what must be visible, stated concretely
  "kind": "visual" for something that must be seen, "derived" for something
          computed from visible facts

Example shape:
[{{"text": "the colour of the van's body panels", "kind": "visual"}}]
"""

VERIFIER_PROMPT = """\
You are checking whether specific pieces of information are visible in the
images provided. You are NOT answering any question and you will not be shown
any answer.

For each requirement below, judge only from the images shown:
  "supported"   - clearly visible and readable in the images
  "unsupported" - absent, unreadable, cut off, or not established by these images

Requirements:
{requirements_block}

Reply with a JSON object mapping each requirement id to one label, e.g.
{{"r1": "supported", "r2": "unsupported"}}
"""


class RequirementCache:
    """Content-addressed A0 cache keyed by question + allowed inputs."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, item: PublicItem) -> str:
        return sha({"question": item.question, "options": list(item.options),
                    "domain": item.domain, "prompt": REQUIREMENTS_PROMPT})

    def get(self, item: PublicItem) -> list[Requirement] | None:
        p = self.root / f"{self._key(item)}.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return [Requirement(**r) for r in d["requirements"]]

    def put(self, item: PublicItem, reqs: list[Requirement]) -> None:
        p = self.root / f"{self._key(item)}.json"
        p.write_text(json.dumps({
            "item_id": item.item_id,
            "requirements": [r.__dict__ for r in reqs],
        }, ensure_ascii=False, indent=2))


def _parse_json_block(text: str):
    """Pull the first JSON array or object out of a reply."""
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text or "", re.DOTALL)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text or "")
    for c in candidates:
        c = c.strip()
        for opener, closer in (("[", "]"), ("{", "}")):
            i, j = c.find(opener), c.rfind(closer)
            if i >= 0 and j > i:
                try:
                    return json.loads(c[i:j + 1])
                except json.JSONDecodeError:
                    continue
    return None


def extract_requirements(item: PublicItem, client: ModelClient,
                         cache: RequirementCache | None = None
                         ) -> list[Requirement]:
    """Run A0 once for this question. Raises on an empty or unparsable result."""
    if cache:
        hit = cache.get(item)
        if hit:
            return hit

    options_block = ""
    prompt = REQUIREMENTS_PROMPT.format(question=item.question,
                                        options_block=options_block)
    resp = client.call([user_message(prompt)], max_tokens=ANSWER_TOKENS)
    parsed = _parse_json_block(resp.content)
    if not isinstance(parsed, list) or not parsed:
        raise ModelFormatError(
            f"A0 extraction produced no requirements for {item.item_id}: "
            f"{resp.content[:200]!r}")

    reqs = []
    for i, obj in enumerate(parsed[:3]):
        if isinstance(obj, dict) and obj.get("text"):
            reqs.append(Requirement(req_id=f"r{i + 1}",
                                    text=str(obj["text"]),
                                    kind=str(obj.get("kind", "visual"))))
        elif isinstance(obj, str) and obj.strip():
            reqs.append(Requirement(req_id=f"r{i + 1}", text=obj.strip()))
    if not reqs:
        raise ModelFormatError(
            f"A0 extraction unparsable for {item.item_id}: "
            f"{resp.content[:200]!r}")

    if cache:
        cache.put(item, reqs)
    return reqs


VALID_LABELS = {"supported", "partial", "unsupported", "unknown"}


def verify_grounding(traj: Trajectory, requirements: list[Requirement],
                    client: ModelClient, view_data_urls: list[str],
                    chunk_size: int = 6) -> GroundingResult:
    """C_ground over the images actually sent.

    Large contexts are split into fixed, non-overlapping chunks of
    requirements and merged deterministically; every requirement is judged
    exactly once. A cheap notes-only judge is not used as a substitute — the
    real images go in the request.
    """
    if not requirements:
        raise ValueError("C_ground needs at least one requirement")
    if not view_data_urls:
        # nothing was sent: every requirement is unsupported by construction
        return GroundingResult.from_labels(
            {r.req_id: "unsupported" for r in requirements})

    labels: dict[str, str] = {}
    request_ids: list[str] = []

    chunks = [requirements[i:i + chunk_size]
              for i in range(0, len(requirements), chunk_size)]
    for chunk in chunks:
        block = "\n".join(f'  {r.req_id}: {r.text}' for r in chunk)
        prompt = VERIFIER_PROMPT.format(
            requirements_block=block,
            notes=(traj.notes or "none")[:600])
        resp = client.call([user_message(prompt, view_data_urls)],
                           max_tokens=ANSWER_TOKENS)
        request_ids.append(resp.request_id or "")
        parsed = _parse_json_block(resp.content)
        got = parsed if isinstance(parsed, dict) else {}
        for r in chunk:
            val = str(got.get(r.req_id, "unknown")).strip().lower()
            labels[r.req_id] = ("supported" if val == "supported" else
                                "unsupported" if val in VALID_LABELS else "unknown")

    return GroundingResult.from_labels(labels, tuple(request_ids))
