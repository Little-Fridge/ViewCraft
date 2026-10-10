"""User-task JSONL loading, local scorers, and configuration validation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from .adapters.registry import DomainSpec
from .schema import PrivateItem, PublicItem, Sample, sha
from .views.config import Coordinates


def normalized_edit_similarity(prediction: str, answer: str) -> float:
    """1 - normalized Levenshtein distance; lowercase, strip outer whitespace."""
    a, b = str(prediction).strip().lower(), str(answer).strip().lower()
    if not a or not b:
        return float(a == b)
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (x != y)))
        previous = current
    return 1.0 - previous[-1] / max(len(a), len(b))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(value: str, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else relative_to / path).resolve()


def load_task(path: Path) -> tuple[dict, list[Sample], Coordinates, DomainSpec, dict]:
    """Load a task without touching any model service or creating run files."""
    path = Path(path).resolve()
    task = yaml.safe_load(path.read_text())
    if not isinstance(task, dict):
        raise ValueError("Task YAML must contain a mapping")
    for key in ("data", "initial_skill"):
        if not isinstance(task.get(key), str):
            raise ValueError(f"Task YAML requires {key}: <path>")
    data = resolve_path(task["data"], path.parent)
    initial = Coordinates.read_snapshot(resolve_path(task["initial_skill"], path.parent))
    from .evolution.chain import Chain
    from .evolution.propose import Candidate
    Chain._assert_expressions_parse(None, Candidate("joint", initial, "initial skill", "initial"))
    answer_style = task.get("answer_style", "short_text")
    if answer_style not in ("short_text", "free_text", "letter", "yes_no", "point"):
        raise ValueError("answer_style must be short_text, free_text, letter, yes_no, or point")
    budget = task.setdefault("budget", {"observations": 6, "max_images_per_request": 4})
    for key in ("observations", "max_images_per_request"):
        if type(budget.get(key)) is not int or budget[key] < 1:
            raise ValueError(f"budget.{key} must be a positive integer")
    evolution = task.setdefault("evolution", {})
    for key, default in (("rounds", 4), ("train_batch_size", 24),
                         ("max_submissions_per_round", 4),
                         ("max_gate_evaluations_per_round", 2)):
        evolution.setdefault(key, default)
        if type(evolution[key]) is not int or evolution[key] < 1:
            raise ValueError(f"evolution.{key} must be a positive integer")
    for key in ("gate_question_cap", "test_question_cap"):
        evolution.setdefault(key, 0)
        if type(evolution[key]) is not int or evolution[key] < 0:
            raise ValueError(f"evolution.{key} must be a nonnegative integer")
    for key in ("strategy_book", "viewcraft_visual_fallback"):
        evolution.setdefault(key, True)
        if not isinstance(evolution[key], bool):
            raise ValueError(f"evolution.{key} must be true or false")
    task.setdefault("seed", 42)
    if type(task["seed"]) is not int:
        raise ValueError("seed must be an integer")

    samples, media_hashes = load_jsonl(data, task)
    scorer, scorer_identity = load_scorer(task.get("scorer", "exact_match"), path.parent)
    # A scorer's name is its identity in saved trajectories.
    spec = DomainSpec(name="custom", data_root=data.parent,
                      answer_style=answer_style, scorer_name=scorer_identity,
                      loader=lambda root: samples, scorer=scorer)
    manifest = {
        "task": task, "data_sha256": file_hash(data),
        "media_sha256": media_hashes, "initial_config_hash": initial.config_hash,
        "scorer": scorer_identity,
        "split_counts": dict(Counter(s.split for s in samples)),
        "split_groups": {sp: len({s.group_key for s in samples if s.split == sp})
                         for sp in ("train", "gate", "test")},
        "assignment": {s.public.item_id: s.split for s in samples},
    }
    return task, samples, initial, spec, manifest


def load_scorer(value: str, directory: Path):
    """Built-in scorer or an explicitly supplied local file.py:function."""
    if value == "exact_match":
        raw = lambda prediction, answer, item: float(str(prediction).strip().casefold() == str(answer).strip().casefold())
        identity = "custom_exact_match_v1"
    elif value == "edit_similarity":
        raw = lambda prediction, answer, item: normalized_edit_similarity(prediction, answer)
        identity = "custom_edit_similarity_v1"
    else:
        if not isinstance(value, str) or ":" not in value:
            raise ValueError("scorer must be exact_match, edit_similarity, or file.py:function")
        filename, function = value.rsplit(":", 1)
        source = resolve_path(filename, directory)
        module_spec = importlib.util.spec_from_file_location("viewcraft_user_scorer", source)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"Cannot load scorer: {source}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        raw = getattr(module, function)
        if not callable(raw):
            raise ValueError(f"Scorer is not callable: {value}")
        identity = f"custom_{function}_{file_hash(source)[:12]}"

    def score(prediction, private, public):
        result = float(raw(prediction, private.answer, public))
        if not math.isfinite(result) or not 0 <= result <= 1:
            raise ValueError(f"Scorer must return a finite value in [0, 1], got {result}")
        return result
    return score, identity


def load_jsonl(path: Path, task: dict) -> tuple[list[Sample], dict]:
    samples, media_hashes = [], {}
    ids, media_groups, group_splits = set(), {}, {}
    explicit = []
    allowed = {"id", "question", "images", "answer", "group_key", "split", "options", "meta"}
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) - allowed:
                raise ValueError(f"Expected an object with fields {sorted(allowed)}")
            for key in ("id", "question", "group_key"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{key} must be a nonempty string")
            if "answer" not in row or row["answer"] is None:
                raise ValueError("answer is required for evolution and evaluation")
            if row["id"] in ids:
                raise ValueError(f"Duplicate id: {row['id']}")
            ids.add(row["id"])
            images = row.get("images")
            if not isinstance(images, list) or not images or not all(isinstance(v, str) for v in images):
                raise ValueError("images must be a nonempty list of local image paths")
            media = tuple(str(resolve_path(v, path.parent)) for v in images)
            for filename in media:
                if filename not in media_hashes:
                    with Image.open(filename) as im:
                        im.verify()
                    media_hashes[filename] = file_hash(Path(filename))
                digest = media_hashes[filename]
                if digest in media_groups and media_groups[digest] != row["group_key"]:
                    raise ValueError("The same image must use the same group_key in all rows")
                media_groups[digest] = row["group_key"]
            split = row.get("split", "")
            if split and split not in ("train", "gate", "test"):
                raise ValueError("split must be train, gate, or test")
            explicit.append(bool(split))
            if row["group_key"] in group_splits and group_splits[row["group_key"]] != split:
                raise ValueError("A group_key cannot span multiple splits")
            group_splits[row["group_key"]] = split
            options = row.get("options", [])
            if not isinstance(options, list) or not all(isinstance(v, str) for v in options):
                raise ValueError("options must be a list of strings")
            meta = row.get("meta", {})
            if not isinstance(meta, dict) or set(meta) - {"frame_times_s", "duration_s"}:
                raise ValueError("meta supports only frame_times_s and duration_s")
            if "frame_times_s" in meta:
                times = meta["frame_times_s"]
                if (not isinstance(times, list) or len(times) != len(media)
                        or any(not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 for t in times)
                        or times != sorted(times)):
                    raise ValueError("frame_times_s must be sorted nonnegative times matching images")
            if "duration_s" in meta and (not isinstance(meta["duration_s"], (int, float))
                    or not math.isfinite(meta["duration_s"]) or meta["duration_s"] < 0):
                raise ValueError("duration_s must be finite and nonnegative")
            # File-safe internal IDs; original IDs remain in the manifest via item_id.
            item_id = "custom/" + sha(row["id"])[:20]
            samples.append(Sample(
                PublicItem(item_id, "custom", row["question"], media, tuple(options), meta=meta),
                PrivateItem(item_id, row["answer"], evidence_source="none", meta={"source_id": row["id"]}),
                split, row["group_key"]))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not samples:
        raise ValueError("Dataset is empty")
    if any(explicit) and not all(explicit):
        raise ValueError("Provide split on every row, or omit it on every row")
    if not any(explicit):
        ratios = task.get("split_ratios", [0.6, 0.2, 0.2])
        if (not isinstance(ratios, list) or len(ratios) != 3
                or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in ratios)
                or not math.isclose(sum(ratios), 1.0)):
            raise ValueError("split_ratios must be three positive numbers summing to 1")
        groups = sorted(group_splits)
        random.Random(task["seed"]).shuffle(groups)
        a, b = int(len(groups) * ratios[0]), int(len(groups) * ratios[1])
        assignment = dict(zip(groups, ["train"] * a + ["gate"] * b + ["test"] * (len(groups) - a - b)))
        samples = [Sample(s.public, s.private, assignment[s.group_key], s.group_key) for s in samples]
    for split in ("train", "gate", "test"):
        if not any(s.split == split for s in samples):
            raise ValueError(f"{split} split is empty; add groups or provide explicit splits")
    return samples, media_hashes
