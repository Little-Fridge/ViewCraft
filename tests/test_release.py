"""Release regression checks. All model traffic uses local test doubles."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def task_copy(tmp_path):
    dst = tmp_path / "example"
    shutil.copytree(ROOT / "examples/counting", dst)
    return dst


def test_custom_validation_and_scorer(task_copy):
    from viewcraft.custom import load_task, load_scorer, normalized_edit_similarity
    _, samples, coords, _, manifest = load_task(task_copy / "task.yaml")
    assert manifest["split_counts"] == {"train": 6, "gate": 3, "test": 3}
    assert not hasattr(samples[0].public, "answer")
    scorer, _ = load_scorer("score.py:score", task_copy)
    assert scorer(samples[0].private.answer, samples[0].private, samples[0].public) == 1
    assert normalized_edit_similarity("1234", "123") == 0.75
    assert normalized_edit_similarity("124", "123") == pytest.approx(2 / 3)
    coords.write_snapshot(task_copy / "roundtrip")
    from viewcraft.views.config import Coordinates
    assert Coordinates.read_snapshot(task_copy / "roundtrip").config_hash == coords.config_hash


@pytest.mark.parametrize("problem", ["duplicate_id", "split_leak", "duplicate_media", "missing_image", "mixed_split", "private_meta"])
def test_bad_data_stops_before_api(task_copy, problem):
    from viewcraft.custom import load_task
    p = task_copy / "data.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    if problem == "duplicate_id":
        rows[1]["id"] = rows[0]["id"]
    elif problem == "split_leak":
        rows[6]["group_key"] = rows[0]["group_key"]
    elif problem == "duplicate_media":
        rows[6]["images"] = rows[0]["images"]
    elif problem == "missing_image":
        rows[0]["images"] = ["absent.png"]
    elif problem == "mixed_split":
        del rows[0]["split"]
    elif problem == "private_meta":
        rows[0]["meta"] = {"answer": "leak"}
    p.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        load_task(task_copy / "task.yaml")


def test_group_splits_are_deterministic(task_copy):
    from viewcraft.custom import load_task
    p = task_copy / "data.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    for row in rows:
        row.pop("split")
    p.write_text("\n".join(json.dumps(row) for row in rows))
    a = load_task(task_copy / "task.yaml")[4]["assignment"]
    b = load_task(task_copy / "task.yaml")[4]["assignment"]
    assert a == b and set(a.values()) == {"train", "gate", "test"}


def test_standard_auth_and_both_endpoint_overrides(monkeypatch, tmp_path):
    from viewcraft import models
    for key in ("AIHUB_APP_ID", "AIHUB_API_KEY", "APP_ID", "APP_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-token")
    monkeypatch.setenv("VIEWCRAFT_ENDPOINT", "https://example.invalid/v1/chat/completions")
    monkeypatch.setattr(models, "limiter_for", lambda *args: SimpleNamespace(acquire=lambda: None))
    cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    for role in ("executor", "optimizer"):
        client = models.ModelClient(role, cfg, tmp_path / role)
        url, headers = client._url_and_headers()
        assert url == os.environ["VIEWCRAFT_ENDPOINT"]
        assert headers["Authorization"] == "Bearer dummy-token"
    cfg["models"]["executor"]["auth"] = "none"
    client = models.ModelClient("executor", cfg, tmp_path / "local")
    assert "Authorization" not in client._url_and_headers()[1]


def test_projection_conditions_and_gate():
    from viewcraft.evolution.gate import ArmResults, GateDecision, decide, projection_eligible
    def result(score, ground):
        return ArmResults(score, ground, 1, True, True, per_item_scores={"a": score})
    assert decide("viewcraft", result(0, 1), result(1, 0)).accepted
    assert decide("viewcraft", result(0, 0), result(0, 1)).accepted
    assert not decide("viewcraft", result(1, 1), result(0, 1)).accepted
    good = GateDecision(False, "loss", -0.1, 0, 0.1)
    assert projection_eligible(good, 1, 4, 1, 2)
    assert not projection_eligible(good, 4, 4, 1, 2)
    assert not projection_eligible(good, 1, 4, 2, 2)
    assert not projection_eligible(GateDecision(False, "tie", 0, 0, 0.1), 1, 4, 1, 2)
    assert not projection_eligible(GateDecision(False, "support loss", -0.1, -0.1, 0.1), 1, 4, 1, 2)


def test_verification_withholds_options_and_notes():
    from viewcraft.consistency.grounded import extract_requirements, verify_grounding
    from viewcraft.schema import PublicItem, Trajectory
    prompts = []
    class Fake:
        def call(self, messages, **kwargs):
            prompts.append(str(messages))
            content = json.dumps([{"text": f"requirement {i}"} for i in range(5)]) if len(prompts) == 1 else '{"r1":"supported","r2":"partial","r3":"unsupported"}'
            return SimpleNamespace(content=content, request_id="offline")
    client = Fake()
    item = PublicItem("i", "custom", "Read the image", options=("SECRET_OPTION",))
    reqs = extract_requirements(item, client)
    assert len(reqs) == 3 and "SECRET_OPTION" not in prompts[0]
    traj = Trajectory("i", "custom", "gate", "viewcraft", "normal", "c", "p", notes="SECRET_NOTE")
    result = verify_grounding(traj, reqs, client, ["data:image/png;base64,AA=="])
    assert result.score == pytest.approx(1 / 3)
    assert "SECRET_NOTE" not in prompts[1]


def test_cache_separates_endpoints(monkeypatch, tmp_path):
    from viewcraft import models
    monkeypatch.delenv("VIEWCRAFT_ENDPOINT", raising=False)
    monkeypatch.setattr(models, "limiter_for", lambda *args: SimpleNamespace(acquire=lambda: None))
    cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    cfg["models"]["executor"]["auth"] = "none"
    calls = []
    def post(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(status_code=200, headers={}, json=lambda: {"choices":[{"message":{"content":"answer"}}]})
    monkeypatch.setattr(models.requests, "post", post)
    for endpoint in ("https://one.invalid", "https://two.invalid", "https://two.invalid"):
        cfg["endpoints"]["standard"] = endpoint
        client = models.ModelClient("executor", copy.deepcopy(cfg), tmp_path / "cache")
        client.call([{"role": "user", "content": "same question"}])
    assert calls == ["https://one.invalid", "https://two.invalid"]


def test_initial_expressions_and_rendering():
    from PIL import Image
    from viewcraft.views.layout import eval_layout_expr
    from viewcraft.views.selection import eval_selection_expr
    from viewcraft.views.exprs import eval_pixel_expr, pixel_env
    from viewcraft.views.zoomgate import eval_zoom_gate
    assert len(eval_selection_expr("uniform(budget)", 12, {}, 4)) == 4
    im = Image.new("RGB", (100, 80), "red")
    assert eval_layout_expr("grid(views, cols=2, label=index)", [im, im], {})
    assert eval_pixel_expr("rgb(red, green, blue)", pixel_env(im, {})) is not None
    assert eval_zoom_gate("zoom_gain > 1.05 or region_frac < 0.12", {"zoom_gain": 2, "region_frac": 0.2})


@pytest.mark.parametrize("protocol", ["", "Inspect the image and finish when ready."])
def test_custom_contracts_keep_executable_answer_action(protocol):
    from viewcraft.runtime import build_task_prompt, parse_model_turn
    from viewcraft.schema import PublicItem
    from viewcraft.views.config import Coordinates, PsiConfig
    coords = Coordinates(psi=PsiConfig(
        answer_contract="Return the integer only in the answer field.",
        protocol_contract=protocol))
    prompt = build_task_prompt(PublicItem("i", "custom", "Count circles"),
                               coords, 2, answer_style="short_text")
    assert '"action": "answer"' in prompt
    assert "Answer action format:" in prompt
    assert '"answer" field' in prompt
    assert parse_model_turn('{"action":"answer","answer":"3"}', 0).answer == "3"


@pytest.mark.parametrize("proposal_kind", ["reader", "joint"])
def test_offline_cli_evolve_resume_test_predict(task_copy, tmp_path, proposal_kind):
    """Real CLI/chain/rendering/ledger, with only HTTP inference replaced."""
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text('''
import json, os
from pathlib import Path
import requests
def post(url, **kw):
    payload = kw["json"]
    messages = payload["messages"]
    text = str(messages)
    with open(os.environ["MOCK_CALLS"], "a") as f:
        f.write(payload["model"] + "\\n")
    if payload["model"] == "offline-optimizer":
        if "strat" in text.lower() and "prevalence" in text and "Current skill" not in text and "cur_skill" not in text and "JOINT" not in text:
            content = json.dumps({"strategies": []})
        else:
            content = json.dumps({"skill": "Inspect all red circles carefully. OFFLINE_REVISED", "expr": "rgb(red, green * 0.9, blue)" if os.environ["MOCK_PROPOSAL"] == "joint" else "", "rationale": "Use a systematic scan."})
    elif "You are analysing what visual evidence" in text:
        content = '[{"text":"the red circles are visible","kind":"visual"}]'
    elif "You are checking whether specific pieces" in text:
        content = '{"r1":"supported"}'
    else:
        system = ' '.join(str(m.get("content", "")) for m in messages if m["role"] == "system")
        answer = "3" if "OFFLINE_REVISED" in system else "0"
        content = json.dumps({"action":"answer", "answer":answer})
    class Response:
        status_code = 200
        headers = {}
        def json(self):
            return {"id":"offline", "model":payload["model"], "choices":[{"message":{"content":content}}], "usage":{}}
    return Response()
requests.post = post
''')
    cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    for role in ("executor", "optimizer"):
        cfg["models"][role].update(resolved_id="offline-" + role, auth="none")
    cfg["throughput"]["default_provider_rpm"] = 0
    config = tmp_path / "models.yaml"
    config.write_text(yaml.safe_dump(cfg))
    task = yaml.safe_load((task_copy / "task.yaml").read_text())
    task["evolution"].update(rounds=1, strategy_book=False)
    (task_copy / "task.yaml").write_text(yaml.safe_dump(task))
    # Fixed targets exercise a known accepted edit; this is a software fixture.
    data = task_copy / "data.jsonl"
    rows = [json.loads(l) for l in data.read_text().splitlines()]
    for row in rows:
        row["answer"] = "3"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("VIEWCRAFT_"):
            env.pop(key)
    env.update(PYTHONPATH=str(hook), VIEWCRAFT_RUNS_ROOT=str(tmp_path / "runs"),
               VIEWCRAFT_LIMITER="process", MOCK_PROPOSAL=proposal_kind, MOCK_CALLS=str(tmp_path / "calls.txt"))
    def run(args, expected=0):
        p = subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=60)
        assert p.returncode == expected, p.stdout + p.stderr
        return p
    cmd = ["scripts/evolve.py", "--task", str(task_copy / "task.yaml"),
           "--config", str(config), "--run-id", "offline", "--workers", "1"]
    run(cmd + ["--check"])
    assert not (tmp_path / "calls.txt").exists()
    run(cmd)
    final = tmp_path / "runs/offline/arms/viewcraft/snapshots/final"
    assert "OFFLINE_REVISED" in (final / "SKILL.md").read_text()
    history = tmp_path / "runs/offline/arms/viewcraft/history.json"
    assert json.loads(history.read_text())[0]["decision"]["accepted"]
    before = (tmp_path / "calls.txt").read_text()
    round0 = (final.parent / "round00/SKILL.md").read_text()
    run(cmd)
    assert (tmp_path / "calls.txt").read_text() == before
    assert (final.parent / "round00/SKILL.md").read_text() == round0
    run(cmd + ["--stage", "test"])
    result = tmp_path / "runs/offline/metrics/test_rows.jsonl"
    assert len(result.read_text().splitlines()) == 3
    run(cmd + ["--stage", "test"])
    assert len(result.read_text().splitlines()) == 3
    pred = run(["scripts/predict.py", "--config", str(config), "--skill", str(final),
                "--images", str(task_copy / "images/09.png"), "--question", "Count red circles"])
    assert pred.stdout.strip() == "3"
    (task_copy / "initial_skill/SKILL.md").write_text("Changed initialization")
    changed = run(cmd, expected=1)
    assert "different data, code, models, or settings" in changed.stderr


def test_qwen_disabled_thinking_payload(monkeypatch, tmp_path):
    from viewcraft import models
    monkeypatch.setattr(models, "limiter_for", lambda *args: SimpleNamespace(acquire=lambda: None))
    cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    cfg["models"]["executor"].update(provider="ali", reasoning="disabled",
                                     max_tokens_field="max_completion_tokens")
    client = models.ModelClient("executor", cfg, tmp_path / "cache")
    payload = client.build_payload([{"role": "user", "content": "test"}], max_tokens=1024)
    assert payload["enable_thinking"] is False
    assert payload["max_completion_tokens"] == 1024
    assert "max_tokens" not in payload
