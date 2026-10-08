"""Capture actual model attachments and responses for a portable visual replay."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from pathlib import Path



class RecordingClient:
    """ModelClient wrapper; never records auth headers, endpoints, or hidden reasoning."""
    def __init__(self, client, directory: Path):
        self.client = client
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "views").mkdir(exist_ok=True)
        self.steps = []

    def __getattr__(self, name):
        return getattr(self.client, name)

    def call(self, messages, **kwargs):
        index = len(self.steps) + 1
        attachments = []
        # Runtime attaches local images as data URLs. Save exact attachment bytes.
        for message in messages:
            if not isinstance(message.get("content"), list):
                continue
            for part in message["content"]:
                if part.get("type") != "image_url":
                    continue
                url = part["image_url"]["url"]
                if not url.startswith("data:image/"):
                    continue
                header, data = url.split(",", 1)
                suffix = ".jpg" if "jpeg" in header else ".png"
                rel = f"views/step-{index:02d}-{len(attachments)+1:02d}{suffix}"
                (self.directory / rel).write_bytes(base64.b64decode(data))
                attachments.append(rel)
        row = {"step": index, "images": attachments, "status": "pending"}
        self.steps.append(row)
        try:
            # A recorded demonstration must actually issue the model call.
            kwargs["use_cache"] = False
            response = self.client.call(messages, **kwargs)
        except Exception as exc:
            row.update(status="error", error_type=type(exc).__name__)
            self.save()
            raise
        row.update(status="ok", response=response.content,
                   model=response.model_reported, request_id=response.request_id,
                   usage=response.usage.__dict__, seconds=response.wall_time_s,
                   from_cache=response.from_cache,
                   recorded_at=datetime.now(timezone.utc).isoformat())
        self.save()
        return response

    def save(self):
        (self.directory / "steps.json").write_text(json.dumps(self.steps, indent=2, ensure_ascii=False))


def render_replay(directory: Path, title: str, question: str, result: dict):
    """Export actual recorded images and returned text as a standalone page."""
    from .standalone import replay_document
    directory = Path(directory)
    (directory / "replay.html").write_text(replay_document(directory, title, question, result))
