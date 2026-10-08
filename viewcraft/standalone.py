"""Small self-contained HTML views of genuine model recordings."""
from __future__ import annotations

import base64
import html
import json
import mimetypes
from pathlib import Path


def asset_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def replay_document(directory: Path, title: str, question: str, result: dict) -> str:
    """Show the actual request images and answers without panels or downloads."""
    steps = json.loads((directory / "steps.json").read_text())
    fontdir = Path(__file__).resolve().parents[1] / "gallery/assets/fonts"
    fonts = ""
    for weight, name in ((400, "ComicNeue-Regular.ttf"), (700, "ComicNeue-Bold.ttf")):
        path = fontdir / name
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            fonts += f'@font-face{{font-family:GalleryHand;src:url(data:font/ttf;base64,{encoded}) format("truetype");font-weight:{weight}}}'
    sections = []
    for row in steps:
        images = ''.join(f'<img src="{asset_url(directory / p)}" alt="Actual request {row["step"]}, image {i+1}">'
                         for i, p in enumerate(row["images"]))
        sections.append(f'<section><h2>Request {row["step"]}</h2><div class="images">{images}</div>'
                        f'<h3>Model response</h3><pre>{html.escape(row.get("response", row.get("error_type", "")))}</pre></section>')
    model = result.get("model_reported") or result.get("model", "")
    license_path = fontdir / "OFL.txt"
    license_text = license_path.read_text().replace('--', '—') if license_path.exists() else ''
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} — ViewCraft</title><style>{fonts}
*{{box-sizing:border-box}}body{{margin:0;background:#faf8f3;color:#282d35;font:19px/1.6 GalleryHand,'Comic Sans MS',cursive}}main{{max-width:1080px;margin:auto;padding:38px 25px}}h1{{font-size:42px;line-height:1.12}}h2{{font-size:28px}}.label{{font-size:14px;color:#898c93;letter-spacing:.1em}}.answer{{padding:18px 24px;background:#e4ede7;border-radius:10px;font-size:26px}}.images{{display:flex;flex-wrap:wrap;gap:16px}}.images img{{object-fit:contain;max-width:100%;max-height:780px;flex:1 1 300px;min-width:0;width:300px}}section{{padding:24px 0;border-bottom:1px solid #dedbd3}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:17px/1.6 GalleryHand,cursive;color:#68717a}}footer{{font-size:14px;color:#8a8f96;margin-top:26px}}
</style><!-- Embedded font license: {license_text} --></head><body><main><p class="label">VIEWCRAFT · RECORDED CASE</p>
<h1>{html.escape(title)}</h1><p>{html.escape(question)}</p><p class="answer">{html.escape(str(result.get("answer", "")))}</p>
{''.join(sections)}<footer>{html.escape(str(model))} · Exact model attachments and returned response. One example, not a general accuracy claim. This page works offline. Typeface: Comic Neue.</footer></main></body></html>'''
