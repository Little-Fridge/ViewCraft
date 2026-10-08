#!/usr/bin/env python3
"""Build a single-file, editorial skill gallery. No downloads, panels or video UI."""
from pathlib import Path
import base64
import html
import io
import json
import re
import sys

from PIL import Image, ImageOps
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from viewcraft.standalone import asset_url, replay_document

ORDER = ['detail-reading','form-reading','chart-reading','table-reasoning','object-counting',
         'image-comparison','temporal-order','motion-state','multi-page-qa','screen-grounding']


def inline_font(path):
    return 'data:font/ttf;base64,' + base64.b64encode(path.read_bytes()).decode()


def plate(directory, request, destination):
    """Presentation of original case inputs; never described as a model view."""
    images=[]
    for name in request['images']:
        with Image.open(directory/'case'/name) as source:
            images.append(source.convert('RGB').copy())
    if len(images)==1:
        canvas=ImageOps.contain(images[0],(1300,1300))
    else:
        cols=2 if len(images) in (2,4) else 3
        rows=(len(images)+cols-1)//cols
        width,height=650,520
        canvas=Image.new('RGB',(cols*width,rows*height),'#faf8f3')
        for i,image in enumerate(images):
            im=ImageOps.contain(image,(width-18,height-18))
            canvas.paste(im,((i%cols)*width+(width-im.width)//2,(i//cols)*height+(height-im.height)//2))
    canvas.save(destination,optimize=True)


def build(gallery):
    presentation=json.loads((gallery/'presentation.json').read_text())
    fontdir=gallery/'assets/fonts'
    regular=inline_font(fontdir/'ComicNeue-Regular.ttf');bold=inline_font(fontdir/'ComicNeue-Bold.ttf')
    license_text=(fontdir/'OFL.txt').read_text().replace('--','—')
    sections=[];navigation=[];catalog=[];readme_rows=[]
    for number,ident in enumerate(ORDER,1):
        directory=gallery/ident
        meta=yaml.safe_load((directory/'skill.yaml').read_text())
        request=json.loads((directory/'case/request.json').read_text())
        run=directory/'case/run';result=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else None
        p=presentation[ident]
        navigation.append(f'<a href="#case-{ident}"><span>{number:02d}</span>{html.escape(p["label"])}</a>')
        plate_path=directory/'case/gallery-preview.png'
        plate(directory,request,plate_path)
        image=asset_url(plate_path)
        if result:
            answer=str(result['answer'])
            model=result.get('model_reported') or result.get('model','')
            reply=f'<div class="answer"><span class="small-label">MODEL ANSWER</span><strong>{html.escape(answer)}</strong><small>{html.escape(model)} · recorded case</small></div>'
            page=replay_document(run,meta['title'],request['question'],result)
            (run/'replay.html').write_text(page);(directory/'case/replay.html').write_text(page)
        else:
            reply='<div class="answer"><span class="small-label">CASE READY</span><strong>Ready to try</strong></div>'
        sections.append(f'''<section class="skill" id="case-{ident}" style="--accent:{p['color']}">
<div class="section-heading"><span class="number">{number:02d}</span><div><span class="category">{html.escape(p['label'])}</span><h2>{html.escape(p['title'])}</h2></div></div>
<div class="skill-body"><div class="example"><figure><img loading="lazy" src="{image}" alt="{html.escape(meta['title'],quote=True)} example input"><figcaption>{len(request['images'])} original input{'s' if len(request['images'])!=1 else ''}</figcaption></figure>
<div class="question"><span class="small-label">THE QUESTION</span><p>{html.escape(request['question'])}</p></div>{reply}</div>
<div class="recipe"><p class="recipe-title">A ViewCraft skill</p>
<div class="coordinate"><span class="symbol">φ</span><div><h3>What to see</h3><p>{html.escape(p['see'])}</p></div></div>
<div class="coordinate"><span class="symbol">ψ</span><div><h3>How to arrange it</h3><p>{html.escape(p['arrange'])}</p></div></div>
<div class="coordinate"><span class="symbol">θ</span><div><h3>How to read it</h3><p>{html.escape(p['read'])}</p></div></div>
<p class="skill-note">One reusable way to look.<br>One procedure that knows how to read it.</p>
</div></div></section>''')
        catalog.append({**meta,'presentation':p,'case':{'question':request['question'],'result':result}})
        readme_rows.append(f'| [{p["title"]}]({ident}/README.md) | {p["label"]} |')
        # Task README stays useful without presenting raw artifact download menus.
        (directory/'README.md').write_text(f'''# {p['title']}

{meta['description']}

## The ViewCraft skill

- **What to see (φ):** {p['see']}
- **How to arrange it (ψ):** {p['arrange']}
- **How to read it (θ):** {p['read']}

![Example input](case/gallery-preview.png)

**Question:** {request['question']}

**Recorded answer:** {str(result['answer']) if result else 'Not yet recorded'}

This is one illustrative case, not a benchmark accuracy claim. The preview shows
original inputs. Exact model attachments and responses remain in the case records.

## Use it

```bash
python scripts/use_skill.py --skill {ident} --record
```

For your own input add `--images my-image.png --question 'Your question'`.
For multi-image/video tasks provide ordered page images or extracted frames.
Use this directory as `initial_skill` to adapt it through `scripts/evolve.py`.
''')
    css=f'''@font-face{{font-family:GalleryHand;src:url("{regular}") format("truetype");font-weight:400;font-display:block}}@font-face{{font-family:GalleryHand;src:url("{bold}") format("truetype");font-weight:700;font-display:block}}
:root{{--ink:#252934;--muted:#71737c;--paper:#faf8f3}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;color:var(--ink);background:var(--paper);font:19px/1.55 GalleryHand,'Comic Sans MS',cursive}}main{{max-width:1170px;margin:auto;padding:36px 42px 50px}}a{{color:inherit;text-decoration:none}}.brand{{display:flex;align-items:center;gap:12px;font-size:23px;font-weight:700}}.brand svg{{width:34px;height:34px}}.brand span:last-child{{font-size:15px;font-weight:400;color:var(--muted);margin-left:auto}}header{{padding:44px 0 26px}}.eyebrow{{font-size:15px;letter-spacing:.15em;text-transform:uppercase;color:#6f7480}}h1{{font-size:clamp(48px,6vw,78px);letter-spacing:-.055em;line-height:1.02;margin:16px 0 22px;font-weight:700}}h1 em{{font-style:normal;position:relative;white-space:nowrap}}h1 em:after{{content:'';position:absolute;left:0;right:0;bottom:-6px;height:8px;border-top:3px solid #bd6d50;border-radius:50%;transform:rotate(-2deg)}}.intro{{max-width:650px;font-size:23px;color:#646977;margin:0 0 20px}}.legend{{display:flex;gap:25px;flex-wrap:wrap;font-size:17px;color:#646977;margin:26px 0 0}}.legend b{{color:var(--ink);font-size:23px;margin-right:5px}}.contents{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:11px 18px;padding:24px 0 30px;border-top:1px solid #dedbd3;border-bottom:1px solid #dedbd3;margin-top:18px}}.contents a{{font-size:17px;display:flex;gap:8px;align-items:baseline}}.contents span{{color:#a0a0a0;font-size:13px}}.contents a:hover{{color:#a15c45}}.skill{{padding:58px 0 65px;border-bottom:1px solid #dedbd3;scroll-margin-top:15px}}.section-heading{{display:flex;align-items:center;gap:19px;margin-bottom:28px}}.number{{width:58px;height:58px;display:grid;place-items:center;background:var(--accent);border-radius:43% 51% 42% 53%;font-size:29px;transform:rotate(-5deg)}}.category{{font-size:14px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase}}h2{{font-size:36px;letter-spacing:-.03em;line-height:1.15;margin:3px 0 0}}.skill-body{{display:grid;grid-template-columns:1.12fr 1fr;gap:38px;align-items:start}}figure{{margin:0;background:white;border:1px solid #e3e1dc;border-radius:12px;padding:14px 14px 8px}}figure img{{display:block;width:100%;height:330px;object-fit:contain}}figcaption{{text-align:right;color:#979ba2;font-size:12px;margin-top:4px}}.small-label{{display:block;font-size:12px;letter-spacing:.11em;color:#80828a;margin-bottom:5px}}.question{{padding:20px 4px 15px}}.question p{{font-size:19px;line-height:1.4;margin:0}}.answer{{background:var(--accent);padding:15px 18px;border-radius:9px;position:relative}}.answer strong{{font-size:25px;display:block;overflow-wrap:anywhere}}.answer small{{display:block;color:#68717d;font-size:12px;margin-top:4px}}.recipe{{padding:12px 2px}}.recipe-title{{font-size:14px;letter-spacing:.11em;text-transform:uppercase;color:#8c8c90;margin:0 0 25px}}.coordinate{{display:grid;grid-template-columns:32px 1fr;gap:13px;margin:0 0 22px}}.symbol{{font-size:25px;font-weight:700;color:#8d7061}}h3{{font-size:23px;margin:0 0 6px;line-height:1.1}}.coordinate p{{margin:0;color:#686d77;font-size:18px;line-height:1.5}}.skill-note{{font-size:18px;color:#90949b;border-top:1px dashed #d8d5cc;padding-top:17px;margin:28px 0 0 45px;font-style:italic}}.closing{{padding:52px 0 24px;display:flex;justify-content:space-between;gap:35px;align-items:center}}.closing h2{{font-size:32px}}.closing p{{color:var(--muted);max-width:550px}}.command{{font:16px/1.6 GalleryHand,cursive;border-left:3px solid #c68e77;padding:12px 20px;background:#f0ece3;overflow-wrap:anywhere}}footer{{font-size:13px;color:#98989e;padding-top:22px;border-top:1px solid #dedbd3}}
@media(max-width:800px){{main{{padding:22px}}.contents{{grid-template-columns:repeat(2,minmax(0,1fr))}}.skill-body{{grid-template-columns:1fr;gap:22px}}.skill{{padding:38px 0}}.recipe{{padding-top:6px}}figure img{{height:340px}}.skill-note{{display:none}}h2{{font-size:30px}}.closing{{display:block}}}}
@media(prefers-reduced-motion:reduce){{html{{scroll-behavior:auto}}}}
'''
    body='''<div class="brand"><svg viewBox="0 0 36 36" fill="none" aria-hidden="true"><path d="M3 18S9 7 18 7s15 11 15 11-6 11-15 11S3 18 3 18Z" stroke="currentColor" stroke-width="2"/><circle cx="18" cy="18" r="5" stroke="currentColor" stroke-width="2"/><path d="m27 2 1 4m5 3-4 1" stroke="#b96d51" stroke-width="2" stroke-linecap="round"/></svg><span>ViewCraft</span><span>the skill gallery</span></div>
<header><div class="eyebrow">10 multimodal skills</div><h1>A better way<br>to <em>look &amp; read.</em></h1><p class="intro">Give your agent a way to see the evidence,<br>and a skill that knows what to do with it.</p><div class="legend"><span><b>φ</b> what to see</span><span><b>ψ</b> how to arrange it</span><span><b>θ</b> how to read it</span></div></header>'''
    body+='<nav class="contents" aria-label="Ten skills">'+''.join(navigation)+'</nav>'+''.join(sections)
    body+='''<section class="closing"><div><h2>Take a skill. Make it yours.</h2><p>Choose a task, bring your own images, and run it with ViewCraft. You can also use any gallery skill as the starting point for evolution.</p></div><div class="command">python scripts/use_skill.py<br>--skill chart-reading</div></section><footer>ViewCraft · Recorded examples use GPT-6 Astra. Each example completed in one request; these cases illustrate usage, not general accuracy. This page works offline. Typeface: Comic Neue.</footer>'''
    page=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ViewCraft — Skill Gallery</title><style>{css}</style><!-- Embedded font license: {license_text} --></head><body><main>{body}</main></body></html>'
    (gallery/'index.html').write_text(page)
    (gallery/'catalog.json').write_text(json.dumps(catalog,indent=2,ensure_ascii=False))
    (gallery/'README.md').write_text('''# ViewCraft Skill Gallery

[Open the gallery](index.html) — one self-contained page with ten tasks, case
images, recorded answers and a concise explanation of each ViewCraft skill.
The page embeds its images and font. No videos, pop-ups or download menus.

| Skill | Task |
|---|---|
'''+ '\n'.join(readme_rows)+'''

Run a skill from the repository root after configuring a model:

```bash
python scripts/use_skill.py --skill chart-reading --record
python scripts/use_skill.py --skill form-reading --images my-form.png --question 'What is the purchase order number?' --record
```

The code directories retain SKILL.md, VIEW.yaml, LAYOUT.yaml, runtime metadata,
case inputs and actual model records. Use a gallery directory as `initial_skill`
with `scripts/evolve.py` to adapt it. Rebuild this page using
`python scripts/build_gallery.py`. Presentation text summarizes the saved skill;
it does not claim that every optional action occurred in the recorded case.

Font: Comic Neue, distributed with its SIL Open Font License in assets/fonts/.
''')
    return len(catalog)


if __name__=='__main__':
    print(f'Built {build(ROOT/"gallery")} skill presentations')
