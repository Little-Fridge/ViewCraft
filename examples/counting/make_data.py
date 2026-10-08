"""Regenerate the bundled synthetic counting images and JSONL; no downloads."""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


def main():
    (ROOT / "images").mkdir(exist_ok=True)
    rows = []
    for i in range(12):
        rng = random.Random(7300 + i)
        red = 3 + i % 6
        image = Image.new("RGB", (960, 640), "white")
        draw = ImageDraw.Draw(image)
        locations = [(90 + x * 150, 95 + y * 145) for y in range(4) for x in range(6)]
        rng.shuffle(locations)
        for j, (x, y) in enumerate(locations[:red + 5]):
            x, y = x + rng.randint(-15, 15), y + rng.randint(-15, 15)
            color = (210, 35, 45) if j < red else (35, 80, 210)
            draw.ellipse((x - 22, y - 22, x + 22, y + 22), fill=color)
        filename = f"images/{i:02d}.png"
        image.save(ROOT / filename)
        rows.append({"id": f"count-{i:02d}", "question": "How many red circles are in the image? Reply with the integer only.",
                     "images": [filename], "answer": str(red), "group_key": f"image-{i:02d}",
                     "split": "train" if i < 6 else "gate" if i < 9 else "test"})
    (ROOT / "data.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"Wrote {len(rows)} samples in {ROOT}")


if __name__ == "__main__":
    main()
