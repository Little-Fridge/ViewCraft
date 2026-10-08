"""Generate a deterministic five-frame tabletop observation, without network access."""
from pathlib import Path
import json
import math

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SIZE = (1200, 860)
TIMES = [0.0, 1.0, 2.0, 3.0, 4.0]
COLORS = {"A": "#D65450", "B": "#397CD1", "C": "#E5A735", "D": "#339B77"}
POSITIONS = {
    "A": [(260, 335), (380, 335), (500, 335), (620, 335), (740, 335)],
    "B": [(920, 335)] * 5,
    "C": [(300, 595), (420, 595), (540, 595), (420, 595), (300, 595)],
    "D": [(810, 595)] * 5,
}


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def centered(draw, xy, text, size, fill):
    f = font(size)
    box = draw.textbbox((0, 0), text, font=f)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2 - box[1]), text, font=f, fill=fill)


def object_body(draw, label, xy):
    x, y = xy
    c = COLORS[label]
    draw.ellipse((x - 64, y + 36, x + 68, y + 62), fill="#DDD8CE")
    if label == "A":
        draw.rounded_rectangle((x - 60, y - 51, x + 60, y + 51), radius=14, fill=c, outline="#A63836", width=3)
        draw.line((x - 46, y - 32, x + 46, y - 32), fill="#F58C83", width=5)
        for dx in (-43, 43):
            draw.line((x + dx, y - 22, x + dx, y + 36), fill="#A63836", width=3)
    elif label == "B":
        draw.rounded_rectangle((x - 57, y - 42, x + 57, y + 43), radius=12, fill=c, outline="#285690", width=3)
        draw.ellipse((x - 57, y - 61, x + 57, y - 23), fill="#88B9F0", outline="#285690", width=3)
        draw.arc((x - 57, y + 23, x + 57, y + 59), 0, 180, fill="#285690", width=3)
    elif label == "C":
        points = [(x + 66 * math.cos(i * math.pi / 3), y + 66 * math.sin(i * math.pi / 3)) for i in range(6)]
        draw.polygon(points, fill=c, outline="#996718", width=3)
        draw.line((x - 24, y - 42, x + 24, y - 42), fill="#FFE1A1", width=5)
    else:
        draw.rounded_rectangle((x - 74, y - 48, x + 74, y + 48), radius=46, fill=c, outline="#216E53", width=3)
        draw.arc((x - 62, y - 37, x + 62, y + 36), 195, 310, fill="#89D2B8", width=4)
    draw.rounded_rectangle((x - 23, y - 22, x + 23, y + 25), radius=8, fill="#FFFFFF")
    centered(draw, (x, y + 1), label, 34, "#192C42")


def render(index):
    im = Image.new("RGB", SIZE, "#EDF1F4")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1200, 144), fill="#172B42")
    d.text((48, 22), "MOTION LAB  /  TABLETOP OBSERVATION", font=font(18), fill="#9CB8CF")
    d.text((46, 56), "Four objects. Five moments.", font=font(36), fill="white")
    d.rounded_rectangle((935, 30, 1152, 114), radius=14, fill="#2A4058")
    centered(d, (1044, 56), f"FRAME {index + 1:02d} / 05", 20, "#DDE9F2")
    centered(d, (1044, 89), f"t = {TIMES[index]:.1f} s", 24, "white")
    d.rounded_rectangle((42, 170, 1158, 756), radius=22, fill="#FAF8F2", outline="#CBD3D9", width=2)
    for x in range(100, 1140, 40):
        d.line((x, 222, x, 716), fill="#E6E3DA", width=1)
    for y in range(235, 720, 40):
        d.line((84, y, 1116, y), fill="#E6E3DA", width=1)
    d.text((76, 187), "FIXED OVERHEAD CAMERA", font=font(16), fill="#617280")
    for x, y in ((84, 222), (1116, 222), (84, 716), (1116, 716)):
        d.line((x - 11, y, x + 11, y), fill="#647A88", width=3)
        d.line((x, y - 11, x, y + 11), fill="#647A88", width=3)
    for label in "ABCD":
        object_body(d, label, POSITIONS[label][index])
    d.text((48, 785), "OBJECT KEY", font=font(16), fill="#677D8F")
    for j, (label, name) in enumerate(zip("ABCD", ("coral crate", "blue drum", "amber hex", "green capsule"))):
        x = 216 + j * 239
        d.rounded_rectangle((x, 781, x + 30, 811), radius=6, fill=COLORS[label])
        d.text((x + 40, 784), f"{label}  {name}", font=font(17), fill="#243B50")
    return im


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(5):
        im = render(i)
        im.save(ROOT / f"frame_{i + 1:02d}.png")
        frames.append(im)
    request = {
        "question": "Which labeled objects remained at the same position in every one of the five frames? An object that moves and returns to its starting position does not count. Return only the labels in alphabetical order, separated by a comma and a space.",
        "images": [f"frame_{i + 1:02d}.png" for i in range(5)],
        "options": [],
        "frame_times_s": TIMES,
    }
    (ROOT / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    (ROOT / "expected.json").write_text(json.dumps({"answer": "B, D", "match": "exact"}, indent=2) + "\n")
    preview = Image.new("RGB", (1500, 808), "#EDF1F4")
    d = ImageDraw.Draw(preview)
    d.text((24, 15), "INPUT PREVIEW  /  FIVE SOURCE FRAMES", font=font(22), fill="#243B50")
    for i, im in enumerate(frames):
        preview.paste(im.resize((480, 344), Image.Resampling.LANCZOS), (15 + (i % 3) * 500, 64 + (i // 3) * 364))
    preview.save(ROOT / "input_preview.png")
    print(f"Generated {len(frames)} frames at {SIZE[0]}x{SIZE[1]} and an input preview.")


if __name__ == "__main__":
    main()
