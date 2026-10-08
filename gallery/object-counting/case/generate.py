"""Generate one deterministic, locally rendered counting case. No API or network."""
from collections import Counter
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SIZE = (1440, 1080)
SCALE = 2
COLORS = {"orange": "#EF882B", "green": "#269D79", "blue": "#367AC5"}
# Each tuple is a physical piece, in row-major order within a compartment.
# Deliberately include same-color and same-shape distractors.
PANELS = (
    (("orange", "triangle"), ("green", "circle"), ("blue", "square"),
     ("orange", "circle"), ("blue", "triangle"), ("orange", "triangle"),
     ("green", "square"), ("blue", "circle"), ("green", "triangle")),
    (("blue", "circle"), ("orange", "square"), ("green", "triangle"),
     ("green", "circle"), ("orange", "triangle"), ("blue", "square"),
     ("blue", "triangle"), ("orange", "circle"), ("green", "square")),
    (("green", "square"), ("orange", "triangle"), ("blue", "circle"),
     ("orange", "triangle"), ("green", "circle"), ("blue", "triangle"),
     ("blue", "square"), ("green", "triangle"), ("orange", "triangle")),
    (("orange", "circle"), ("blue", "square"), ("green", "triangle"),
     ("blue", "triangle"), ("green", "square"), ("orange", "square"),
     ("green", "circle"), ("orange", "triangle"), ("blue", "circle")),
)


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size * SCALE)
    except OSError:
        return ImageFont.load_default()


def box(coords):
    return tuple(round(v * SCALE) for v in coords)


def polygon(cx, cy, radius, sides, angle):
    return [(round((cx + radius * math.cos(angle + i * math.tau / sides)) * SCALE),
             round((cy + radius * math.sin(angle + i * math.tau / sides)) * SCALE))
            for i in range(sides)]


def draw_piece(draw, color, shape, cx, cy, rotation, radius):
    fill = COLORS[color]
    for offset, paint in ((5, "#DDDCD3"), (0, fill)):
        if shape == "circle":
            draw.ellipse(box((cx-radius, cy-radius+offset, cx+radius, cy+radius+offset)), fill=paint)
        elif shape == "square":
            half = radius * 0.85
            draw.rectangle(box((cx-half, cy-half+offset, cx+half, cy+half+offset)), fill=paint)
        elif shape == "triangle":
            draw.polygon(polygon(cx, cy+offset, radius*1.17, 3, -math.pi/2+rotation), fill=paint)
        else:
            raise ValueError(shape)


def generate():
    image = Image.new("RGB", (SIZE[0]*SCALE, SIZE[1]*SCALE), "#F3F1E9")
    d = ImageDraw.Draw(image)
    d.rounded_rectangle(box((72, 52, 80, 172)), radius=4*SCALE, fill="#213D42")
    d.text(box((104, 48)), "MATERIAL SORT", font=font(18), fill="#647878")
    d.text(box((100, 79)), "The geometry tray", font=font(48), fill="#213D42")
    d.text(box((103, 145)), "Colored pieces arranged in four compartments", font=font(22), fill="#667776")
    manifest = []
    for panel, pieces in enumerate(PANELS):
        left = 72 + (panel % 2)*664
        top = 220 + (panel // 2)*416
        d.rounded_rectangle(box((left, top+5, left+632, top+391)), radius=24*SCALE, fill="#E0E2D9")
        d.rounded_rectangle(box((left, top, left+632, top+386)), radius=24*SCALE, fill="#FFFEF9", outline="#D6DDD6", width=2*SCALE)
        d.text(box((left+24, top+17)), chr(65+panel), font=font(18), fill="#728580")
        for index, (color, shape) in enumerate(pieces):
            cx = left + 112 + (index % 3)*204
            cy = top + 98 + (index // 3)*110
            radius = (37, 40, 42)[(panel+index) % 3]
            rotation = ((panel*3+index) % 4)*math.pi/2
            draw_piece(d, color, shape, cx, cy, rotation, radius)
            manifest.append({"panel": chr(65+panel), "color": color, "shape": shape,
                             "center_px": [cx, cy], "radius": radius})
    image = image.resize(SIZE, Image.Resampling.LANCZOS)
    image.save(HERE / "input.png", optimize=False)
    target = ("orange", "triangle")
    counts = Counter((obj["color"], obj["shape"]) for obj in manifest)
    assert len(manifest) == 36 and counts[target] == 7
    request = {"question": "How many orange triangles are there across all four compartments? Count only pieces that are both orange and triangular. Answer with only the integer.",
               "images": ["input.png"], "options": []}
    expected = {"answer": str(counts[target]), "match": "numeric", "tolerance": 0}
    for name, value in (("request.json", request), ("expected.json", expected)):
        (HERE / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {SIZE[0]}x{SIZE[1]} input.png; {len(manifest)} separated pieces.")


if __name__ == "__main__":
    generate()
