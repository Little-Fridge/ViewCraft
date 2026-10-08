"""Render the deterministic gallery input with Pillow; no network or API calls."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1600, 1100
SCALE = 2
MONTHS = ("January", "February", "March", "April")
SERIES = (
    ("Physical books", "#4263D5", (120, 150, 180, 160)),
    ("Digital books", "#148779", (90, 110, 125, 140)),
)


def font(size, bold=False):
    try:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(name, size * SCALE)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size * SCALE)
        except OSError:
            return ImageFont.load_default()


def generate():
    image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), "#EEF2F7")
    draw = ImageDraw.Draw(image)

    def box(coords, fill, radius=0, outline=None):
        rect = tuple(round(v * SCALE) for v in coords)
        if radius:
            draw.rounded_rectangle(rect, radius=radius * SCALE, fill=fill, outline=outline)
        else:
            draw.rectangle(rect, fill=fill, outline=outline)

    def line(coords, fill, width=1):
        draw.line(tuple(round(v * SCALE) for v in coords), fill=fill, width=width * SCALE)

    def text(x, y, value, size=24, fill="#243449", bold=False, anchor="la"):
        draw.text((x * SCALE, y * SCALE), value, font=font(size, bold), fill=fill, anchor=anchor)

    box((36, 36, 1564, 1064), "#FFFFFF", radius=26)
    box((36, 36, 1564, 211), "#14263D", radius=26)
    box((36, 175, 1564, 211), "#14263D")
    box((76, 77, 83, 162), "#58D5BE", radius=3)
    text(106, 68, "COMMUNITY LIBRARY  /  CIRCULATION", 19, "#A9BED5", True)
    text(104, 103, "Library loans by month", 46, "#FFFFFF", True)
    text(106, 163, "January-April  |  Monthly borrowing activity", 21, "#D6E1EF")

    text(100, 246, "BORROWING FORMAT", 17, "#65758A", True)
    for index, (name, color, _) in enumerate(SERIES):
        x = 880 + index * 300
        box((x, 245, x + 26, 271), color, radius=4)
        text(x + 39, 241, name, 24, bold=True)

    left, right, top, bottom = 202, 1474, 355, 899
    plot_height = bottom - top
    y_max = 200

    def y_position(value):
        return bottom - value / y_max * plot_height

    for tick in range(0, y_max + 1, 25):
        y = y_position(tick)
        line((left, y, right, y), "#DAE2EB", width=1)
        text(left - 22, y, str(tick), 22, "#5A697D", anchor="rm")
    line((left, top, left, bottom), "#8292A7", width=2)
    line((left, bottom, right, bottom), "#8292A7", width=2)

    # Rotate only the axis title; all numerical and category labels stay horizontal.
    axis_font = font(25, True)
    axis_title = "Number of loans"
    bounds = axis_font.getbbox(axis_title)
    label = Image.new("RGBA", (bounds[2] - bounds[0] + 20, bounds[3] - bounds[1] + 20))
    ImageDraw.Draw(label).text((10 - bounds[0], 10 - bounds[1]), axis_title, font=axis_font, fill="#243449")
    label = label.rotate(90, expand=True)
    image.paste(label, (85 * SCALE - label.width // 2, int((top + bottom) / 2 * SCALE) - label.height // 2), label)

    group_width = (right - left) / len(MONTHS)
    bar_width, half_gap = 79, 10
    for index, month in enumerate(MONTHS):
        center = left + group_width * (index + 0.5)
        for series_index, (_, color, values) in enumerate(SERIES):
            x0 = center - half_gap - bar_width if series_index == 0 else center + half_gap
            value = values[index]
            y = y_position(value)
            box((x0, y, x0 + bar_width, bottom - 1), color)
            text(x0 + bar_width / 2, y - 12, str(value), 26, fill=color, bold=True, anchor="mb")
        text(center, bottom + 23, month, 25, bold=True, anchor="ma")

    text((left + right) / 2, 964, "Month", 24, "#526278", True, anchor="ma")
    line((82, 1015, 1518, 1015), "#E1E7EF")
    text(84, 1030, "ILLUSTRATIVE DATA", 15, "#65758A", True)
    text(1516, 1030, "Each bar represents one borrowing format in one month.", 16, "#65758A", anchor="ra")

    output = Path(__file__).resolve().parent / "input.png"
    image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).save(output, optimize=True)
    print(f"Generated {output} ({WIDTH} x {HEIGHT})")


if __name__ == "__main__":
    generate()
