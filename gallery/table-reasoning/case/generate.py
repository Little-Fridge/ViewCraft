"""Render the deterministic table input using only local Pillow assets."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Each tuple is one displayed record; this is source data, not a model trace.
ROWS = (
    ("O-4101", "North", "Kits", "Shipped", 18, 12),
    ("O-4102", "South", "Kits", "Shipped", 42, 12),
    ("O-4103", "North", "Parts", "Shipped", 36, 4),
    ("O-4104", "North", "Kits", "Pending", 25, 12),
    ("O-4105", "North", "Kits", "Shipped", 27, 10),
    ("O-4106", "West", "Kits", "Shipped", 19, 15),
    ("O-4107", "North", "Labels", "Shipped", 80, 2),
    ("O-4108", "North", "Kits", "Shipped", 14, 15),
    ("O-4109", "North", "Kits", "Returned", 12, 12),
    ("O-4110", "East", "Parts", "Shipped", 44, 4),
    ("O-4111", "North", "Kits", "Shipped", 31, 10),
    ("O-4112", "North", "Kits", "Cancelled", 16, 12),
)

WIDTH, HEIGHT = 1600, 1260
EDGES = (64, 244, 434, 634, 884, 1054, 1284, 1536)
HEADER_Y, HEADER_H, ROW_H = 304, 64, 62
INK = "#183047"
MUTED = "#53677A"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def draw_input():
    image = Image.new("RGB", (WIDTH, HEIGHT), "#EFF3F7")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 190), fill="#132C43")
    draw.rectangle((64, 40, 70, 143), fill="#49C6B8")
    draw.text((94, 36), "FIELDWORK  /  OPERATIONS", font=font(21, True), fill="#9FE4DB")
    draw.text((91, 78), "Dispatch ledger", font=font(49, True), fill="white")
    draw.text((94, 145), "Daily order register  |  Fulfillment desk", font=font(23), fill="#D3DFEA")

    draw.rounded_rectangle((64, 215, 1536, 278), radius=12, fill="white")
    draw.text((86, 233), "12 orders", font=font(24, True), fill=INK)
    draw.line((260, 229, 260, 263), fill="#DCE4EB", width=2)
    draw.text((284, 233), "4 depots", font=font(24), fill=MUTED)
    draw.text((1514, 233), "Currency: USD  |  Quantity: units", font=font(24), fill=MUTED, anchor="ra")

    draw.rectangle((EDGES[0], HEADER_Y, EDGES[-1], HEADER_Y + HEADER_H), fill="#24465F")
    headers = ("Order", "Depot", "Category", "Status", "Units", "Unit price", "Line total")
    for col, label in enumerate(headers):
        numeric = col >= 4
        x = EDGES[col + 1] - 24 if numeric else EDGES[col] + 24
        draw.text((x, HEADER_Y + 20), label, font=font(25, True), fill="white", anchor="ra" if numeric else "la")

    status_colors = {
        "Shipped": ("#E5F2EF", "#246456"),
        "Pending": ("#FFF0D9", "#855C1B"),
        "Returned": ("#EDEAF8", "#67538D"),
        "Cancelled": ("#F7E8E8", "#924C4C"),
    }
    for index, row in enumerate(ROWS):
        top = HEADER_Y + HEADER_H + index * ROW_H
        draw.rectangle((EDGES[0], top, EDGES[-1], top + ROW_H), fill="white" if index % 2 == 0 else "#F5F8FA")
        order, depot, category, status, units, price = row
        values = (order, depot, category, status, str(units), f"${price:.2f}", f"${units * price:.2f}")
        for col, value in enumerate(values):
            if col == 3:
                bg, fg = status_colors[status]
                left = EDGES[col] + 20
                draw.rounded_rectangle((left, top + 11, left + 194, top + 51), radius=8, fill=bg)
                draw.text((left + 13, top + 17), value, font=font(25), fill=fg)
            else:
                numeric = col >= 4
                x = EDGES[col + 1] - 24 if numeric else EDGES[col] + 24
                draw.text((x, top + 17), value, font=font(27, col == 4), fill=INK if col < 5 else MUTED, anchor="ra" if numeric else "la")
        draw.line((EDGES[0], top + ROW_H, EDGES[-1], top + ROW_H), fill="#DCE4EB", width=1)

    bottom = HEADER_Y + HEADER_H + len(ROWS) * ROW_H
    for boundary in (EDGES[4], EDGES[5]):
        draw.line((boundary, HEADER_Y + HEADER_H, boundary, bottom), fill="#D5E0E8", width=2)
    draw.text((64, bottom + 28), "Each row is one order. Line total = Units x Unit price.", font=font(23), fill=MUTED)
    draw.text((64, bottom + 67), "All statuses shown  /  No subtotal rows", font=font(21), fill=MUTED)
    draw.text((1536, bottom + 67), "REGISTER  /  01", font=font(21, True), fill=MUTED, anchor="ra")
    return image


def main():
    destination = Path(__file__).resolve().parent / "input.png"
    image = draw_input()
    image.save(destination, format="PNG", optimize=False)
    print(f"Generated {destination.name}: {image.width} x {image.height}")


if __name__ == "__main__":
    main()
