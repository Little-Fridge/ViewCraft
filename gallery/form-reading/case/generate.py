"""Render the deterministic fictional invoice used by the form-reading case.

No network, API, or external assets are used. Run from any working directory.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1440, 1800
INK = "#172B3A"
MUTED = "#536777"
TEAL = "#136F73"
LINE = "#D6E1E5"


def font(size, bold=False):
    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def generate():
    image = Image.new("RGB", (WIDTH, HEIGHT), "#EDF2F4")
    draw = ImageDraw.Draw(image)
    draw.rectangle((55, 47, 1395, 1765), fill="#DFE7EB")
    draw.rectangle((40, 32, 1380, 1748), fill="white")
    draw.rectangle((40, 32, 1380, 47), fill=TEAL)

    def text(x, y, value, size=27, color=INK, bold=False):
        draw.text((x, y), value, font=font(size, bold), fill=color)

    def right(x, y, value, size=27, color=INK, bold=False):
        face = font(size, bold)
        draw.text((x - draw.textlength(value, font=face), y), value,
                  font=face, fill=color)

    def rule(y, left=100, end=1320):
        draw.line((left, y, end, y), fill=LINE, width=2)

    draw.rounded_rectangle((100, 103, 170, 173), radius=15, fill=TEAL)
    text(112, 111, "N", 47, "white", True)
    text(192, 101, "NORTHLINE", 42, bold=True)
    text(193, 155, "OFFICE SUPPLY CO.", 22, MUTED)
    right(1320, 101, "INVOICE", 48, bold=True)
    right(1320, 163, "Commercial account", 23, MUTED)
    text(100, 224, "218 Harbor Avenue  |  Portland, OR 97205", 24, MUTED)
    text(100, 260, "Business equipment and workplace essentials", 24, MUTED)
    rule(316)

    # All three reference fields share the same typography and prominence.
    # Their similar digits make the visible label association meaningful.
    x_edges = [100, 507, 914, 1320]
    draw.rounded_rectangle((100, 352, 1320, 611), radius=12,
                           fill="#F5F8F9", outline=LINE, width=2)
    for x in x_edges[1:-1]:
        draw.line((x, 352, x, 611), fill=LINE, width=2)
    rule(482)
    fields = [
        (0, 0, "Invoice number", "INV-48217"),
        (1, 0, "Purchase order number", "PO-48271"),
        (2, 0, "Customer number", "CUS-48721"),
        (0, 1, "Invoice date", "2026-09-18"),
        (1, 1, "Payment due", "2026-10-18"),
        (2, 1, "Sales order number", "SO-48712"),
    ]
    for col, row, label, value in fields:
        x, y = x_edges[col] + 24, 374 + row * 130
        text(x, y, label, 24, MUTED)
        text(x, y + 43, value, 34, bold=True)

    text(100, 659, "BILL TO", 21, TEAL, True)
    text(735, 659, "SHIP TO", 21, TEAL, True)
    for x, lines in [
        (100, ["Alder Studio LLC", "Accounts Payable", "640 Pine Street, Suite 200", "Portland, OR 97204"]),
        (735, ["Alder Studio LLC", "Receiving / East entrance", "640 Pine Street, Suite 200", "Portland, OR 97204"]),
    ]:
        for index, value in enumerate(lines):
            text(x, 702 + index * 39, value, 28 if index == 0 else 25,
                 INK if index == 0 else MUTED, index == 0)

    draw.rounded_rectangle((100, 902, 1320, 962), radius=8, fill=INK)
    text(122, 918, "DESCRIPTION", 21, "white", True)
    right(891, 918, "QTY", 21, "white", True)
    right(1091, 918, "UNIT PRICE", 21, "white", True)
    right(1296, 918, "AMOUNT", 21, "white", True)
    items = [
        ("Adjustable monitor arm", "6", "$85.00", "$510.00"),
        ("Desk cable organizer", "12", "$15.00", "$180.00"),
        ("Task light / graphite", "4", "$62.00", "$248.00"),
    ]
    for index, (label, qty, price, amount) in enumerate(items):
        y = 990 + index * 82
        text(122, y, label, 27)
        right(891, y, qty)
        right(1091, y, price)
        right(1296, y, amount)
        rule(y + 60)

    text(100, 1290, "PAYMENT TERMS", 21, TEAL, True)
    text(100, 1332, "Net 30 days", 27)
    text(100, 1377, "Please include the invoice number", 23, MUTED)
    text(100, 1411, "with your remittance.", 23, MUTED)
    for y, label, amount in [
        (1274, "Subtotal", "$938.00"),
        (1323, "Delivery", "$24.00"),
        (1372, "Tax", "$0.00"),
    ]:
        text(865, y, label, 26, MUTED)
        right(1296, y, amount, 27)
    draw.rounded_rectangle((838, 1434, 1320, 1520), radius=10, fill="#E5F2F0")
    text(865, 1460, "TOTAL DUE", 23, TEAL, True)
    right(1296, 1454, "$962.00", 36, TEAL, True)

    rule(1590)
    text(100, 1621, "Thank you for your business.", 26, bold=True)
    text(100, 1663, "Northline Office Supply Co.  /  Accounts department", 22, MUTED)
    right(1320, 1663, "Page 1 of 1", 22, MUTED)

    ROOT.mkdir(parents=True, exist_ok=True)
    output = ROOT / "input.png"
    image.save(output)
    print(f"Generated {output.name}: {WIDTH} x {HEIGHT}")


if __name__ == "__main__":
    generate()
