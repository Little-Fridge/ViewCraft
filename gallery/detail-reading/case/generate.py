"""Draw deterministic fictional product artwork with Pillow; no external data."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 2800, 3600
BG = "#EDF1F2"
INK = "#16313D"
MUTED = "#526773"
TEAL = "#007E80"
LINE = "#D4DFE3"
WHITE = "#FFFFFF"


def font(size, bold=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def generate():
    im = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(im)

    def text(x, y, value, size=28, fill=INK, bold=False):
        d.text((x, y), value, font=font(size, bold), fill=fill)

    def rule(x0, y, x1, fill=LINE, width=2):
        d.line((x0, y, x1, y), fill=fill, width=width)

    def panel(x, y, number, title, rows, note):
        w, h = 1280, 610
        d.rounded_rectangle((x, y, x+w, y+h), radius=20, fill=WHITE)
        d.rounded_rectangle((x+34, y+34, x+110, y+100), radius=12, fill=TEAL)
        text(x+51, y+45, number, 33, WHITE, True)
        text(x+139, y+46, title, 32, INK, True)
        rule(x+36, y+131, x+w-36)
        for index, (label, value) in enumerate(rows):
            yy = y+164+index*65
            text(x+43, yy, label, 24, MUTED)
            text(x+642, yy, value, 24, INK)
            rule(x+43, yy+47, x+w-43, "#EAF0F2", 1)
        text(x+43, y+553, note, 22, MUTED)

    d.rectangle((0, 0, WIDTH, 24), fill=TEAL)
    text(100, 78, "NORTHLINE  /  INSTRUMENTS", 37, TEAL, True)
    text(100, 164, "AERIS M4", 104, INK, True)
    text(104, 301, "Environmental monitoring node", 43)
    text(105, 373, "PRODUCT SPECIFICATION     /     FIELD SERIES", 26, MUTED)
    d.rounded_rectangle((2180, 94, 2700, 179), radius=40, fill=INK)
    text(2227, 116, "TECHNICAL OVERVIEW", 26, WHITE, True)
    text(2220, 216, "Document  PS-M4-216", 26, MUTED)
    text(2220, 261, "Revision  04  /  EN", 26, MUTED)
    rule(100, 461, 2700)

    # Product silhouette: vector-like mechanical drawing with restrained callouts.
    d.rounded_rectangle((100, 510, 2700, 1390), radius=26, fill="#E0E9EC")
    text(155, 554, "DEVICE PROFILE", 28, TEAL, True)
    text(158, 609, "Compact enclosure. Continuous measurement.", 26, MUTED)
    d.ellipse((945, 1180, 1885, 1275), fill="#CCD9DE")
    d.rounded_rectangle((1025, 650, 1795, 1190), radius=74, fill="#BCCBD2", outline="#718A98", width=4)
    d.rounded_rectangle((995, 610, 1765, 1140), radius=72, fill="#FAFCFD", outline="#78929F", width=5)
    d.rounded_rectangle((1080, 694, 1680, 918), radius=24, fill=INK)
    text(1121, 729, "AERIS", 35, "#A8DAD5", True)
    text(1121, 790, "22.6", 75, WHITE)
    text(1350, 837, "deg C", 25, "#A8DAD5")
    d.ellipse((1590, 738, 1614, 762), fill="#62DCA3")
    for j in range(11):
        d.rounded_rectangle((1110+j*44, 976, 1125+j*44, 1050), radius=7, fill="#ACBDC5")
    for xx, yy in ((1035, 650), (1720, 650), (1035, 1093), (1720, 1093)):
        d.ellipse((xx-10, yy-10, xx+10, yy+10), fill="#9EB0B9")
        d.line((xx-5, yy, xx+5, yy), fill=MUTED, width=2)
    d.rounded_rectangle((1240, 1140, 1485, 1220), radius=15, fill="#718995")
    d.rectangle((1270, 1190, 1455, 1255), fill=INK)
    d.line((1660, 800, 1940, 735, 2450, 735), fill=TEAL, width=3)
    text(1960, 657, "STATUS DISPLAY", 28, TEAL, True)
    text(1960, 763, "Live temperature + node status", 23, MUTED)
    d.line((1130, 1020, 785, 1100, 305, 1100), fill=TEAL, width=3)
    text(305, 1025, "VENTED SENSOR BAY", 28, TEAL, True)
    text(305, 1133, "Replaceable protective membrane", 23, MUTED)
    text(1960, 1188, "Shown without mounting bracket", 22, MUTED)
    text(1960, 1230, "Illustration is not to scale", 22, MUTED)

    panel(100, 1450, "01", "MEASUREMENT", [
        ("Temperature range", "-20 to +70 deg C"),
        ("Temperature accuracy", "+/- 0.2 deg C"),
        ("Humidity range", "0 to 95 % RH"),
        ("Humidity accuracy", "+/- 2 % RH"),
        ("Sampling interval", "2 s / configurable"),
        ("Sensor cartridge", "SNS-M4-227")
    ], "Accuracy values apply under reference conditions.")
    panel(1420, 1450, "02", "POWER & INTERFACE", [
        ("Supply input", "12 to 24 V DC"),
        ("Typical consumption", "1.8 W"),
        ("Wired interface", "RS-485 / Modbus RTU"),
        ("Connector type", "M12 / 5-pin"),
        ("Cable assembly", "CBL-5P-204"),
        ("Configuration port", "USB-C / service only")
    ], "Use a regulated supply with a common ground.")
    panel(100, 2110, "03", "MECHANICAL", [
        ("Enclosure dimensions", "118 x 82 x 34 mm"),
        ("Net mass", "246 g"),
        ("Housing material", "PC / ABS blend"),
        ("Protection rating", "IP54 / installed"),
        ("Mounting pitch", "64 mm"),
        ("Bracket reference", "BRK-64-A2")
    ], "Allow 30 mm clearance around the sensor vents.")
    panel(1420, 2110, "04", "SERVICE & TRACEABILITY", [
        ("Product model", "AM4-RS-24"),
        ("Unit serial", "SN-6R8P-3527"),
        ("Calibration reference", "CAL-7Q4M-82K6"),
        ("Inspection lot", "LOT-2639-B"),
        ("Service document", "SRV-M4-118"),
        ("Firmware branch", "v2.6 / stable")
    ], "Quote the relevant identifier when requesting service.")
    panel(100, 2770, "05", "INSTALLATION", [
        ("Mounting orientation", "Vertical / vents downward"),
        ("Operating altitude", "Up to 2,000 m"),
        ("Storage temperature", "-30 to +80 deg C"),
        ("Warm-up time", "5 min"),
        ("Recommended fastener", "M4 x 12"),
        ("Installation guide", "INS-M4-063")
    ], "Keep the sensing area away from local heat sources.")
    panel(1420, 2770, "06", "PACKAGE CONTENTS", [
        ("Monitoring node", "1 unit"),
        ("Wall bracket", "1 piece"),
        ("Mounting screws", "2 pieces"),
        ("Quick reference card", "QRC-M4-092"),
        ("Protective vent cap", "CAP-V2-014"),
        ("Packing reference", "PKG-M4-24B")
    ], "Cable assemblies are supplied separately.")
    rule(100, 3434, 2700)
    text(100, 3471, "NORTHLINE INSTRUMENTS   /   ENGINEERED FOR FIELD OBSERVATION", 23, TEAL, True)
    text(100, 3520, "Fictional product artwork for a visual reading example.", 22, MUTED)
    text(2410, 3495, "SHEET  01 / 01", 25, MUTED)
    im.save(HERE / "input.png", optimize=True)
    im.thumbnail((700, 900), Image.Resampling.LANCZOS)
    im.save(HERE / "preview.png", optimize=True)
    print(f"Generated {HERE / 'input.png'} ({WIDTH} x {HEIGHT}) and preview.png")


if __name__ == "__main__":
    generate()
