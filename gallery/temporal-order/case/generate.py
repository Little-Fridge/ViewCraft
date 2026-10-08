"""Generate deterministic illustrated input frames, using only Pillow."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
W, H = 1200, 900
TIMES = (0, 4, 8, 12, 16)
INK = "#243a46"
SKIN = "#e9ad87"


def font(size, bold=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def scene(index):
    im = Image.new("RGB", (W, H), "#eff3f0")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 125), fill=INK)
    d.text((44, 26), "PARCEL STUDIO", fill="#ffffff", font=font(32, True))
    d.text((45, 72), "CAM 02   /   OVERHEAD VIEW", fill="#aec9cb", font=font(20))
    d.rounded_rectangle((926, 26, 1158, 97), radius=16, fill="#38515b")
    d.ellipse((948, 52, 963, 67), fill="#ed886d")
    d.text((981, 40), f"00:{TIMES[index]:02d}", fill="white", font=font(34, True))

    # Consistent workbench and background objects establish scene continuity.
    d.rounded_rectangle((35, 151, 1165, 820), radius=26, fill="#dce7e1")
    for y in range(180, 820, 48):
        d.line((40, y, 1158, y), fill="#d3dfd8", width=2)
    d.rounded_rectangle((61, 192, 206, 733), radius=16, fill="#f6f7ed")
    d.text((84, 215), "TOOLS", fill="#688077", font=font(20, True))
    # Scissors.
    d.ellipse((86, 279, 126, 329), outline="#4f7e82", width=9)
    d.ellipse((134, 279, 174, 329), outline="#4f7e82", width=9)
    d.line((110, 320, 158, 419), fill="#91a5a5", width=9)
    d.line((155, 320, 112, 419), fill="#91a5a5", width=9)
    d.ellipse((128, 348, 139, 359), fill=INK)
    # A pen.
    d.rounded_rectangle((123, 504, 143, 666), radius=5, fill="#385661")
    d.polygon([(123, 666), (143, 666), (133, 685)], fill="#ba9b72")
    d.line((129, 516, 129, 552), fill="#91bab9", width=3)

    # Box shadow and lower body; the box top occupies the same place in all frames.
    d.rounded_rectangle((291, 325, 915, 729), radius=14, fill="#b9c8be")
    d.rectangle((313, 343, 881, 701), fill="#a97647", outline="#805630", width=4)
    d.rectangle((321, 630, 873, 691), fill="#b98552")
    d.text((529, 651), "PARCEL 07", fill="#704c2e", font=font(22, True))

    def hand(x, y, upward=True):
        # Sleeves and rounded hands make the active object contact visible.
        if upward:
            d.rounded_rectangle((x - 35, y + 34, x + 35, 819), radius=12, fill="#426d82")
            d.rectangle((x - 37, y + 28, x + 37, y + 49), fill="#d1e1e3")
        else:
            d.rounded_rectangle((x - 35, 155, x + 35, y - 28), radius=12, fill="#426d82")
            d.rectangle((x - 37, y - 49, x + 37, y - 28), fill="#d1e1e3")
        d.rounded_rectangle((x - 34, y - 43, x + 34, y + 39), radius=24, fill=SKIN, outline="#bb805f", width=2)
        for delta in (-16, -2, 12):
            d.line((x + delta, y - 28, x + delta, y - 7), fill="#ce936e", width=2)

    def book(x, y):
        d.rounded_rectangle((x, y, x + 151, y + 180), radius=9, fill="#205b87", outline="#163f5f", width=4)
        d.rectangle((x + 12, y + 8, x + 146, y + 171), fill="#398dc1")
        d.rectangle((x + 17, y + 159, x + 144, y + 172), fill="#f2ead7")
        d.text((x + 34, y + 39), "FIELD", fill="white", font=font(23, True))
        d.text((x + 24, y + 73), "NOTES", fill="white", font=font(23, True))

    def ball(x, y):
        d.ellipse((x - 65, y - 65, x + 65, y + 65), fill="#ee983a", outline="#ae632b", width=4)
        d.arc((x - 35, y - 65, x + 35, y + 65), 90, 270, fill="#b46729", width=4)
        d.arc((x - 65, y - 28, x + 65, y + 32), 5, 177, fill="#b46729", width=4)
        d.ellipse((x - 32, y - 39, x - 12, y - 23), fill="#ffd18a")

    if index < 2:
        # Open flaps extend from all four sides, exposing the contents.
        d.polygon([(313, 343), (881, 343), (826, 261), (368, 261)], fill="#d5ac76", outline="#805630", width=4)
        d.polygon([(313, 343), (313, 626), (249, 657), (249, 387)], fill="#c99960", outline="#805630", width=4)
        d.polygon([(881, 343), (881, 626), (946, 657), (946, 387)], fill="#c99960", outline="#805630", width=4)
        d.rectangle((328, 359, 866, 621), fill="#765333")
        d.rounded_rectangle((343, 371, 851, 609), radius=15, fill="#e5cba0")
        book(403, 401)
        if index == 0:
            ball(1031, 566)
            hand(516, 540)
        else:
            ball(726, 484)
            hand(738, 530)
        d.polygon([(313, 626), (881, 626), (829, 735), (365, 735)], fill="#dab480", outline="#805630", width=4)
    else:
        # Two meeting flaps distinguish closure, then sealing, then labeling.
        d.rectangle((313, 343, 881, 626), fill="#d6ac77", outline="#805630", width=4)
        d.line((597, 344, 597, 625), fill="#795331", width=5)
        d.line((330, 358, 581, 358), fill="#e7c99f", width=4)
        d.line((611, 358, 864, 358), fill="#e7c99f", width=4)
        if index == 2:
            hand(485, 524)
            hand(709, 422, upward=False)
        if index >= 3:
            d.rectangle((563, 344, 631, 627), fill="#ead288")
            d.line((596, 345, 596, 627), fill="#c5a96b", width=2)
            d.rectangle((563, 628, 631, 687), fill="#c8aa60")
        if index == 3:
            hand(722, 564)
            # Tape dispenser nose contacts the new strip on the box seam.
            d.polygon([(587, 512), (654, 474), (728, 484), (752, 541), (657, 570), (587, 549)], fill="#cb614e", outline="#944031", width=4)
            d.ellipse((634, 477, 712, 549), fill="#ebd189", outline="#b69649", width=4)
            d.ellipse((656, 496, 691, 529), fill="#f9efcf", outline="#b69649", width=3)
            d.rectangle((582, 513, 597, 549), fill="#91a5a5")
        if index == 4:
            d.rounded_rectangle((661, 414, 845, 553), radius=7, fill="#fffdf6", outline="#c0b59c", width=2)
            d.text((680, 429), "SHIP TO", fill=INK, font=font(16, True))
            for y, length in ((459, 131), (472, 99), (485, 116)):
                d.line((680, y, 680 + length, y), fill="#67757a", width=3)
            for n in range(27):
                x = 680 + n * 5
                d.rectangle((x, 505, x + (1 if n % 3 else 3), 533), fill=INK)
            hand(821, 546)

    # Neutral timestamp rail; no captions reveal event names or the answer.
    d.text((44, 851), "PACKING BENCH   /   SOURCE FRAME", fill="#637a72", font=font(19))
    for n, sec in enumerate(TIMES):
        x = 712 + n * 96
        d.rounded_rectangle((x, 842, x + 81, 885), radius=9, fill=INK if n == index else "#dce6df")
        d.text((x + 10, 852), f"{sec:02d}s", fill="white" if n == index else "#6b8077", font=font(19, True))
    return im


def generate():
    frames = [scene(i) for i in range(5)]
    for i, frame in enumerate(frames, 1):
        frame.save(ROOT / f"frame_{i:02d}.png")
    # A labeled input contact sheet is separate from runtime/replay artifacts.
    preview = Image.new("RGB", (1500, 884), "#eff3f0")
    d = ImageDraw.Draw(preview)
    d.text((26, 22), "INPUT PREVIEW / Five timestamped scene frames", fill=INK, font=font(27, True))
    for i, frame in enumerate(frames):
        x, y = 24 + (i % 3) * 496, 78 + (i // 3) * 395
        preview.paste(frame.resize((472, 354), Image.Resampling.LANCZOS), (x, y))
    d.text((1033, 541), "PARCEL", fill=INK, font=font(43, True))
    d.text((1033, 596), "STUDIO", fill=INK, font=font(43, True))
    d.text((1033, 662), "00:00 - 00:16", fill="#688077", font=font(25))
    d.text((1033, 704), "Input frames only", fill="#688077", font=font(20))
    preview.save(ROOT / "input_preview.png")
    print("Generated five 1200x900 frames and input_preview.png")


if __name__ == "__main__":
    generate()
