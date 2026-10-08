"""Generate a deterministic pair of desk illustrations; no network required."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SIZE = (1400, 1050)
MUG_CENTERS = ((229, 267), (1150, 808))
# These bounds include the object's cast shadow and anti-aliased silhouette.
CHANGE_BOXES = ((160, 198, 340, 363), (1081, 739, 1261, 904))


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def rounded(d, box, fill, radius=15, outline=None, width=1):
    d.rounded_rectangle(box, radius, fill=fill, outline=outline, width=width)


def base_scene():
    im = Image.new("RGB", SIZE, "#f2eee6")
    d = ImageDraw.Draw(im)
    d.text((52, 28), "STUDIO  /  WORKSPACE", font=font(24), fill="#334440")
    d.text((1040, 31), "OVERHEAD VIEW", font=font(18), fill="#67736b")
    rounded(d, (39, 100, 1364, 1018), "#c5b49d", 34)
    rounded(d, (29, 88, 1354, 1005), "#dfc9a8", 34, "#c6ac85", 2)
    for y in range(122, 995, 39):
        d.line((48, y, 1337, y), fill="#d7bf9b", width=1)
        d.line((70, y + 4, 1310, y + 4), fill="#e4cfaf", width=1)
    # Fixed cork coasters; the empty source remains a distinct desk landmark.
    for x, y in MUG_CENTERS:
        rounded(d, (x - 79, y - 75, x + 81, y + 77), "#b99161", 27)
        rounded(d, (x - 73, y - 69, x + 75, y + 71), "#c5a16e", 23)
        for dx, dy in ((-55, -42), (-29, 53), (46, -48), (58, 30), (-57, 18)):
            d.ellipse((x + dx, y + dy, x + dx + 3, y + dy + 3), fill="#ad865a")
    # Desk mat, display and stand.
    rounded(d, (385, 147, 1003, 803), "#b59b7a", 30)
    rounded(d, (375, 137, 993, 792), "#50645d", 30)
    rounded(d, (628, 445, 739, 565), "#7c8987", 15)
    rounded(d, (558, 543, 815, 575), "#bac4c0", 14)
    rounded(d, (436, 184, 947, 491), "#314440", 20)
    rounded(d, (424, 172, 935, 480), "#263534", 20)
    rounded(d, (445, 193, 914, 438), "#e7eddd", 8)
    d.rectangle((445, 193, 914, 233), fill="#cad6c4")
    d.text((463, 202), "PROJECT NOTES", font=font(15), fill="#48604d")
    rounded(d, (469, 252, 613, 411), "#bacbb2", 8)
    for y, width in ((265, 227), (287, 199), (309, 212), (349, 166), (371, 192)):
        rounded(d, (637, y, 637 + width, y + 8), "#a9b99f", 3)
    d.ellipse((674, 452, 684, 462), fill="#8fa998")
    # Keyboard with individually drawn keycaps and space bar.
    rounded(d, (455, 612, 844, 763), "#344841", 14)
    rounded(d, (447, 604, 836, 754), "#c9d0c7", 14)
    for row in range(4):
        for col in range(12):
            x, y = 460 + col * 30, 617 + row * 25
            rounded(d, (x, y, x + 24, y + 19), "#f5f4ea", 4)
    rounded(d, (549, 718, 737, 741), "#f5f4ea", 4)
    for x in (460, 490, 760, 790):
        rounded(d, (x, 718, x + 24, 741), "#eceee2", 4)
    rounded(d, (876, 647, 947, 753), "#d8ded3", 34)
    d.line((911, 650, 911, 693), fill="#89978c", width=2)
    rounded(d, (906, 667, 916, 688), "#8c9b90", 4)
    # Small plant with a ceramic pot.
    d.ellipse((1127, 172, 1268, 305), fill="#b19877")
    d.ellipse((1115, 160, 1256, 293), fill="#eee8dc")
    d.ellipse((1128, 174, 1242, 278), fill="#6d5a42")
    for box, color in [((1145, 123, 1182, 221), "#416e4d"),
                       ((1190, 143, 1230, 234), "#5b8052"),
                       ((1095, 188, 1191, 225), "#6b8f5d"),
                       ((1185, 207, 1284, 245), "#426c4b"),
                       ((1140, 220, 1185, 298), "#709766"),
                       ((1181, 217, 1222, 283), "#5d8758")]:
        d.ellipse(box, fill=color)
    # Stationery tray and its unchanged contents.
    rounded(d, (1046, 367, 1299, 627), "#ad9170", 20)
    rounded(d, (1036, 357, 1289, 617), "#eee7d8", 20)
    rounded(d, (1051, 373, 1274, 601), "#d7cdbb", 12)
    d.line((1051, 507, 1274, 507), fill="#bbae96", width=7)
    for y, color in ((400, "#b98044"), (436, "#516d83"), (472, "#697453")):
        rounded(d, (1080, y, 1217, y + 11), color, 4)
        d.polygon(((1217, y), (1238, y + 6), (1217, y + 11)), fill="#e8bd83")
        d.polygon(((1231, y + 4), (1238, y + 6), (1231, y + 8)), fill="#39413b")
    rounded(d, (1076, 535, 1138, 578), "#db9f91", 6)
    rounded(d, (1094, 535, 1121, 578), "#f2e9dc", 2)
    d.ellipse((1182, 530, 1241, 589), fill="#b0915c")
    d.ellipse((1195, 543, 1228, 576), fill="#d7cdbb")
    # Closed notebook, pen and phone.
    rounded(d, (120, 473, 330, 750), "#b19877", 12)
    rounded(d, (109, 461, 319, 738), "#b46b53", 12)
    d.rectangle((122, 466, 130, 733), fill="#8d5343")
    d.rectangle((281, 462, 289, 737), fill="#734739")
    rounded(d, (151, 511, 258, 555), "#e8d4b9", 4)
    d.text((165, 522), "NOTES", font=font(20), fill="#765846")
    rounded(d, (337, 495, 351, 693), "#324b55", 5)
    d.polygon(((337, 693), (351, 693), (344, 716)), fill="#dbc3a0")
    d.rectangle((337, 511, 351, 520), fill="#c7c2a6")
    rounded(d, (142, 811, 256, 954), "#b29877", 17)
    rounded(d, (134, 803, 248, 946), "#293d40", 17)
    rounded(d, (143, 816, 239, 930), "#82969a", 10)
    d.text((157, 851), "09:41", font=font(23), fill="#eaf0e8")
    # A small paper card, kept away from both change regions.
    rounded(d, (505, 864, 864, 956), "#c3aa88", 8)
    rounded(d, (499, 858, 858, 950), "#f5efdf", 8)
    d.text((521, 876), "MAKE ROOM TO THINK", font=font(20), fill="#59665b")
    d.line((522, 916, 803, 916), fill="#c7c8b6", width=2)
    return im


def mug(im, center):
    """Draw the same mug and shadow at either of two integer offsets."""
    x, y = center
    layer = Image.new("RGBA", (180, 165))
    d = ImageDraw.Draw(layer)
    d.ellipse((21, 27, 146, 153), fill=(56, 49, 34, 48))
    d.ellipse((113, 60, 175, 118), fill=(56, 49, 34, 48))
    d.ellipse((102, 42, 163, 104), fill="#205d61")
    d.ellipse((116, 54, 150, 92), fill="#c5a16e")
    d.ellipse((7, 7, 132, 132), fill="#205d61")
    d.ellipse((10, 4, 131, 125), fill="#2e8588")
    d.ellipse((20, 14, 122, 116), fill="#a2d0c9")
    d.ellipse((28, 22, 114, 108), fill="#4c3529")
    d.arc((30, 24, 110, 104), 207, 286, fill="#a48361", width=3)
    d.arc((10, 4, 131, 125), 195, 263, fill="#69b4b0", width=4)
    im.paste(layer, (x - 69, y - 69), layer)


def main():
    base = base_scene()
    for index, center in enumerate(MUG_CENTERS, 1):
        scene = base.copy()
        mug(scene, center)
        scene.save(HERE / f"input_{index:02d}.png")
    print("Generated input_01.png and input_02.png (1400 x 1050).")


if __name__ == "__main__":
    main()
