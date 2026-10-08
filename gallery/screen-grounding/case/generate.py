"""Draw a deterministic synthetic dashboard directly in its source geometry."""
from pathlib import Path
import json

from PIL import Image, ImageDraw, ImageFont

CASE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1600, 1000
TARGET_BOX = (1360, 760, 1488, 800)
INK = "#152239"
MUTED = "#738198"
LINE = "#E5EAF2"


def font(size, bold=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main():
    CASE.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (WIDTH, HEIGHT), "#F4F6FA")
    d = ImageDraw.Draw(im)

    def text(x, y, value, size=18, color=INK, bold=False):
        d.text((x, y), value, font=font(size, bold), fill=color)

    def box(bounds, fill="white", radius=12, outline=None):
        d.rounded_rectangle(bounds, radius, fill=fill, outline=outline, width=1)

    def center(bounds, value, size=17, color=INK, bold=False):
        x0, y0, x1, y1 = bounds
        f = font(size, bold)
        b = d.textbbox((0, 0), value, font=f)
        d.text(((x0+x1-b[2]+b[0])/2-b[0], (y0+y1-b[3]+b[1])/2-b[1]), value, font=f, fill=color)

    def button(bounds, value, primary=False):
        box(bounds, "#4961DB" if primary else "#FFFFFF", 8, None if primary else "#CFD8E7")
        center(bounds, value, 17, "white" if primary else "#344B9B", True)

    # Persistent navigation and workspace identity.
    d.rectangle((0, 0, 235, HEIGHT), fill="#142238")
    box((26, 29, 63, 66), "#6577EF", 9)
    center((26, 29, 63, 66), "N", 23, "white", True)
    text(76, 33, "NORTHSTAR", 20, "#F5F7FF", True)
    box((22, 106, 213, 165), "#20324C", 9)
    text(38, 117, "Acme Operations", 16, "#F5F7FF", True)
    text(38, 140, "Enterprise workspace", 12, "#A9B8CF")
    text(28, 199, "WORKSPACE", 12, "#8194B0", True)
    for i, label in enumerate(["Overview", "Pipelines", "Job runs", "Schedules", "Connections", "Activity log"]):
        top = 231 + i * 56
        if label == "Job runs":
            box((16, top, 220, top+44), "#314877", 8)
        d.rounded_rectangle((32, top+13, 48, top+29), 3, outline="#A8B9D3", width=2)
        text(62, top+10, label, 17, "#FFFFFF" if label == "Job runs" else "#BDC9DC", label == "Job runs")
    box((22, 806, 212, 904), "#20324C", 10)
    d.ellipse((38, 826, 47, 835), fill="#47D6AD")
    text(57, 820, "All systems online", 14, "#DDE6F6")
    text(38, 852, "API latency  42 ms", 13, "#A6B7D0")
    text(38, 874, "View status page", 13, "#A6B7D0")
    text(28, 948, "Settings", 16, "#BDC9DC")

    # Main header.
    d.rectangle((236, 0, 1599, 85), fill="white")
    d.line((236, 85, 1599, 85), fill=LINE)
    text(276, 32, "Workspace  /  Operations  /  Job runs", 17, MUTED)
    box((1113, 23, 1423, 62), "#F6F8FB", 8, LINE)
    text(1130, 33, "Search workspace...", 14, MUTED)
    d.ellipse((1480, 23, 1520, 63), fill="#E9EAFE")
    center((1480, 23, 1520, 63), "AL", 14, "#4E58A6", True)
    text(276, 117, "Job runs", 34, INK, True)
    text(276, 169, "Monitor pipeline execution and manage recent runs.", 18, MUTED)
    button((1360, 125, 1536, 173), "+  Run job", True)

    # Summary cards: deliberate dashboard content, no answer markers.
    cards = [(276, 676, "Total runs", "1,284", "+12.8% this week", "#2A9273"),
             (706, 1106, "Success rate", "98.6%", "+0.4% this week", "#2A9273"),
             (1136, 1536, "Needs attention", "2", "Failed runs in this view", "#B26A28")]
    for left, right, label, number, detail, color in cards:
        box((left, 221, right, 354), "white", 12, LINE)
        text(left+24, 239, label, 16, MUTED)
        text(left+24, 269, number, 34, INK, True)
        text(left+24, 319, detail, 14, color)
        for j, h in enumerate([14, 26, 20, 38, 31, 47, 42]):
            x = right-111+j*11
            box((x, 308-h, x+6, 308), "#D7DDF7", 2)

    # Activity panel with shared row geometry and repeated row controls.
    box((276, 385, 1536, 945), "white", 12, LINE)
    text(300, 406, "Recent runs", 23, INK, True)
    text(300, 443, "Latest execution for each pipeline and environment", 15, MUTED)
    button((1332, 405, 1512, 445), "Export log")
    box((300, 479, 686, 517), "#F7F9FC", 7, LINE)
    text(315, 488, "Search by pipeline name", 14, MUTED)
    button((984, 479, 1240, 517), "All environments")
    button((1256, 479, 1512, 517), "Last 24 hours")
    d.rectangle((277, 535, 1535, 559), fill="#F8FAFD")
    for x, label in [(322, "PIPELINE"), (698, "ENVIRONMENT"), (928, "STATUS"), (1118, "STARTED"), (1376, "ACTION")]:
        text(x, 539, label, 12, MUTED, True)
    rows = [
        ("Harbor ingestion", "run_hbr_0418", "Production", "Completed", "10:48 AM", "View", "3m 12s"),
        ("Meridian sync", "run_mrd_0731", "Staging", "Failed", "10:42 AM", "Retry", "18s"),
        ("Meridian sync", "run_mrd_0730", "Production", "Failed", "10:39 AM", "Retry", "21s"),
        ("Atlas export", "run_atl_0154", "Production", "Running", "10:35 AM", "View", "6m 08s"),
    ]
    for i, (name, run, env, status, started, action, elapsed) in enumerate(rows):
        top = 560 + 88*i
        d.line((300, top+87, 1512, top+87), fill=LINE)
        text(322, top+19, name, 19, INK, True)
        text(322, top+49, run, 13, MUTED)
        env_box = (698, top+26, 858, top+60)
        box(env_box, "#F0F2FA" if env == "Production" else "#F5F0FD", 7)
        center(env_box, env, 14, "#5F6882" if env == "Production" else "#8362A9")
        fill, fg = {"Completed": ("#E8F7F0", "#237F5D"), "Failed": ("#FFF0EF", "#BA4B45"), "Running": ("#ECF1FF", "#4265B3")}[status]
        status_box = (928, top+26, 1068, top+60)
        box(status_box, fill, 15)
        center(status_box, status, 14, fg, True)
        text(1118, top+20, started, 16)
        text(1118, top+49, elapsed, 13, MUTED)
        bounds = (1360, top+24, 1488, top+64)
        button(bounds, action)
        if i == 2:
            assert bounds == TARGET_BOX
    text(300, 923, "Showing 4 recent runs", 12, MUTED)
    text(1288, 923, "Updated a few seconds ago", 12, MUTED)
    text(276, 969, "Northstar Cloud  /  Operations console", 12, MUTED)
    text(1350, 969, "Help  /  Documentation", 12, MUTED)

    im.save(CASE / "input.png", optimize=True)
    request = {"question": "Locate the Retry button in the Meridian sync row whose environment is Production. Return its center as normalized (x, y) coordinates in the original screenshot.", "images": ["input.png"], "options": []}
    x0, y0, x1, y1 = TARGET_BOX
    expected = {"answer": f"{(x0+x1)/2/WIDTH:.4f}, {(y0+y1)/2/HEIGHT:.4f}", "match": "point", "tolerance": 0.01, "region": [x0/WIDTH, y0/HEIGHT, x1/WIDTH, y1/HEIGHT]}
    for name, obj in [("request.json", request), ("expected.json", expected)]:
        (CASE/name).write_text(json.dumps(obj, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
