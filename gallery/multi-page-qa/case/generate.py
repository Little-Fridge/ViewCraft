"""Generate a fictional four-page report with no network or model calls."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
W, H = 1200, 1600
INK, MUTED = '#203B45', '#63747A'
TEAL, PAPER, LINE = '#127D80', '#FAF8F2', '#D9E0DC'
ACTUAL, TARGET = 1240, 1500


def font(size, bold=False):
    try:
        return ImageFont.truetype('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def text(draw, xy, value, size=28, fill=INK, bold=False):
    draw.text(xy, value, font=font(size, bold), fill=fill)


def lines(draw, xy, values, size=27, fill=MUTED, gap=42):
    for i, value in enumerate(values):
        text(draw, (xy[0], xy[1] + i * gap), value, size, fill)


def base(page, section, title, subtitle):
    im = Image.new('RGB', (W, H), PAPER)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 18), fill=TEAL)
    text(d, (72, 61), 'FIELDNOTE / OPERATIONS', 24, TEAL, True)
    text(d, (890, 61), 'APRIL 2026', 24, MUTED)
    d.line((72, 120, 1128, 120), fill=LINE, width=2)
    text(d, (72, 161), section.upper(), 24, TEAL, True)
    text(d, (72, 211), title, 53, INK, True)
    text(d, (72, 293), subtitle, 27, MUTED)
    d.line((72, 1493, 1128, 1493), fill=LINE, width=2)
    text(d, (72, 1524), 'Fieldnote Assembly  /  Internal monthly brief', 21, MUTED)
    text(d, (1007, 1520), f'{page:02d} / 04', 23, TEAL, True)
    return im, d


def table(d, top, heads, rows, widths):
    left, height = 72, 95
    d.rounded_rectangle((left, top, 1128, top + 70), radius=12, fill=INK)
    for x, header in zip(widths, heads):
        text(d, (x, top + 21), header, 24, '#FFFFFF', True)
    for i, row in enumerate(rows):
        y = top + 70 + i * height
        if i % 2 == 0:
            d.rectangle((left, y, 1128, y + height), fill='#EDF2EE')
        d.line((left, y + height, 1128, y + height), fill=LINE, width=1)
        for x, cell in zip(widths, row):
            text(d, (x, y + 31), str(cell), 28)


def main():
    pages = []
    im, d = base(1, 'Report guide', 'Monthly operations brief', 'Assembly performance, people and delivery commitments')
    d.rounded_rectangle((72, 383, 1128, 682), radius=22, fill=INK)
    text(d, (108, 418), 'REPORT SCOPE', 23, '#93D5C9', True)
    text(d, (108, 474), 'April 1-30, 2026', 43, '#FFFFFF', True)
    lines(d, (108, 552), ['Production counts reflect the month-end close.', 'Planning figures are the approved monthly commitments.'], 26, '#D7E7E3')
    text(d, (72, 752), 'IN THIS REPORT', 25, TEAL, True)
    for y, num, title, desc in [
        (821, '02', 'Production close', 'Accepted units and items awaiting inspection'),
        (999, '03', 'People & coverage', 'Team size and scheduled assembly shifts'),
        (1177, '04', 'Delivery commitments', 'Approved accepted-unit targets by product'),
    ]:
        d.rounded_rectangle((72, y, 151, y + 76), radius=14, fill='#DDEDE5')
        text(d, (87, y + 18), num, 31, TEAL, True)
        text(d, (182, y + 3), title, 33, INK, True)
        text(d, (182, y + 59), desc, 26, MUTED)
        d.line((182, y + 134, 1128, y + 134), fill=LINE, width=2)
    pages.append(im)

    im, d = base(2, 'Production close', 'Completed and accepted', 'April 2026  /  Unit counts by product')
    text(d, (72, 389), 'MONTH-END REGISTER', 25, TEAL, True)
    table(d, 452, ['Product', 'Accepted units', 'Awaiting inspection'], [
        ['Atlas kit', f'{ACTUAL:,}', '85'],
        ['Beacon kit', '960', '40'],
        ['Cove kit', '780', '25'],
    ], [98, 475, 804])
    d.rounded_rectangle((72, 877, 1128, 1082), radius=20, fill='#E5EEE7')
    text(d, (107, 910), 'WHAT COUNTS AS ACCEPTED?', 25, TEAL, True)
    lines(d, (107, 964), ['Units that passed final inspection during April.', 'Items awaiting inspection are excluded from accepted totals.'], 27)
    text(d, (72, 1160), 'Closing notes', 32, INK, True)
    lines(d, (72, 1220), ['All three product lines use the same inspection standard.', 'Delivery targets appear in the commitments section.', 'Counts shown here are monthly, not year-to-date.'])
    pages.append(im)

    im, d = base(3, 'People & coverage', 'Assembly team coverage', 'April 2026  /  Staffing schedule')
    for x, value, label in [(72, '28', 'Assembly staff'), (620, '22', 'Working days')]:
        d.rounded_rectangle((x, 397, x + 508, 635), radius=22, fill='#E7EDE4')
        text(d, (x + 33, 426), value, 78, TEAL, True)
        text(d, (x + 33, 548), label, 29)
    table(d, 730, ['Team', 'Staff', 'Scheduled shifts'], [
        ['Day assembly', '16', '22'], ['Evening assembly', '8', '22'], ['Inspection', '4', '20']
    ], [98, 565, 804])
    text(d, (72, 1177), 'Coverage note', 32, INK, True)
    lines(d, (72, 1238), ['Staff counts describe available coverage only.', 'Each shift can support several product lines.', 'The schedule does not report completed unit counts.'])
    pages.append(im)

    im, d = base(4, 'Delivery commitments', 'Approved monthly targets', 'April 2026  /  Accepted units required by month end')
    text(d, (72, 389), 'PLANNING REGISTER', 25, TEAL, True)
    table(d, 452, ['Product', 'Target units', 'Due date'], [
        ['Atlas kit', f'{TARGET:,}', 'April 30'],
        ['Beacon kit', '1,100', 'April 30'],
        ['Cove kit', '900', 'April 30'],
    ], [98, 510, 852])
    d.rounded_rectangle((72, 877, 1128, 1082), radius=20, fill='#F1E9D6')
    text(d, (107, 910), 'TARGET BASIS', 25, '#8A662B', True)
    lines(d, (107, 964), ['Targets apply to accepted units for April only.', 'Use the production close to compare actual delivery.'], 27)
    text(d, (72, 1160), 'Planning notes', 32, INK, True)
    lines(d, (72, 1220), ['The monthly targets were approved before production began.', 'No carryover from prior months is included.', 'Targets share the product names used in the production register.'])
    pages.append(im)

    names = [f'page_{i:02d}.png' for i in range(1, 5)]
    for im, name in zip(pages, names):
        im.save(HERE / name)
    preview = Image.new('RGB', (1256, 892), '#E5E8E2')
    d = ImageDraw.Draw(preview)
    text(d, (25, 18), 'INPUT PREVIEW / Four-page operations brief', 26, INK, True)
    for i, im in enumerate(pages):
        preview.paste(im.resize((294, 392), Image.Resampling.LANCZOS), (25 + (i % 2) * 610, 69 + (i // 2) * 411))
        text(d, (333 + (i % 2) * 610, 86 + (i // 2) * 411), f'PAGE {i + 1:02d}', 22, TEAL, True)
        labels = [('Report', 'guide'), ('Production', 'close'), ('People &', 'coverage'), ('Delivery', 'commitments')][i]
        lines(d, (333 + (i % 2) * 610, 135 + (i // 2) * 411), labels, 24)
    preview.save(HERE / 'preview.png')
    request = {
        'question': 'According to the April 2026 report, how many additional accepted Atlas kits were needed to meet the monthly target? Answer with the number only.',
        'images': names,
        'options': [],
    }
    (HERE / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
    (HERE / 'expected.json').write_text(json.dumps({'answer': str(TARGET - ACTUAL), 'match': 'exact'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
