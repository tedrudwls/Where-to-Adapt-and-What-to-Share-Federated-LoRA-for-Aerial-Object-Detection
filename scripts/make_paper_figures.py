#!/usr/bin/env python3
"""Generate 360-dpi PNG figures from the audited primary results."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures"
OUT.mkdir(exist_ok=True)

BLUE = "#00629B"
CYAN = "#00A6D6"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
GRAY = "#58595B"
LIGHT = "#F3F6F8"
BLACK = "#1A1A1A"
WHITE = "#FFFFFF"

FONT = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
FONT_ITALIC = Path("/System/Library/Fonts/Supplemental/Arial Italic.ttf")


def font(size: int, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_ITALIC if italic else FONT
    return ImageFont.truetype(str(path), size=size)


def centered_multiline(draw: ImageDraw.ImageDraw, box, text: str, fnt, fill=BLACK, spacing=8) -> None:
    x0, y0, x1, y1 = box
    bb = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=spacing, align="center")
    width, height = bb[2] - bb[0], bb[3] - bb[1]
    draw.multiline_text(((x0 + x1 - width) / 2, (y0 + y1 - height) / 2), text,
                        font=fnt, fill=fill, spacing=spacing, align="center")


def rounded_box(draw, box, text, face, edge=GRAY, fnt=None, radius=18, width=3) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=face, outline=edge, width=width)
    centered_multiline(draw, box, text, fnt or font(28), spacing=7)


def arrow(draw, start, end, fill=GRAY, width=4, both=False) -> None:
    draw.line([start, end], fill=fill, width=width)

    def head(a, b):
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        length = 18
        spread = 0.55
        p1 = (b[0] - length * math.cos(angle - spread), b[1] - length * math.sin(angle - spread))
        p2 = (b[0] - length * math.cos(angle + spread), b[1] - length * math.sin(angle + spread))
        draw.polygon([b, p1, p2], fill=fill)

    head(start, end)
    if both:
        head(end, start)


def save(img: Image.Image, stem: str) -> None:
    img.save(OUT / f"{stem}.png", dpi=(360, 360), optimize=True)


def framework() -> None:
    img = Image.new("RGB", (2578, 1602), WHITE)
    draw = ImageDraw.Draw(img)
    draw.text((100, 70), "RT-DETR-L dual-target adaptation", font=font(42, bold=True), fill=BLUE)

    boxes = [
        ((100, 245, 560, 445), "CNN feature extractor\n42 Conv-LoRA targets", "#DCEFF7", BLUE),
        ((700, 245, 1090, 445), "Hybrid encoder\n(frozen base)", "#EEF1F3", GRAY),
        ((1230, 245, 1765, 445), "Transformer decoder\n6 fused Q/K/V +\n6 cross-value targets", "#E5F5EC", GREEN),
        ((1910, 245, 2400, 445), "4-class task head\n(shared in every FL mode)", "#FFF0D9", ORANGE),
    ]
    for box, text, face, edge in boxes:
        rounded_box(draw, box, text, face, edge, font(29, bold="task" in text))
    for x1, x2 in [(560, 700), (1090, 1230), (1765, 1910)]:
        arrow(draw, (x1 + 20, 345), (x2 - 20, 345), width=5)

    rounded_box(draw, (800, 630, 1778, 865),
                "Federated server\nSample-count-weighted FedAvg\nof the designated shared state",
                "#EAF3F8", BLUE, font(34, bold=True), width=4)

    xs = [90, 865, 1640]
    for idx, x in enumerate(xs):
        rounded_box(draw, (x, 1110, x + 650, 1450),
                    f"Client {idx}\nNon-IID local images\nTrain A, B, and task head\nRetain designated local factor",
                    LIGHT, GRAY, font(29), width=3)
        arrow(draw, (1289, 865), (x + 325, 1090), fill=CYAN, width=6, both=True)

    draw.text((100, 930), "Round payload (download + upload)", font=font(31, bold=True), fill=GRAY)
    legend = [
        (100, "FL LoRA: A + B + head", BLUE),
        (860, "FedSA-LoRA: A + head; B local", GREEN),
        (1690, "Fixed Share-B: B + head; A local", ORANGE),
    ]
    for x, text, color in legend:
        draw.line((x, 1020, x + 70, 1020), fill=color, width=14)
        draw.text((x + 90, 995), text, font=font(26), fill=BLACK)

    msg = "Raw images never leave a client; this data-locality property is not a formal privacy guarantee."
    bb = draw.textbbox((0, 0), msg, font=font(27, italic=True))
    draw.text(((2578 - (bb[2] - bb[0])) / 2, 1515), msg, font=font(27, italic=True), fill=RED)
    save(img, "framework")


def _axes(draw, bounds, x_label, y_label, y_min, y_max, x_ticks, y_ticks):
    left, top, right, bottom = bounds
    draw.line((left, top, left, bottom), fill=BLACK, width=4)
    draw.line((left, bottom, right, bottom), fill=BLACK, width=4)
    for _, xpos, label in x_ticks:
        draw.line((xpos, bottom, xpos, bottom + 12), fill=BLACK, width=3)
        bb = draw.textbbox((0, 0), label, font=font(25))
        draw.text((xpos - (bb[2] - bb[0]) / 2, bottom + 20), label, font=font(25), fill=BLACK)
    for value in y_ticks:
        ypos = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
        draw.line((left, ypos, right, ypos), fill="#D6DADD", width=2)
        label = f"{value:.2f}"
        bb = draw.textbbox((0, 0), label, font=font(24))
        draw.text((left - (bb[2] - bb[0]) - 18, ypos - 14), label, font=font(24), fill=BLACK)
    bb = draw.textbbox((0, 0), x_label, font=font(29))
    draw.text(((left + right - (bb[2] - bb[0])) / 2, bottom + 80), x_label, font=font(29), fill=BLACK)
    label_img = Image.new("RGBA", (900, 70), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((0, 0), y_label, font=font(29), fill=BLACK)
    return label_img.rotate(90, expand=True)


def communication_tradeoff() -> None:
    methods = ["FL Full FT", "FL LoRA", "FedSA-LoRA", "Fixed Share-B"]
    communication = [15783.2352, 212.78016, 131.67936, 85.0464]
    means = [0.6746, 0.6541, 0.5357, 0.5629]
    sds = [0.0041, 0.0068, 0.0549, 0.0400]
    colors = [RED, BLUE, GREEN, ORANGE]
    markers = ["square", "circle", "triangle", "diamond"]
    img = Image.new("RGB", (2578, 1278), WHITE)
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = 300, 90, 2440, 1000
    xmin, xmax = math.log10(70), math.log10(20000)
    ymin, ymax = 0.45, 0.72
    xtick_vals = [100, 300, 1000, 3000, 10000]
    xticks = [(v, left + (math.log10(v) - xmin) / (xmax - xmin) * (right - left), str(v)) for v in xtick_vals]
    rot = _axes(draw, (left, top, right, bottom),
                "Cumulative 20-round tensor communication (MB, log scale)",
                "Common pooled-test AP", ymin, ymax, xticks,
                [0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    img.paste(rot, (55, int((top + bottom - rot.height) / 2)), rot)

    for name, xval, yval, sd, color, marker in zip(methods, communication, means, sds, colors, markers):
        x = left + (math.log10(xval) - xmin) / (xmax - xmin) * (right - left)
        y = bottom - (yval - ymin) / (ymax - ymin) * (bottom - top)
        y1 = bottom - (yval - sd - ymin) / (ymax - ymin) * (bottom - top)
        y2 = bottom - (yval + sd - ymin) / (ymax - ymin) * (bottom - top)
        draw.line((x, y1, x, y2), fill=color, width=5)
        draw.line((x - 14, y1, x + 14, y1), fill=color, width=5)
        draw.line((x - 14, y2, x + 14, y2), fill=color, width=5)
        if marker == "circle":
            draw.ellipse((x - 16, y - 16, x + 16, y + 16), fill=color, outline=BLACK, width=2)
        elif marker == "square":
            draw.rectangle((x - 16, y - 16, x + 16, y + 16), fill=color, outline=BLACK, width=2)
        elif marker == "triangle":
            draw.polygon([(x, y - 19), (x - 19, y + 17), (x + 19, y + 17)], fill=color, outline=BLACK)
        else:
            draw.polygon([(x, y - 20), (x - 20, y), (x, y + 20), (x + 20, y)], fill=color, outline=BLACK)
        dx, dy = (25, -45) if name != "FedSA-LoRA" else (25, 25)
        draw.text((x + dx, y + dy), name, font=font(27, bold=True), fill=color)

    draw.text((330, 1100), "Points: mean across three paired seeds; bars: run sample SD",
              font=font(25, italic=True), fill=GRAY)
    save(img, "communication_utility_tradeoff")


def minimum_positive_support() -> None:
    methods = ["Local LoRA", "FL LoRA", "FedSA-LoRA", "Fixed Share-B"]
    means = [0.4170, 0.6923, 0.4544, 0.5207]
    sds = [0.0361, 0.0209, 0.0388, 0.0161]
    colors = [GRAY, BLUE, GREEN, ORANGE]
    img = Image.new("RGB", (2578, 1224), WHITE)
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = 280, 90, 2440, 960
    ymin, ymax = 0.0, 0.80
    centers = [left + (idx + 0.5) * (right - left) / 4 for idx in range(4)]
    xticks = [(idx, center, name) for idx, (name, center) in enumerate(zip(methods, centers))]
    rot = _axes(draw, (left, top, right, bottom), "Method",
                "Minimum-positive-support common-test AP", ymin, ymax, xticks,
                [0.0, 0.2, 0.4, 0.6, 0.8])
    img.paste(rot, (35, int((top + bottom - rot.height) / 2)), rot)

    width = 260
    for center, value, sd, color in zip(centers, means, sds, colors):
        y = bottom - value / ymax * (bottom - top)
        draw.rectangle((center - width / 2, y, center + width / 2, bottom), fill=color, outline=BLACK, width=3)
        y_top = bottom - (value + sd) / ymax * (bottom - top)
        y_bottom = bottom - (value - sd) / ymax * (bottom - top)
        draw.line((center, y_top, center, y_bottom), fill=BLACK, width=4)
        draw.line((center - 18, y_top, center + 18, y_top), fill=BLACK, width=4)
        draw.line((center - 18, y_bottom, center + 18, y_bottom), fill=BLACK, width=4)
        label = f"{value:.4f}"
        bb = draw.textbbox((0, 0), label, font=font(26, bold=True))
        draw.text((center - (bb[2] - bb[0]) / 2, y_top - 48), label,
                  font=font(26, bold=True), fill=BLACK)
    draw.text((310, 1060), "Mean ± run sample SD over three paired seeds",
              font=font(25, italic=True), fill=GRAY)
    save(img, "minimum_positive_support")


def main() -> None:
    framework()
    communication_tradeoff()
    minimum_positive_support()
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
