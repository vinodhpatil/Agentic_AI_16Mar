#!/usr/bin/env python3
"""Generate maskable PWA icons: a microphone-in-circle mark on the dark editorial bg.

Run: python3 generate_icons.py
Produces icon-192.png and icon-512.png in this folder.
"""
import os
from PIL import Image, ImageDraw

BG = (7, 7, 13)          # #07070d near-black
ACCENT = (62, 207, 207)  # #3ecfcf growth teal
DIM = (40, 40, 56)


def draw_icon(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(img)
    cx = cy = size / 2

    # Outer accent ring (kept within the maskable safe area ~80%).
    ring_r = size * 0.40
    ring_w = max(2, int(size * 0.025))
    d.ellipse(
        [cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r],
        outline=ACCENT,
        width=ring_w,
    )

    # Microphone capsule.
    mic_w = size * 0.16
    mic_h = size * 0.26
    mic_top = cy - size * 0.20
    mic_box = [cx - mic_w / 2, mic_top, cx + mic_w / 2, mic_top + mic_h]
    d.rounded_rectangle(mic_box, radius=mic_w / 2, fill=ACCENT)

    # Mic stand arc (the U-cradle) drawn as an arc.
    arc_r = size * 0.13
    arc_box = [cx - arc_r, cy - arc_r * 0.55, cx + arc_r, cy + arc_r * 1.1]
    d.arc(arc_box, start=20, end=160, fill=ACCENT, width=max(2, int(size * 0.022)))

    # Stem + base.
    stem_top = cy + arc_r * 1.05
    stem_bottom = cy + size * 0.20
    d.line([cx, stem_top, cx, stem_bottom], fill=ACCENT, width=max(2, int(size * 0.022)))
    base_w = size * 0.12
    d.line(
        [cx - base_w / 2, stem_bottom, cx + base_w / 2, stem_bottom],
        fill=ACCENT,
        width=max(2, int(size * 0.022)),
    )
    return img


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for size in (192, 512):
        img = draw_icon(size)
        out = os.path.join(here, f"icon-{size}.png")
        img.save(out, "PNG")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
