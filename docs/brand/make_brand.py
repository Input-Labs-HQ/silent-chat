"""Input Labs mark: a text cursor (I-beam) whose stem is a wave, meaning new ways to give a computer input (lips,
voice, eyes), in the same blue-to-violet as Silent Chat. Writes an SVG and PNGs from the same geometry.

  .venv/bin/python docs/brand/make_brand.py
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
S = 1024
TOP, BOT = (91, 140, 255), (124, 63, 228)  # gradient, top-left -> bottom-right
SERIF_W, SERIF_H = 300, 64
Y_TOP, Y_BOT = 236, 788  # serif centres
STROKE, AMP, CYCLES = 62, 74, 1.5


def wave_points(n=240):
    """The stem: a wave from the top serif to the bottom one, flat where it meets them."""
    y0, y1 = Y_TOP + SERIF_H / 2, Y_BOT - SERIF_H / 2
    pts = []
    for i in range(n + 1):
        u = i / n
        env = math.sin(math.pi * u) ** 1.4
        x = S / 2 + AMP * env * math.sin(2 * math.pi * CYCLES * u)
        pts.append((x, y0 + (y1 - y0) * u))
    return pts


def svg():
    pts = wave_points()
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    sx = S / 2 - SERIF_W / 2
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {S} {S}" width="{S}" height="{S}">
  <title>Input Labs</title>
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="rgb{TOP}"/>
      <stop offset="1" stop-color="rgb{BOT}"/>
    </linearGradient>
  </defs>
  <rect width="{S}" height="{S}" rx="230" fill="url(#g)"/>
  <rect x="{sx}" y="{Y_TOP - SERIF_H / 2}" width="{SERIF_W}" height="{SERIF_H}" rx="{SERIF_H / 2}" fill="#fff"/>
  <rect x="{sx}" y="{Y_BOT - SERIF_H / 2}" width="{SERIF_W}" height="{SERIF_H}" rx="{SERIF_H / 2}" fill="#fff"/>
  <path d="{d}" fill="none" stroke="#fff" stroke-width="{STROKE}" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""


def icon(size, rounded=True):
    k = 4
    n = S * k
    y, x = np.mgrid[0:n, 0:n] / n
    t = ((x + y) / 2)[..., None]
    rgb = (np.array(TOP) * (1 - t) + np.array(BOT) * t).astype(np.uint8)
    img = Image.fromarray(rgb).convert("RGBA")
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, n - 1, n - 1], radius=230 * k if rounded else 0, fill=255)
    img.putalpha(mask)
    d = ImageDraw.Draw(img)
    for yc in (Y_TOP, Y_BOT):
        d.rounded_rectangle([(S / 2 - SERIF_W / 2) * k, (yc - SERIF_H / 2) * k, (S / 2 + SERIF_W / 2) * k,
                             (yc + SERIF_H / 2) * k], radius=SERIF_H / 2 * k, fill="white")
    r = STROKE / 2 * k
    for (x0, y0), (x1, y1) in zip(wave_points(600), wave_points(600)[1:]):
        d.line([(x0 * k, y0 * k), (x1 * k, y1 * k)], fill="white", width=int(STROKE * k))
    for xx, yy in wave_points(600)[::6] + [wave_points()[0], wave_points()[-1]]:
        d.ellipse([xx * k - r, yy * k - r, xx * k + r, yy * k + r], fill="white")
    return img.resize((size, size), Image.LANCZOS)


def wordmark(dark):
    """Icon + "Input Labs", for READMEs, slides and the web (dark or light background)."""
    h = 240
    ic = icon(h)
    f = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 132)
    f.set_variation_by_name("Bold")
    text = "Input Labs"
    tw = ImageDraw.Draw(ic).textlength(text, font=f)
    W = int(h + 56 + tw + 24)
    img = Image.new("RGBA", (W, h), (0, 0, 0, 0))
    img.alpha_composite(ic, (0, 0))
    ImageDraw.Draw(img).text((h + 56, h / 2 - 84), text, font=f, fill=(255, 255, 255) if dark else (24, 24, 34))
    return img


if __name__ == "__main__":
    (HERE / "input-labs.svg").write_text(svg())
    big = icon(1024)
    big.save(HERE / "input-labs-1024.png")
    for s in (512, 256, 64):
        big.resize((s, s), Image.LANCZOS).save(HERE / f"input-labs-{s}.png")
    icon(1024, rounded=False).resize((500, 500), Image.LANCZOS).convert("RGB").save(HERE / "input-labs-avatar.png")
    wordmark(dark=False).save(HERE / "input-labs-wordmark-light.png")
    wordmark(dark=True).save(HERE / "input-labs-wordmark-dark.png")
    print("written to", HERE)
