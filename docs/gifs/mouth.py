"""A clean, icon-style animated mouth: white lips on the app icon's gradient squircle, drawn 4x and scaled down."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SS = 4


def _squircle(size):
    n = size * SS
    y, x = np.mgrid[0:n, 0:n] / n
    top, bot = np.array([90, 140, 255]), np.array([128, 70, 235])  # (the icon: blue at the top, violet below)
    t = (0.75 * y + 0.25 * x)[..., None]
    rgb = (top * (1 - t) + bot * t).astype(np.uint8)
    img = Image.fromarray(rgb).convert("RGBA")
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, n - 1, n - 1], radius=int(n * 0.23), fill=255)
    img.putalpha(mask)
    return img


def mouth(open_, round_, size=240):
    """open_ 0..1 (closed .. wide), round_ 0..1 (spread .. rounded 'oo')."""
    img = _squircle(size)
    d = ImageDraw.Draw(img)
    n = size * SS
    cx, cy = n / 2, n / 2 + n * 0.02
    w = n * (0.33 - 0.08 * round_)  # half-width
    gap = n * 0.17 * open_
    xs = np.linspace(-1, 1, 200)

    def env(x, p):  # lip shape: rounded ends
        return np.clip(1 - np.abs(x) ** p, 0, 1) ** 0.5

    seam = lambda x: n * 0.012 * (1 - x ** 2)  # (a soft upward curve, like the icon)
    up_th = lambda x: n * 0.085 * env(x, 2.2) * (1 + 0.28 * np.exp(-((np.abs(x) - 0.3) / 0.16) ** 2)
                                                 - 0.3 * np.exp(-(x / 0.09) ** 2))
    lo_th = lambda x: n * 0.105 * env(x, 2.0)
    open_curve = lambda x: gap / 2 * env(x, 1.6)
    up_inner = [(cx + x * w, cy - seam(x) - open_curve(x)) for x in xs]
    up_outer = [(cx + x * w, cy - seam(x) - open_curve(x) - up_th(x)) for x in xs[::-1]]
    lo_inner = [(cx + x * w, cy - seam(x) + open_curve(x)) for x in xs]
    lo_outer = [(cx + x * w, cy - seam(x) + open_curve(x) + lo_th(x)) for x in xs[::-1]]
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    off = n * 0.012
    sd.polygon([(x, y + off) for x, y in up_inner + up_outer], fill=(40, 10, 90, 70))
    sd.polygon([(x, y + off) for x, y in lo_inner + lo_outer], fill=(40, 10, 90, 70))
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(n * 0.012)))
    d = ImageDraw.Draw(img)
    if open_ > 0.02:  # the inside of the mouth, and a hint of teeth
        d.polygon(up_inner + lo_inner[::-1], fill=(52, 22, 96, 255))
    d.polygon(up_inner + up_outer, fill=(255, 255, 255, 255))
    d.polygon(lo_inner + lo_outer, fill=(255, 255, 255, 255))
    if open_ <= 0.02:  # closed: the line where the lips meet
        d.line([(cx + x * w * 0.96, cy - seam(x)) for x in xs], fill=(150, 120, 235, 255), width=int(n * 0.012))
    return img.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    import sys
    tiles = [mouth(o, r) for o, r in ((0, 0), (0.3, 0), (0.7, 0.1), (0.5, 0.8), (0.12, 0))]
    out = Image.new("RGBA", (240 * len(tiles) + 20 * (len(tiles) - 1), 240), (30, 30, 40, 255))
    for i, t in enumerate(tiles):
        out.alpha_composite(t, (i * 260, 0))
    out.save(sys.argv[1])
