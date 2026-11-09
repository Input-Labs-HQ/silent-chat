"""Renders the README's feature GIFs. The island is the real app's (island.py), drawn offscreen frame by frame at
exact times; the mouth is an animated icon-style drawing; every window, name and message is made up.

  .venv/bin/python docs/gifs/make_gifs.py            # all
  .venv/bin/python docs/gifs/make_gifs.py discord    # one (discord, split, spotify, misread, privacy)

Needs ffmpeg. Album covers come from Spotify's image CDN (artwork © its owners).
"""
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import mouth as mouth_mod  # noqa: E402

FPS = 15
FONT = "/System/Library/Fonts/SFNS.ttf"
COVERS = {"gods_plan": "https://i.scdn.co/image/ab67616d0000b273f907de96b9a4fbc04accc0d5"}


def font(size, weight="Regular"):
    f = ImageFont.truetype(FONT, size)
    f.set_variation_by_name(weight)
    return f


# ----------------------------------------------------------------------------- the island, offscreen

class Film:
    """Draws island.py's real island at a given time and state, without ever showing it on screen."""

    def __init__(self):
        from AppKit import NSApplication
        import island
        self.island_mod = island
        self.app = NSApplication.sharedApplication()
        self.app.setActivationPolicy_(1)
        self.clock = [1000.0]
        island.time.monotonic = lambda: self.clock[0]

    def new(self):
        I = self.island_mod.Island()
        I.panel.orderFrontRegardless = lambda: None
        I.panel.setAlphaValue_(0.0)
        return I

    def frame(self, I, t, **kw):
        import time as _t
        self.clock[0] = 1000.0 + t
        I.update(**kw)
        if kw.get("playing") and I.art is None:
            for _ in range(30):  # (the cover loads in the background)
                _t.sleep(0.1)
                if I.art is not None:
                    break
            I.update(**kw)
        if not I.shown:
            return None
        from AppKit import NSPNGFileType
        v = I.view
        rep = v.bitmapImageRepForCachingDisplayInRect_(v.bounds())
        v.cacheDisplayInRect_toBitmapImageRep_(v.bounds(), rep)
        data = rep.representationUsingType_properties_(NSPNGFileType, {})
        import io
        im = Image.open(io.BytesIO(bytes(data))).convert("RGBA")
        return im.resize((im.width // 2, im.height // 2), Image.LANCZOS)


# ----------------------------------------------------------------------------- the mouth

def mouth_track(text, start, syl=0.17):
    """Keyframes (t, open, round) mouthing `text` from `start`; -> (keys, end time)."""
    keys, t = [(0.0, 0, 0), (start, 0, 0)], start
    for word in re.findall(r"[a-z']+|\d+", text.lower()):
        if word.isdigit():
            word = {"4": "four", "2": "two", "6": "six"}.get(word, "one")
        sylls = re.findall(r"[^aeiouy]*[aeiouy]+(?:[^aeiouy]*$)?", word) or [word]
        for s in sylls:
            v = (re.search(r"[aeiouy]+", s) or re.search(r".", s)).group(0)
            o = {"a": .8, "o": .65, "e": .5, "i": .4, "u": .35, "y": .4}.get(v[0], .4)
            r = .8 if v[0] in "ou" or "w" in s else 0.1
            if s[:1] in "pbm":
                keys.append((t + 0.02, 0, 0))
            elif s[:1] in "fv":
                keys.append((t + 0.02, 0.06, 0))
            keys.append((t + syl * 0.55, o, r))
            keys.append((t + syl * 0.95, o * 0.35, r * 0.5))
            t += syl
        t += 0.04
    keys += [(t + 0.12, 0, 0), (999, 0, 0)]
    return keys, t + 0.12


def mouth_at(keys, t):
    for (t0, o0, r0), (t1, o1, r1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            a = (t - t0) / max(1e-6, t1 - t0)
            a = a * a * (3 - 2 * a)
            return o0 + (o1 - o0) * a, r0 + (r1 - r0) * a
    return 0, 0


# ----------------------------------------------------------------------------- the scene

def wallpaper(W, H):
    y, x = np.mgrid[0:H, 0:W]
    u, v = x / W, y / H
    bg = np.clip(np.stack([40 + 90 * u + 40 * np.exp(-((u - .75) ** 2 + (v - .9) ** 2) * 6),
                           60 + 40 * (1 - v) + 30 * np.exp(-((u - .25) ** 2 + (v - 1) ** 2) * 5),
                           150 + 80 * (1 - u * .5) - 30 * v], -1), 0, 255).astype(np.uint8)
    return Image.fromarray(bg).convert("RGBA")


def menubar(img, app, items=("File", "Edit", "View", "Window")):
    W = img.width
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(lay).rectangle([0, 0, W, 28], fill=(0, 0, 0, 50))
    img.alpha_composite(lay)
    d = ImageDraw.Draw(img)
    f, fb = font(13, "Medium"), font(13, "Bold")
    x = 18
    d.text((x, 6), "", font=font(14), fill="white")
    x += 28
    for i, it in enumerate((app,) + tuple(items)):
        d.text((x, 6), it, font=fb if i == 0 else f, fill="white")
        x += d.textlength(it, font=fb if i == 0 else f) + 16
    d.text((W - 18 - d.textlength("Mon 3:41 PM", font=f), 6), "Mon 3:41 PM", font=f, fill="white")


def window(img, box, title, alpha=1.0, body=None, dark=False):
    """A macOS-style window: shadow, rounded corners, traffic lights, title; body(draw, inner box)."""
    x0, y0, x1, y1 = [int(v) for v in box]
    if alpha <= 0.01 or x1 - x0 < 20:
        return
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle([x0 + 2, y0 + 8, x1 + 2, y1 + 10], radius=12, fill=(0, 0, 0, 90))
    lay.alpha_composite(sh.filter(ImageFilter.GaussianBlur(10)))
    d = ImageDraw.Draw(lay)
    bgc = (32, 33, 38, 255) if dark else (250, 250, 252, 255)
    d.rounded_rectangle([x0, y0, x1, y1], radius=12, fill=bgc)
    d.rounded_rectangle([x0, y0, x1, y0 + 30], radius=12, fill=(44, 45, 52, 255) if dark else (236, 236, 240, 255))
    d.rectangle([x0, y0 + 18, x1, y0 + 30], fill=(44, 45, 52, 255) if dark else (236, 236, 240, 255))
    for i, c in enumerate(((255, 95, 87), (254, 188, 46), (40, 200, 64))):
        d.ellipse([x0 + 12 + i * 18, y0 + 10, x0 + 24 + i * 18, y0 + 22], fill=c + (255,))
    tf = font(12, "Semibold")
    d.text(((x0 + x1) / 2 - d.textlength(title, font=tf) / 2, y0 + 8), title, font=tf,
           fill=(220, 220, 228) if dark else (60, 60, 70))
    if body:
        body(d, (x0, y0 + 30, x1, y1), lay)
    if alpha < 1:
        a = lay.getchannel("A").point(lambda p: int(p * alpha))
        lay.putalpha(a)
    img.alpha_composite(lay)


def caption(img, text, size=19):
    W, H = img.size
    cf = font(size, "Semibold")
    d = ImageDraw.Draw(img)
    tw = d.textlength(text, font=cf)
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(lay).rounded_rectangle([W / 2 - tw / 2 - 16, H - 56, W / 2 + tw / 2 + 16, H - 20], radius=18,
                                          fill=(0, 0, 0, 115))
    img.alpha_composite(lay)
    ImageDraw.Draw(img).text((W / 2 - tw / 2, H - 51), text, font=cf, fill="white")


def mouth_tile(img, keys, t, x, y, size=170, nod=0.0, label=None):
    o, r = mouth_at(keys, t)
    tile = mouth_mod.mouth(o, r, size=size)
    img.alpha_composite(tile, (int(x), int(y + nod)))
    if label:
        talking, text = label
        lf = font(12, "Semibold")
        d = ImageDraw.Draw(img)
        lw = d.textlength(text, font=lf)
        lx = x + size / 2 - lw / 2 + (7 if talking else 0)
        if talking:
            d.ellipse([lx - 13, y + size + 13, lx - 5, y + size + 21], fill=(255, 95, 95))
        d.text((lx, y + size + 8), text, font=lf, fill=(240, 240, 250))


def ease(a):
    a = max(0.0, min(1.0, a))
    return a * a * (3 - 2 * a)


def encode(frames_dir, out):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", f"{frames_dir}/f%03d.png",
                    "-vf", "palettegen=max_colors=200:stats_mode=diff", f"{frames_dir}/pal.png"], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", f"{frames_dir}/f%03d.png",
                    "-i", f"{frames_dir}/pal.png", "-lavfi", "paletteuse=dither=sierra2_4a:diff_mode=rectangle",
                    "-loop", "0", str(out)], check=True)


def film(name, duration, W, H, draw, film_):
    """draw(t, island) -> PIL RGB frame. Writes docs/gif-<name>.gif."""
    I = film_.new()
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(int(duration * FPS)):
            t = i / FPS
            draw(t, I).convert("RGB").save(f"{tmp}/f{i:03d}.png")
        out = ROOT / "docs" / f"gif-{name}.gif"
        encode(tmp, out)
    print(f"{out.name}: {out.stat().st_size // 1024} KB")


def put_island(img, im):
    if im is not None:
        img.alpha_composite(im, (int(img.width / 2 - im.width / 2), 0))
    else:
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([img.width / 2 - 46, -12, img.width / 2 + 46, 16], radius=8, fill="black")


# ----------------------------------------------------------------------------- the GIFs

def gif_discord(F):
    """Mouth 'open Discord and message Alex …': it opens the chat, writes it, asks, you nod, it's sent."""
    W, H = 960, 520
    phrase = "open Discord and message Alex that I'll be available to meet at 4 PM"
    keys, te = mouth_track(phrase, 0.7)
    t_open, t_type0, t_type1, t_ask, t_nod, t_sent = te + 0.3, te + 0.9, te + 2.0, te + 2.2, te + 3.3, te + 4.1
    msg = "I'll be available to meet at 4 PM."
    base = wallpaper(W, H)

    def notes(d, b, lay):
        x0, y0, x1, y1 = b
        tf = font(15, "Bold")
        d.text((x0 + 22, y0 + 18), "Project notes", font=tf, fill=(40, 40, 50))
        for k, line in enumerate(["Draft the launch post", "Collect feedback from testers", "Fix the settings layout"]):
            d.text((x0 + 22, y0 + 52 + k * 24), "•  " + line, font=font(13), fill=(70, 70, 80))

    def chat(typed, sent):
        def body(d, b, lay):
            x0, y0, x1, y1 = b
            d.rectangle([x0, y0, x0 + 150, y1 - 12], fill=(38, 39, 45, 255))
            for k, (n, c) in enumerate((("Alex", (88, 101, 242)), ("Sam", (59, 165, 93)), ("Jordan", (237, 66, 69)))):
                yy = y0 + 16 + k * 40
                if n == "Alex":
                    d.rounded_rectangle([x0 + 8, yy - 6, x0 + 142, yy + 28], radius=6, fill=(60, 62, 70, 255))
                d.ellipse([x0 + 16, yy, x0 + 38, yy + 22], fill=c + (255,))
                d.text((x0 + 22, yy + 3), n[0], font=font(12, "Bold"), fill="white")
                d.text((x0 + 46, yy + 3), n, font=font(13, "Medium"), fill=(225, 225, 232))
            d.text((x0 + 170, y0 + 12), "@ Alex", font=font(14, "Bold"), fill=(235, 235, 240))
            d.line([x0 + 160, y0 + 40, x1 - 10, y0 + 40], fill=(60, 60, 68), width=1)
            rows = [("Alex", (88, 101, 242), "Are you free to meet later today?")]
            if sent:
                rows.append(("You", (150, 120, 235), msg))
            for k, (n, c, text) in enumerate(rows):
                yy = y0 + 56 + k * 52
                d.ellipse([x0 + 170, yy, x0 + 200, yy + 30], fill=c + (255,))
                d.text((x0 + 210, yy - 1), n, font=font(13, "Bold"), fill=(235, 235, 240))
                d.text((x0 + 210, yy + 17), text, font=font(13), fill=(205, 205, 212))
            fy = y1 - 52
            d.rounded_rectangle([x0 + 166, fy, x1 - 14, fy + 36], radius=8, fill=(56, 58, 66, 255))
            d.text((x0 + 180, fy + 9), typed or "Message @Alex", font=font(13),
                   fill=(230, 230, 236) if typed else (140, 140, 150))
        return body

    def draw(t, I):
        img = base.copy()
        opened = t >= t_open
        menubar(img, "Discord" if opened else "Notes")
        window(img, (360, 70, 900, 400), "Notes", body=notes)
        if opened:
            a = ease((t - t_open) / 0.35)
            typed = msg[:int(len(msg) * ease((t - t_type0) / (t_type1 - t_type0)))] if t >= t_type0 and t < t_sent else ""
            window(img, (300 + 40 * (1 - a), 64 + 16 * (1 - a), 920, 420), "Discord", alpha=a,
                   body=chat(typed, t >= t_sent), dark=True)
        if t < 0.7:
            kw = dict(visible=False)
        elif t < te:
            kw = dict(ind="mouth", amp=0.4 + 0.5 * abs(math.sin(t * 9)))
        elif t < t_ask:
            kw = dict(ind="working")
        elif t < t_sent:
            kw = dict(lines=[("You", phrase.replace("4 PM", "4 PM")),
                             ("Claude", f"Ready to send Alex: “{msg}” Send it?")])
        else:
            kw = dict(lines=[("Claude", "Sent to Alex.")])
        put_island(img, F.frame(I, t, **kw))
        nod = 9 * math.sin((t - t_nod) * 12) if t_nod <= t < t_nod + 0.55 else 0
        mouth_tile(img, keys, t, 60, 130, nod=nod,
                   label=(0.7 <= t < te, "Mouthing, no sound" if 0.7 <= t < te else ("Nodding = yes" if t_nod <= t < t_sent else "Silent Chat sees your lips")))
        cap = ("Mouth it. No sound, no keyboard." if t < 0.7 else
               "“open Discord and message Alex that I'll be available to meet at 4 PM”" if t < te else
               "It opens the chat and writes it…" if t < t_ask else
               "…and asks before sending. Nod for yes." if t < t_sent else "Sent, hands-free.")
        caption(img, cap, 16 if t < te and t >= 0.7 else 19)
        return img
    film("discord", t_sent + 2.2, W, H, draw, F)


def gif_split(F):
    """Mouth 'open my calendar and to-do list side by side': both open, tiled left and right."""
    W, H = 960, 540
    phrase = "open my calendar and to-do list side by side"
    keys, te = mouth_track(phrase, 0.7)
    t_go, t_done = te + 0.4, te + 1.6
    base = wallpaper(W, H)

    def calendar(d, b, lay):
        x0, y0, x1, y1 = b
        d.text((x0 + 16, y0 + 10), "Today", font=font(15, "Bold"), fill=(40, 40, 50))
        hours = ["9 AM", "10", "11", "12 PM", "1", "2", "3", "4", "5", "6"]
        top, step = y0 + 40, (y1 - y0 - 52) / len(hours)
        for k, h in enumerate(hours):
            yy = top + k * step
            d.text((x0 + 12, yy - 7), h, font=font(10), fill=(140, 140, 150))
            d.line([x0 + 54, yy, x1 - 12, yy], fill=(232, 232, 238))
        for (s, e, title, c) in ((0.5, 1.0, "Team standup", (90, 140, 255)), (5.0, 6.0, "Design review", (175, 110, 240)),
                                 (7.0, 7.5, "Meet Alex", (255, 150, 70)), (9.0, 9.9, "Gym", (60, 190, 120))):
            ya, yb = top + s * step, top + e * step
            soft = tuple(int(v * 0.22 + 255 * 0.78) for v in c) + (255,)  # (solid: the layer doesn't blend)
            d.rounded_rectangle([x0 + 60, ya + 1, x1 - 16, yb - 1], radius=5, fill=soft, outline=c + (255,))
            d.rectangle([x0 + 60, ya + 1, x0 + 63, yb - 1], fill=c + (255,))
            d.text((x0 + 70, ya + 2), title, font=font(11, "Semibold"), fill=tuple(int(v * 0.55) for v in c))

    def tasks(d, b, lay):
        x0, y0, x1, y1 = b
        d.text((x0 + 16, y0 + 10), "Tasks", font=font(15, "Bold"), fill=(40, 40, 50))
        for k, (task, done) in enumerate((("Send the invoice", False), ("Reply to Alex about 4 PM", True),
                                          ("Book a dentist appointment", False), ("Pick up groceries", False),
                                          ("Plan the weekend trip", False), ("Water the plants", True))):
            yy = y0 + 46 + k * 34
            d.ellipse([x0 + 16, yy, x0 + 32, yy + 16], outline=(170, 120, 240), width=2,
                      fill=(170, 120, 240) if done else None)
            d.text((x0 + 42, yy), task, font=font(13), fill=(150, 150, 160) if done else (45, 45, 55))
            if done:
                tw = d.textlength(task, font=font(13))
                d.line([x0 + 42, yy + 9, x0 + 42 + tw, yy + 9], fill=(150, 150, 160), width=1)

    def browser(d, b, lay):
        x0, y0, x1, y1 = b
        d.rounded_rectangle([x0 + 16, y0 + 12, x1 - 16, y0 + 36], radius=8, fill=(236, 236, 240))
        d.text((x0 + 28, y0 + 17), "news.example.com", font=font(12), fill=(110, 110, 120))
        for k in range(6):
            d.rounded_rectangle([x0 + 24, y0 + 56 + k * 36, x1 - 24 - (k % 3) * 60, y0 + 70 + k * 36], radius=4,
                                fill=(226, 226, 232))

    def draw(t, I):
        img = base.copy()
        a = ease((t - t_go) / 0.6)
        menubar(img, "Calendar" if t >= t_go else "Safari")
        if a < 1:
            window(img, (330, 60, 910, 430), "News", alpha=1 - a, body=browser)
        if t >= t_go:
            Lx0, Lx1 = 280, 612
            Rx0, Rx1 = 620, 950
            mid = (Lx0 + Rx1) / 2
            window(img, (mid - (mid - Lx0) * a - 60 * (1 - a), 60, mid - (mid - Lx1) * a + 60 * (1 - a), 470),
                   "Calendar", alpha=a, body=calendar)
            window(img, (mid - (mid - Rx0) * a - 40 * (1 - a), 70, mid - (mid - Rx1) * a + 80 * (1 - a), 470),
                   "Tasks", alpha=a, body=tasks)
        if t < 0.7:
            kw = dict(visible=False)
        elif t < te:
            kw = dict(ind="mouth", amp=0.4 + 0.5 * abs(math.sin(t * 9)))
        elif t < t_done:
            kw = dict(ind="working")
        else:
            kw = dict(lines=[("You", phrase), ("Claude", "Calendar on the left, your tasks on the right.")])
        put_island(img, F.frame(I, t, **kw))
        mouth_tile(img, keys, t, 50, 150, label=(0.7 <= t < te, "Mouthing, no sound" if 0.7 <= t < te
                                                 else "Silent Chat sees your lips"))
        cap = ("Mouth it…" if t < 0.7 else "“open my calendar and to-do list side by side”" if t < te else
               "It arranges your windows…" if t < t_done else "Side by side, in a second.")
        caption(img, cap)
        return img
    film("split", t_done + 2.4, W, H, draw, F)


def cover_file(key):
    p = Path(tempfile.gettempdir()) / f"silentchat_{key}.jpg"
    if not p.exists():
        subprocess.run(["curl", "-s", "-o", str(p), COVERS[key]], check=True)
    return "file://" + str(p)


def scenery_video(img, box):
    """A made-up video playing (sunset over hills) in a dark player window."""
    x0, y0, x1, y1 = [int(v) for v in box]
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(box, radius=14, fill=(18, 18, 24, 255))
    pw, ph = x1 - x0 - 20, y1 - y0 - 60
    yy = np.linspace(0, 1, ph)[:, None]
    pic = np.zeros((ph, pw, 3), np.uint8)
    pic[..., 0] = (250 - 60 * yy).astype(np.uint8)
    pic[..., 1] = (170 - 40 * yy).astype(np.uint8)
    pic[..., 2] = (120 + 40 * yy).astype(np.uint8)
    pim = Image.fromarray(pic)
    pd = ImageDraw.Draw(pim)
    pd.ellipse([pw * .62, ph * .22, pw * .78, ph * .22 + pw * .16], fill=(255, 236, 180))
    pd.polygon([(0, ph), (0, ph * .7), (pw * .3, ph * .5), (pw * .6, ph * .72), (pw, ph * .55), (pw, ph)], fill=(70, 60, 110))
    pd.polygon([(0, ph), (0, ph * .85), (pw * .4, ph * .68), (pw * .8, ph * .9), (pw, ph * .8), (pw, ph)], fill=(45, 40, 80))
    m = Image.new("L", pim.size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, pw, ph], radius=8, fill=255)
    img.paste(pim, (x0 + 10, y0 + 10), m)
    d = ImageDraw.Draw(img)
    d.text((x0 + 18, y1 - 40), "▶  Cooking Japanese curry at home", font=font(14, "Semibold"), fill=(235, 235, 240))
    d.rounded_rectangle([x0 + 18, y1 - 16, x1 - 18, y1 - 12], radius=2, fill=(80, 80, 90))
    d.rounded_rectangle([x0 + 18, y1 - 16, x0 + 18 + (x1 - x0 - 36) * 0.38, y1 - 12], radius=2, fill=(235, 60, 60))


def gif_spotify(F):
    """While a video plays: 'play Drake on Spotify' -> God's Plan, without leaving the video."""
    W, H = 960, 440
    phrase = "play Drake on Spotify"
    keys, te = mouth_track(phrase, 0.8)
    t_done = te + 1.0
    art = cover_file("gods_plan")
    base = wallpaper(W, H)
    menubar(base, "YouTube", ("File", "Edit", "View"))
    scenery_video(base, (W * 0.47, 70, W * 0.95, 330))

    def draw(t, I):
        img = base.copy()
        if t < 0.8:
            kw = dict(visible=False)
        elif t < te:
            kw = dict(ind="mouth", amp=0.4 + 0.5 * abs(math.sin(t * 9)))
        elif t < t_done:
            kw = dict(ind="working")
        else:
            kw = dict(lines=[("You", phrase)],
                      playing={"title": "God's Plan", "artist": "Drake", "state": "playing", "art": art})
        put_island(img, F.frame(I, t, **kw))
        mouth_tile(img, keys, t, 60, 100, size=190,
                   label=(0.8 <= t < te, "Mouthing, no sound" if 0.8 <= t < te else "Silent Chat sees your lips"))
        caption(img, "Watching a video. Mouth to Silent Chat, no sound:" if t < 0.8 else
                "“play Drake on Spotify”" if t < te else "Checked on your Mac, then Claude finds it…" if t < t_done
                else "Playing, and you never left your video.")
        return img
    film("spotify", t_done + 3.4, W, H, draw, F)


def gif_misread(F):
    """Hands-free picks up you talking to someone else: struck through, nothing happens."""
    W, H = 960, 440
    phrase = "so I'm going to talk a little bit about it"
    keys, te = mouth_track(phrase, 0.6)
    t_strike = te + 0.5
    base = wallpaper(W, H)
    menubar(base, "FaceTime", ("Video", "Call", "View"))

    def call(d, b, lay):
        x0, y0, x1, y1 = b
        d.rectangle([x0, y0, x1, y1 - 12], fill=(30, 30, 36, 255))
        d.ellipse([(x0 + x1) / 2 - 50, y0 + 60, (x0 + x1) / 2 + 50, y0 + 160], fill=(120, 150, 210, 255))
        d.text(((x0 + x1) / 2 - 7, y0 + 92), "J", font=font(30, "Bold"), fill="white")
        d.text(((x0 + x1) / 2 - 30, y0 + 175), "Jordan", font=font(16, "Semibold"), fill=(235, 235, 240))

    def draw(t, I):
        img = base.copy()
        window(img, (420, 66, 900, 390), "FaceTime", body=call, dark=True)
        if t < 0.6:
            kw = dict(visible=False)
        elif t < te:
            kw = dict(ind="mouth", amp=0.4 + 0.5 * abs(math.sin(t * 9)))
        elif t < t_strike:
            kw = dict(ind="loading")
        elif t < t_strike + 1.6:
            kw = dict(lines=[("You", "So I'm going to talk a little bit about it", 1000.0 + t_strike)])
        else:
            kw = dict(visible=False)
        put_island(img, F.frame(I, t, **kw))
        mouth_tile(img, keys, t, 60, 100, size=190, label=(False, "Talking to Jordan"))
        caption(img, "Hands-free on, talking to someone else…" if t < t_strike else
                "Not meant for Silent Chat: crossed out, nothing happens." if t < t_strike + 2.0 else
                "A small model on your Mac makes that call. Free.")
        return img
    film("misread", t_strike + 4.0, W, H, draw, F)


def gif_privacy(F):
    """A bank window in front: the privacy check keeps it from Claude."""
    W, H = 960, 440
    phrase = "what's my balance"
    keys, te = mouth_track(phrase, 0.7)
    t_reply = te + 1.0
    base = wallpaper(W, H)
    menubar(base, "Safari", ("File", "Edit", "View"))

    def bank(d, b, lay):
        x0, y0, x1, y1 = b
        d.rounded_rectangle([x0 + 16, y0 + 10, x1 - 16, y0 + 32], radius=8, fill=(236, 236, 240))
        d.text((x0 + 28, y0 + 14), "🔒 northwind-bank.example/accounts", font=font(12), fill=(110, 110, 120))
        d.text((x0 + 22, y0 + 48), "Northwind Bank", font=font(18, "Bold"), fill=(20, 60, 120))
        for k, (n, v) in enumerate((("Everyday checking ••4821", "$3,204.11"), ("Savings ••9930", "$12,870.00"))):
            yy = y0 + 92 + k * 54
            d.rounded_rectangle([x0 + 20, yy, x1 - 20, yy + 44], radius=8, fill=(244, 246, 250), outline=(225, 230, 238))
            d.text((x0 + 34, yy + 13), n, font=font(13, "Medium"), fill=(50, 50, 60))
            d.text((x1 - 34 - d.textlength(v, font=font(14, "Bold")), yy + 12), v, font=font(14, "Bold"),
                   fill=(20, 60, 120))

    def draw(t, I):
        img = base.copy()
        window(img, (400, 62, 910, 380), "Northwind Bank", body=bank)
        if t < 0.7:
            kw = dict(visible=False)
        elif t < te:
            kw = dict(ind="mouth", amp=0.4 + 0.5 * abs(math.sin(t * 9)))
        elif t < t_reply:
            kw = dict(ind="working")
        else:
            kw = dict(lines=[("You", "what's my balance?"),
                             ("Claude", "That window looks private (financial), so it wasn't shared with me. Tell me "
                                        "what you need from it.")])
        put_island(img, F.frame(I, t, **kw))
        mouth_tile(img, keys, t, 60, 100, size=190,
                   label=(0.7 <= t < te, "Mouthing, no sound" if 0.7 <= t < te else "Silent Chat sees your lips"))
        caption(img, "Banking site in front…" if t < 0.7 else "“what's my balance?”" if t < te else
                "A check on your Mac looks first…" if t < t_reply else "Private windows never reach Claude.")
        return img
    film("privacy", t_reply + 3.4, W, H, draw, F)


GIFS = {"discord": gif_discord, "split": gif_split, "spotify": gif_spotify, "misread": gif_misread,
        "privacy": gif_privacy}

if __name__ == "__main__":
    F = Film()
    for name in (sys.argv[1:] or GIFS):
        GIFS[name](F)
