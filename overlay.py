"""Floating Liquid Glass overlay for silent chat.

Native macOS 26 Liquid Glass (NSGlassEffectView; frosted NSVisualEffectView on older systems) in a borderless panel
that floats above every app, including full-screen ones, never takes keyboard focus, can be dragged anywhere and
snaps to the nearest screen edge. It remembers its position.

  status bar    state (LIVE / LISTENING / ...) and three small badges, one per face input:
                  eye    open / half / closed (a long blink starts listening; a ring fills while you hold it)
                  mouth  closed / talking / smile (in TYPE mode a smile deletes words)
                  brow   down / up (in TYPE mode a raise presses Enter; a ring fills while you hold it)
  buttons       TYPE, VOICE, FACE, MIC, x (hide), each its own glass capsule
  transcript    the conversation

Glyphs and text are drawn as light on black and turned into premultiplied alpha (black = clear), so the glass shows
through and the glow sits on top of it.
"""
import json
import textwrap
from pathlib import Path

import cv2
import numpy as np
import objc
from AppKit import (NSApplication, NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered, NSColor, NSData,
                    NSDate, NSDefaultRunLoopMode, NSEvent, NSEventMaskAny, NSImage, NSMakeRect, NSPanel, NSScreen,
                    NSStatusWindowLevel, NSView, NSWindowCollectionBehaviorCanJoinAllSpaces,
                    NSWindowCollectionBehaviorFullScreenAuxiliary, NSWindowCollectionBehaviorIgnoresCycle,
                    NSWindowCollectionBehaviorStationary, NSWindowStyleMaskBorderless,
                    NSWindowStyleMaskNonactivatingPanel)

# layout, in points (y down)
W = 300
STATUS_Y, STATUS_H = 0, 44
BADGE_R, BADGE_GAP = 12, 7  # face-input badges on the status bar
PILLS_Y, PILL_H = 54, 36
PILLS = [("type", "TYPE", 58), ("voice", "VOICE", 64), ("face", "FACE", 58), ("mic", "MIC", 50), ("close", "x", 34)]
PILL_GAP = (W - sum(w for _, _, w in PILLS)) / (len(PILLS) - 1)
TEXT_Y, TEXT_H = 100, 150
H = TEXT_Y + TEXT_H

CYAN, ICE, WHITE = (255, 226, 150), (255, 240, 205), (255, 255, 255)  # BGR
RED, AMBER, GREEN, VIOLET = (110, 110, 255), (90, 200, 255), (150, 235, 140), (230, 150, 255)
STATE = {"idle": ("LIVE", CYAN), "capturing": ("LISTENING", RED), "waiting": ("BLINK SEEN", CYAN),
         "thinking": ("THINKING", AMBER), "speaking": ("SPEAKING", GREEN), "noface": ("NO FACE", (90, 160, 255))}
PILL_COLOR = {"type": AMBER, "voice": GREEN, "face": CYAN, "mic": VIOLET}


# ----------------------------------------------------------------------------- drawing helpers (light on black)

def _dim(color, f):
    return tuple(int(c * f) for c in color)


def _thick(s):
    return max(1, s // 2 + 1)


def _text(img, t, x, y, size, color, s):
    cv2.putText(img, t, (int(x * s), int(y * s)), cv2.FONT_HERSHEY_SIMPLEX, size * s, color, _thick(s), cv2.LINE_AA)


def _text_w(t, size, s):
    return cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, size * s, _thick(s))[0][0] / s


def _glow(img, sigma, gain=0.9):
    """Add a soft glow to everything drawn in img (fast: blur a downscaled copy)."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, sigma / 4))
    img[:] = cv2.add(img, cv2.convertScaleAbs(cv2.resize(small, (w, h)), alpha=gain))


def _curve(cx, cy, half_w, bulge, n=33):
    """Points of a smooth arc: y = cy - bulge * (1 - x^2) over x in [-1, 1] (positive bulge = up)."""
    x = np.linspace(-1, 1, n)
    return np.stack([cx + x * half_w, cy - bulge * (1 - x ** 2)], 1)


def _pts(a, s):
    return np.round(np.asarray(a) * s).astype(np.int32)


# ----------------------------------------------------------------------------- badges (coordinates in points)

def _line(img, pts, color, s, width=1.5, closed=False):
    cv2.polylines(img, [_pts(pts, s)], closed, color, max(1, int(round(width * s))), cv2.LINE_AA)


def badge(img, cx, cy, s, kind, d):
    """A small flat badge with a simple glyph, Discord-style. d: that input's live state (None = no face)."""
    c, r = (int(cx * s), int(cy * s)), int(BADGE_R * s)
    tint, glyph, progress, ring = None, _dim(WHITE, 0.55), 0.0, CYAN
    if d is not None:
        if kind == "eyes":
            progress = d["progress"]
            tint = RED if progress >= 1 else CYAN if d["state"] != "OPEN" or progress > 0 else None
        elif kind == "mouth":
            tint = {"TALKING": RED, "DELETING": AMBER, "SMILE": AMBER, "OPEN": CYAN}.get(d["state"])
        else:
            progress, ring = d["progress"], GREEN
            tint = GREEN if d["state"] == "UP" else None
    if tint is not None:
        cv2.circle(img, c, r, _dim(tint, 0.30), -1, cv2.LINE_AA)
        glyph = WHITE
    else:
        cv2.circle(img, c, r, _dim(WHITE, 0.07), -1, cv2.LINE_AA)
    cv2.circle(img, c, r, _dim(tint or WHITE, 0.55 if tint else 0.22), max(1, s // 2), cv2.LINE_AA)
    if 0 < progress < 1:  # hold progress, as a thin ring just outside the badge
        cv2.ellipse(img, c, (r + int(3 * s), r + int(3 * s)), -90, 0, 360 * progress, ring, max(1, int(1.5 * s)),
                    cv2.LINE_AA)
    if d is None:
        _line(img, [(cx - 4, cy), (cx + 4, cy)], _dim(WHITE, 0.3), s)
        return
    if kind == "eyes":
        o = float(np.clip(1 - d["closure"], 0, 1))
        if o > 0.25:
            upper, lower = _curve(cx, cy, 6.5, 4 * o), _curve(cx, cy, 6.5, -4 * o)
            _line(img, np.vstack([upper, lower[::-1]]), glyph, s, closed=True)
            cv2.circle(img, c, max(1, int(1.8 * s * o)), glyph, -1, cv2.LINE_AA)
        else:  # closed lid with lashes
            _line(img, _curve(cx, cy - 1, 6.5, -2.5), glyph, s)
            for x in (-0.55, 0, 0.55):
                px, py = cx + x * 6.5, cy - 1 + 2.5 * (1 - x ** 2)
                _line(img, [(px, py), (px + 1.5 * x, py + 2.5)], glyph, s, width=1.2)
    elif kind == "mouth":
        state = d["state"]
        if state in ("SMILE", "DELETING"):
            _line(img, _curve(cx, cy - 1.5, 6, -4), glyph, s, width=1.7)
        elif state in ("TALKING", "OPEN"):
            h = 1.5 + 3.5 * float(np.clip(d["openness"], 0, 1))
            cv2.ellipse(img, c, (int(5.5 * s), int(h * s)), 0, 0, 360, glyph, max(1, int(1.5 * s)), cv2.LINE_AA)
        else:
            _line(img, _curve(cx, cy, 6, -0.8), glyph, s, width=1.7)
    else:  # brow: a thick, slightly slanted brow that rises with yours, over a small eye
        y = cy - 2 - 3 * float(np.clip(d["lift"], 0, 1))
        _line(img, [(cx - 6.5, y + 1.5), (cx - 2, y - 1), (cx + 6.5, y + 0.5)], glyph, s, width=2.2)
        eye_y = cy + 4
        _line(img, np.vstack([_curve(cx, eye_y, 4.5, 1.6), _curve(cx, eye_y, 4.5, -1.6)[::-1]]), _dim(glyph, 0.8), s,
              width=1.1, closed=True)


# ----------------------------------------------------------------------------- card renderers (BGR, light on black)

def render_status(s, state, status, typing, inputs=None):
    canvas = np.zeros((STATUS_H * s, W * s, 3), np.uint8)
    label, accent = STATE.get(state, STATE["idle"])
    dot = (int(22 * s), int(STATUS_H / 2 * s))
    cv2.circle(canvas, dot, int(4 * s), accent, -1, cv2.LINE_AA)
    _glow(canvas, 6 * s)
    cv2.circle(canvas, dot, int(3 * s), WHITE if state != "idle" else accent, -1, cv2.LINE_AA)
    _text(canvas, label, 34, STATUS_H / 2 + 5, 0.42, _dim(WHITE, 0.95), s)
    # a short detail after the label (e.g. "1.4 S" while listening); longer hints live in the transcript card
    rest = status[len(label):].lstrip(" .:-") if status.upper().startswith(label) else ""
    if typing and state == "idle":
        rest = "TYPE MODE"
    if rest and len(rest) <= 16:
        _text(canvas, rest.upper(), 34 + _text_w(label, 0.42, s) + 8, STATUS_H / 2 + 5, 0.33, _dim(WHITE, 0.55), s)
    # face-input badges, right-aligned: eye, mouth, brow
    for i, kind in enumerate(("brows", "mouth", "eyes")):
        cx = W - 14 - BADGE_R - i * (2 * BADGE_R + BADGE_GAP)
        badge(canvas, cx, STATUS_H / 2, s, kind, (inputs or {}).get(kind))
    return canvas


def render_pill_labels(s, typing, voice, face, gesture, mic):
    """Labels and status dots for the glass capsule buttons. Returns canvas and button rects (points, row-relative)."""
    canvas = np.zeros((PILL_H * s, W * s, 3), np.uint8)
    state = {"type": typing, "voice": voice, "face": face, "mic": mic, "close": None}
    buttons, x = {}, 0.0
    for name, text, bw in PILLS:
        buttons[name] = (x, 0, x + bw, PILL_H)
        on = state[name]
        tw = _text_w(text, 0.36, s)
        if on is None:
            _text(canvas, text, x + (bw - tw) / 2, PILL_H / 2 + 5, 0.42, _dim(WHITE, 0.7), s)
        else:
            dot = (int((x + 13) * s), int(PILL_H / 2 * s))
            lit = on or (name == "face" and face and gesture)
            cv2.circle(canvas, dot, int(3 * s), PILL_COLOR[name] if on else _dim(WHITE, 0.25), -1, cv2.LINE_AA)
            _text(canvas, text, x + 19 + (bw - 25 - tw) / 2, PILL_H / 2 + 5, 0.36, _dim(WHITE, 0.95 if lit else 0.5), s)
        x += bw + PILL_GAP
    return canvas, buttons


def render_text(s, conversation, notice):
    """Transcript: newest at the bottom, older lines fade."""
    canvas = np.zeros((TEXT_H * s, W * s, 3), np.uint8)
    lines = []
    for who, t in conversation[-8:]:
        col = {"You": (255, 230, 170), "Typed": (140, 200, 255)}.get(who, (245, 245, 245))
        lines += [(part, col) for part in textwrap.wrap(f"{who}: {t}", 46)]
    if notice:
        lines += [(part, (140, 175, 255)) for part in textwrap.wrap(notice, 46)]
    if not lines:
        lines = [("Hold Option or blink long, then mouth.", _dim(WHITE, 0.45))]
    y = TEXT_H - 16
    for i, (part, col) in enumerate(reversed(lines)):
        if y < 16:
            break
        _text(canvas, part, 16, y, 0.37, _dim(col, max(0.35, 1 - 0.09 * i)), s)
        y -= 15
    return canvas


def to_rgba(canvas):
    """Light-on-black BGR -> premultiplied RGBA: alpha = brightness, so black is clear and the glass shows through."""
    rgba = np.empty(canvas.shape[:2] + (4,), np.uint8)
    rgba[..., :3] = canvas[..., ::-1]
    rgba[..., 3] = canvas.max(axis=2)
    return rgba


def render(s, inputs, state, status, voice, face, gesture, conversation, notice, typing=False, mic=False):
    """The whole overlay as one premultiplied RGBA image (for previews/tests) + button rects in points."""
    canvas = np.zeros((H * s, W * s, 3), np.uint8)
    canvas[STATUS_Y * s:(STATUS_Y + STATUS_H) * s] = render_status(s, state, status, typing, inputs)
    labels, buttons = render_pill_labels(s, typing, voice, face, gesture, mic)
    canvas[PILLS_Y * s:(PILLS_Y + PILL_H) * s] = labels
    canvas[TEXT_Y * s:] = render_text(s, conversation, notice)
    return to_rgba(canvas), {k: (a, b + PILLS_Y, c, d + PILLS_Y) for k, (a, b, c, d) in buttons.items()}


# ----------------------------------------------------------------------------- the macOS panel

class OverlayView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(OverlayView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.ov_image, self.ov_owner, self.ov_drag, self.ov_moved, self.ov_name = None, None, None, False, ""
        return self

    def isFlipped(self):
        return True  # top-left origin, like the canvas

    def acceptsFirstMouse_(self, event):
        return True

    def mouseDownCanMoveWindow(self):
        return False

    def drawRect_(self, rect):
        if self.ov_image is not None:
            self.ov_image.drawInRect_(self.bounds())

    def mouseDown_(self, event):
        self.ov_drag = (NSEvent.mouseLocation(), self.window().frame().origin)
        self.ov_moved = False

    def mouseDragged_(self, event):
        if self.ov_drag is None:
            return
        start, origin = self.ov_drag
        now = NSEvent.mouseLocation()
        dx, dy = now.x - start.x, now.y - start.y
        if abs(dx) + abs(dy) > 3:
            self.ov_moved = True
        self.window().setFrameOrigin_((origin.x + dx, origin.y + dy))

    def mouseUp_(self, event):
        if self.ov_drag is not None and self.ov_owner is not None:
            if self.ov_moved:
                self.ov_owner.snap()
            else:
                p = self.convertPoint_fromView_(event.locationInWindow(), None)
                self.ov_owner.click(self.ov_name, p.x, p.y)
        self.ov_drag = None


def _nsimage(rgba, w_pt, h_pt):
    import Quartz
    h, w = rgba.shape[:2]
    provider = Quartz.CGDataProviderCreateWithCFData(NSData.dataWithBytes_length_(rgba.tobytes(), rgba.nbytes))
    image = Quartz.CGImageCreate(w, h, 8, 32, w * 4, Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceSRGB),
                                 Quartz.kCGImageAlphaPremultipliedLast, provider, None, False,
                                 Quartz.kCGRenderingIntentDefault)
    return NSImage.alloc().initWithCGImage_size_(image, (w_pt, h_pt))


def _glass(frame, radius, tint):
    """A Liquid Glass shape (macOS 26), or a frosted fallback on older systems."""
    import AppKit
    if hasattr(AppKit, "NSGlassEffectView"):
        view = AppKit.NSGlassEffectView.alloc().initWithFrame_(frame)
        view.setCornerRadius_(radius)
        view.setStyle_(AppKit.NSGlassEffectViewStyleRegular)
        view.setTintColor_(tint)
        return view
    view = AppKit.NSVisualEffectView.alloc().initWithFrame_(frame)
    view.setMaterial_(AppKit.NSVisualEffectMaterialHUDWindow)
    view.setBlendingMode_(AppKit.NSVisualEffectBlendingModeBehindWindow)
    view.setState_(AppKit.NSVisualEffectStateActive)
    view.setWantsLayer_(True)
    view.layer().setCornerRadius_(radius)
    view.layer().setMasksToBounds_(True)
    return view


def _rgba(r, g, b, a):
    return NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, a)


TINT_CARD = _rgba(0.03, 0.04, 0.08, 0.30)
TINT_OFF = _rgba(0.03, 0.04, 0.07, 0.25)
TINT_ON = {"type": _rgba(1.0, 0.75, 0.35, 0.22), "voice": _rgba(0.55, 0.92, 0.6, 0.22),
           "face": _rgba(0.6, 0.88, 1.0, 0.22), "mic": _rgba(1.0, 0.6, 0.9, 0.22)}


class Overlay:
    def __init__(self, on_button, state_file):
        self.on_button, self.state_file, self.buttons, self.cache = on_button, Path(state_file), {}, {}
        self.app = NSApplication.sharedApplication()
        self.app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # no Dock icon; floats over full screen
        self.app.finishLaunching()
        screen = NSScreen.mainScreen().visibleFrame()
        saved = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        side = saved.get("side", "right")
        x = screen.origin.x + 12 if side == "left" else screen.origin.x + screen.size.width - W - 12
        y = saved.get("y", screen.origin.y + screen.size.height - H - 12)
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, W, H), NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered, False)
        self.panel.setFloatingPanel_(True)  # must come before setLevel_, which it would otherwise reset to "floating"
        self.panel.setLevel_(NSStatusWindowLevel)  # above full-screen apps, like Zoom's floating controls
        self.panel.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces
                                          | NSWindowCollectionBehaviorFullScreenAuxiliary
                                          | NSWindowCollectionBehaviorStationary
                                          | NSWindowCollectionBehaviorIgnoresCycle)
        self.panel.setHidesOnDeactivate_(False)
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(NSColor.clearColor())
        self.panel.setHasShadow_(False)
        root = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))

        def frame(x0, y0, w, h):  # y-down layout -> AppKit's y-up coordinates
            return NSMakeRect(x0, H - y0 - h, w, h)

        root.addSubview_(_glass(frame(0, STATUS_Y, W, STATUS_H), STATUS_H / 2, TINT_CARD))
        self.pill_glass, x = {}, 0.0
        for name, _, bw in PILLS:
            self.pill_glass[name] = _glass(frame(x, PILLS_Y, bw, PILL_H), PILL_H / 2, TINT_OFF)
            root.addSubview_(self.pill_glass[name])
            x += bw + PILL_GAP
        root.addSubview_(_glass(frame(0, TEXT_Y, W, TEXT_H), 22, TINT_CARD))
        self.views = {}
        for name, y0, h in (("status", STATUS_Y, STATUS_H), ("pills", PILLS_Y, PILL_H), ("text", TEXT_Y, TEXT_H)):
            view = OverlayView.alloc().initWithFrame_(frame(0, y0, W, h))  # glyphs on top of the glass
            view.ov_owner, view.ov_name = self, name
            root.addSubview_(view)
            self.views[name] = view
        self.panel.setContentView_(root)
        self.panel.orderFrontRegardless()
        self.scale = max(1, int(round(self.panel.backingScaleFactor())))
        self.snap(animate=False)

    def pump(self):
        """Process pending UI events (clicks, drags). Call every frame."""
        while True:
            ev = self.app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                NSEventMaskAny, NSDate.distantPast(), NSDefaultRunLoopMode, True)
            if ev is None:
                break
            self.app.sendEvent_(ev)
        self.app.updateWindows()

    def _put(self, name, canvas, height):
        self.views[name].ov_image = _nsimage(to_rgba(canvas), W, height)
        self.views[name].setNeedsDisplay_(True)

    def show(self, inputs, state, status, voice, face, gesture, conversation, notice, typing=False, mic=False):
        s = self.scale
        self._put("status", render_status(s, state, status, typing, inputs), STATUS_H)  # live badges: every frame
        for name, key, draw, h in (
                ("text", (tuple(conversation[-8:]), notice), lambda: render_text(s, conversation, notice), TEXT_H),):
            if self.cache.get(name) != key:  # the rest only redraws when it changes
                self._put(name, draw(), h)
                self.cache[name] = key
        key = (typing, voice, face, gesture, mic)
        if self.cache.get("pills") != key:
            canvas, self.buttons = render_pill_labels(s, *key)
            self._put("pills", canvas, PILL_H)
            for name, on in (("type", typing), ("voice", voice), ("face", face), ("mic", mic)):
                self.pill_glass[name].setTintColor_(TINT_ON[name] if on else TINT_OFF)
            self.cache["pills"] = key

    def click(self, view, x, y):
        if view != "pills":
            return
        for name, (x0, y0, x1, y1) in self.buttons.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                self.on_button(name)

    def snap(self, animate=True):
        """Stick to the nearest left/right screen edge, inside the visible area."""
        f = self.panel.frame()
        screen = (self.panel.screen() or NSScreen.mainScreen()).visibleFrame()
        left = screen.origin.x + 12
        right = screen.origin.x + screen.size.width - f.size.width - 12
        side = "left" if f.origin.x + f.size.width / 2 < screen.origin.x + screen.size.width / 2 else "right"
        top = screen.origin.y + screen.size.height - f.size.height - 12
        y = min(max(f.origin.y, screen.origin.y + 12), top)
        self.panel.setFrame_display_animate_(
            NSMakeRect(left if side == "left" else right, y, f.size.width, f.size.height), True, animate)
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps({"side": side, "y": y}))
        except OSError:
            pass

    def hide(self):
        self.panel.orderOut_(None)

    def unhide(self):
        self.panel.orderFrontRegardless()

    def close(self):
        self.panel.orderOut_(None)
        self.panel.close()
