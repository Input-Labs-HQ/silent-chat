"""Tap Right Command to click what you're looking at.

A tap is Right Command pressed and released on its own, quickly, with no other key or click in between, so
Cmd shortcuts (with either Command key) work exactly as before. The gaze point snaps to the nearest clickable
thing (button, link, tab, field ...) within how far off gaze typically is; a ring flashes on it, then it's clicked.
Risky buttons (Send, Delete, Pay, Buy ...) need a second tap, or a steady look at them, before they're clicked.
"""
import math
import os
import re
import threading
import time

RIGHT_CMD = 54  # (key code)
SKIP_PIDS = {os.getpid()}
TAP_MAX = 0.35  # s: held longer than this, it's not a tap
RISKY = re.compile(r"\b(send|delete|remove|erase|discard|trash|pay|buy|purchase|order|checkout|check out|transfer|"
                   r"submit|confirm|unsubscribe|leave|log ?out|sign ?out|uninstall|post|publish|reply all)\b", re.I)
CLICKABLE = {"AXButton", "AXLink", "AXMenuItem", "AXMenuBarItem", "AXMenuButton", "AXPopUpButton", "AXCheckBox",
             "AXRadioButton", "AXTab", "AXTextField", "AXTextArea", "AXSearchField", "AXComboBox", "AXCell", "AXRow",
             "AXDisclosureTriangle", "AXSlider", "AXIncrementor", "AXColorWell", "AXImage", "AXStaticText"}
WEAK = {"AXImage", "AXStaticText", "AXRow", "AXCell"}  # (clickable only sometimes: used if nothing better is near)


class TapDetector:
    """Fed from the keyboard event tap (any thread); `taps` collects tap times for the main loop."""

    def __init__(self):
        self.down_at, self.spoiled, self.taps = None, False, []

    def flags(self, keycode, cmd_down, t):
        if keycode == RIGHT_CMD:
            if cmd_down and self.down_at is None:
                self.down_at, self.spoiled = t, False
            elif not cmd_down and self.down_at is not None:
                if not self.spoiled and t - self.down_at <= TAP_MAX:
                    self.taps.append(t)
                self.down_at = None
        elif self.down_at is not None:
            self.spoiled = True  # (another modifier joined in: a shortcut, not a tap)

    def other(self):
        """Any key or mouse button while Right Command is down: a shortcut or a Cmd-click."""
        if self.down_at is not None:
            self.spoiled = True

    def take(self):
        out, self.taps = self.taps, []
        return out


def rect_distance(px, py, rect):
    x, y, w, h = rect
    dx = max(x - px, 0.0, px - (x + w))
    dy = max(y - py, 0.0, py - (y + h))
    return math.hypot(dx, dy)


def choose(px, py, radius, candidates):
    """candidates: [{"role", "label", "rect"}]. The nearest clickable one within `radius` (inside counts as 0;
    small targets win ties), weak roles only if nothing proper is near. -> candidate or None."""
    best = None
    for c in candidates:
        if c["role"] not in CLICKABLE or c["rect"][2] < 2 or c["rect"][3] < 2:
            continue
        if c["rect"][2] * c["rect"][3] > 1200 * 700:  # (a whole pane: not a target)
            continue
        d = rect_distance(px, py, c["rect"])
        if d > radius:
            continue
        score = d + (radius if c["role"] in WEAK else 0.0) + 0.002 * math.sqrt(c["rect"][2] * c["rect"][3])
        if best is None or score < best[0]:
            best = (score, c)
    return best[1] if best else None


def risky(c):
    return bool(c and RISKY.search(c.get("label") or ""))


def centre(rect):
    return rect[0] + rect[2] / 2, rect[1] + rect[3] / 2


def probe(px, py, radius):
    """What's clickable around (px, py): Accessibility hit tests on a small spiral of points. -> (candidates, app,
    pid). Coordinates are screen points, y down from the top of the main screen."""
    import ApplicationServices as AX
    from AppKit import NSRunningApplication
    system = AX.AXUIElementCreateSystemWide()
    pts = [(px, py)]
    for frac, n in ((0.35, 6), (0.7, 8), (1.0, 10)):
        for k in range(n):
            a = 2 * math.pi * k / n
            pts.append((px + frac * radius * math.cos(a), py + frac * radius * math.sin(a)))
    seen, out, app, pid0 = set(), [], None, None
    t_end = time.monotonic() + 0.12
    for x, y in pts:
        if time.monotonic() > t_end:
            break
        err, el = AX.AXUIElementCopyElementAtPosition(system, x, y, None)
        if err != 0 or el is None:
            continue
        e, pid = AX.AXUIElementGetPid(el, None)
        if e != 0 or pid in SKIP_PIDS:  # (silent chat's own windows: the island, the highlight)
            continue
        if pid0 is None:
            pid0 = pid
            ra = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid0)
            app = ra.localizedName() if ra else None
        for _ in range(4):  # climb to the clickable thing it belongs to (text inside a button, etc.)
            role = _get(el, "AXRole")
            if role in CLICKABLE and role not in WEAK:
                break
            parent = _get(el, "AXParent")
            if parent is None or _get(parent, "AXRole") in ("AXWindow", "AXApplication", "AXWebArea", None):
                break
            el = parent
        rect = _rect(el)
        if rect is None or rect in seen:
            continue
        seen.add(rect)
        lab = next((v for v in (_get(el, a) for a in ("AXTitle", "AXDescription", "AXValue", "AXHelp"))
                    if isinstance(v, str) and v.strip()), "")
        out.append({"role": _get(el, "AXRole"), "label": lab.strip()[:60], "rect": rect})
    return out, app, pid0


def _get(el, attr):
    import ApplicationServices as AX
    try:
        err, v = AX.AXUIElementCopyAttributeValue(el, attr, None)
        return v if err == 0 else None
    except Exception:
        return None


def _rect(el):
    import ApplicationServices as AX
    pos, size = _get(el, "AXPosition"), _get(el, "AXSize")
    if pos is None or size is None:
        return None
    ok1, p = AX.AXValueGetValue(pos, AX.kAXValueCGPointType, None)
    ok2, s = AX.AXValueGetValue(size, AX.kAXValueCGSizeType, None)
    if not (ok1 and ok2):
        return None
    return (float(p.x), float(p.y), float(s.width), float(s.height))


def window_title(pid):
    import ApplicationServices as AX
    app = AX.AXUIElementCreateApplication(pid)
    win = _get(app, "AXFocusedWindow") or _get(app, "AXMainWindow")
    t = _get(win, "AXTitle") if win is not None else None
    return t if isinstance(t, str) else ""


_VIEW = []


def _view_class():
    """The ring's view (an NSView subclass, made once: PyObjC can't define the same class twice)."""
    if not _VIEW:
        from AppKit import NSBezierPath, NSView

        class GazeClickRingView(NSView):
            def drawRect_(self, r):
                self.colour.setStroke()
                b = self.bounds().size
                p = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(((2, 2), (b.width - 4, b.height - 4)), 7, 7)
                p.setLineWidth_(3)
                p.stroke()
        _VIEW.append(GazeClickRingView)
    return _VIEW[0]


class Ring:
    """A ring drawn around the target for a moment before the click (so a wrong snap is visible)."""

    def __init__(self):
        self.win = None

    def show(self, rect, risky_=False):
        from AppKit import NSBackingStoreBuffered, NSColor, NSScreen, NSStatusWindowLevel, NSWindow
        main_h = NSScreen.screens()[0].frame().size.height
        x, y, w, h = rect
        pad = 5
        frame = ((x - pad, main_h - (y + h) - pad), (w + 2 * pad, h + 2 * pad))
        if self.win is None:
            self.win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(frame, 0,
                                                                                     NSBackingStoreBuffered, False)
            self.win.setOpaque_(False)
            self.win.setBackgroundColor_(NSColor.clearColor())
            self.win.setIgnoresMouseEvents_(True)
            self.win.setLevel_(NSStatusWindowLevel + 3)
            self.win.setCollectionBehavior_(1 << 0 | 1 << 4)  # (all spaces, stationary)
            self.win.setHasShadow_(False)
        self.win.setFrame_display_(frame, True)
        view = _view_class().alloc().initWithFrame_(((0, 0), frame[1]))
        view.colour = NSColor.systemOrangeColor() if risky_ else NSColor.systemBlueColor()
        self.win.setContentView_(view)
        self.win.orderFrontRegardless()

    def hide(self):
        if self.win is not None:
            self.win.orderOut_(None)


# ----------------------------------------------------------------------------- the glass highlight

_GLASS = []


def _glass_class():
    """A faint liquid-glass sheen over a control: a whisper of fill, a bright rim fading down its edge, a specular
    streak along the top. No blur, so the control's text stays perfectly readable."""
    if not _GLASS:
        from AppKit import NSBezierPath, NSColor, NSGradient, NSView

        class GazeGlassView(NSView):
            def drawRect_(self, r):
                b = self.bounds()
                w, h = b.size.width, b.size.height
                rad = min(h / 2.0, 9.0)
                inset = ((1.5, 1.5), (w - 3, h - 3))
                body = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(inset, rad, rad)
                NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.07).setFill()
                body.fill()
                g = NSGradient.alloc().initWithColors_atLocations_colorSpace_(
                    [NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.16), NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.0)],
                    [0.0, 1.0], NSColor.whiteColor().colorSpace())
                top = ((3, h * 0.5), (w - 6, h * 0.5 - 2.5))
                g.drawInBezierPath_angle_(NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(top, rad * 0.8,
                                                                                                    rad * 0.8), 90)
                NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.42).setStroke()
                body.setLineWidth_(1.0)
                body.stroke()
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.35, 0.55, 1.0, 0.22).setStroke()
                outer = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(((0.5, 0.5), (w - 1, h - 1)),
                                                                                rad + 1, rad + 1)
                outer.setLineWidth_(1.0)
                outer.stroke()
        _GLASS.append(GazeGlassView)
    return _GLASS[0]


class Highlighter:
    """Keeps finding what Right Command would press where you're looking, and marks it with faint glass. Pressing
    Right Command then presses exactly that. A worker thread does the Accessibility lookups (a few a second, only
    when your gaze has moved); the main loop calls tick() each frame to show the result.

    It sticks to what it has (hysteresis): a new target only wins when your gaze has clearly left the old one,
    so it doesn't flicker between neighbours. Nothing shows when gaze isn't sure enough (snap radius > MAX_R),
    over private windows, or over silent chat itself."""

    MAX_R = 170.0
    EVERY = 0.2   # s between lookups at most
    MOVE = 18.0   # pt the gaze has to move before it looks again (or every 1.5 s)

    def __init__(self, blocked=None):
        self.blocked = blocked or (lambda app, pid: False)
        self.want, self.target, self.shown, self.win = None, None, None, None
        self.lock, self.wake = threading.Lock(), threading.Event()
        self.last_probe, self.last_at, self.lookups = None, 0.0, 0
        threading.Thread(target=self._worker, daemon=True).start()

    def update(self, gp, radius, active):
        """Each frame: the smoothed gaze point (or None), the snap radius, and whether to show anything."""
        if not active or gp is None or radius > self.MAX_R:
            with self.lock:
                self.want, self.target = None, None
            return
        with self.lock:
            self.want = (float(gp[0]), float(gp[1]), float(radius), time.monotonic())
        self.wake.set()

    def current(self, max_age=1.2):
        """The highlighted target, if it's fresh: what Right Command should press."""
        with self.lock:
            t = self.target
        return t if t and time.monotonic() - t["at"] < max_age else None

    def _worker(self):
        while True:
            self.wake.wait(1.0)
            self.wake.clear()
            with self.lock:
                want, cur = self.want, self.target
            now = time.monotonic()
            if want is None or now - want[3] > 0.5:  # (no fresh gaze: camera off, face lost)
                continue
            x, y, r, _ = want
            if cur is not None:  # still on it (with some slack): keep it, just refresh
                rx, ry, rw, rh = cur["rect"]
                m = 0.45 * r
                if rx - m <= x <= rx + rw + m and ry - m <= y <= ry + rh + m and now - cur["probed"] < 1.5:
                    with self.lock:
                        if self.target is cur:
                            cur["at"] = now
                    continue
            if (self.last_probe is not None and math.hypot(x - self.last_probe[0], y - self.last_probe[1]) < self.MOVE
                    and now - self.last_at < 1.5) or now - self.last_at < self.EVERY:
                continue
            self.last_probe, self.last_at = (x, y), now
            try:
                cands, app, pid = probe(x, y, r)
                self.lookups += 1
                c = choose(x, y, r, cands)
                if c is not None and app and pid and self.blocked(app, pid):
                    c = None
            except Exception:
                c = None
            with self.lock:
                if self.want is None:
                    self.target = None
                elif c is None:
                    self.target = None
                elif cur is not None and cur["rect"] == c["rect"]:
                    cur["at"] = cur["probed"] = now
                else:
                    self.target = {**c, "app": app or "?", "pid": pid, "at": now, "probed": now}

    def tick(self):
        """Main thread, each frame: show / move / hide the glass."""
        t = self.current()
        rect = t["rect"] if t else None
        if rect == self.shown:
            return
        self.shown = rect
        if rect is None:
            if self.win is not None:
                self.win.orderOut_(None)
            return
        from AppKit import (NSAnimationContext, NSBackingStoreBuffered, NSColor, NSScreen, NSStatusWindowLevel,
                            NSWindow)
        main_h = NSScreen.screens()[0].frame().size.height
        x, y, w, h = rect
        pad = 3
        frame = ((x - pad, main_h - (y + h) - pad), (w + 2 * pad, h + 2 * pad))
        if self.win is None:
            self.win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(frame, 0,
                                                                                     NSBackingStoreBuffered, False)
            self.win.setOpaque_(False)
            self.win.setBackgroundColor_(NSColor.clearColor())
            self.win.setIgnoresMouseEvents_(True)
            self.win.setLevel_(NSStatusWindowLevel + 2)
            self.win.setCollectionBehavior_(1 << 0 | 1 << 4 | 1 << 8)  # (all spaces, stationary, full-screen aux)
            self.win.setHasShadow_(False)
            view = _glass_class().alloc().initWithFrame_(((0, 0), frame[1]))
            view.setAutoresizingMask_(18)  # (width + height follow the window)
            self.win.setContentView_(view)
            self.win.setFrame_display_(frame, True)
            self.win.setAlphaValue_(0.0)
            self.win.orderFrontRegardless()
            NSAnimationContext.beginGrouping()
            NSAnimationContext.currentContext().setDuration_(0.18)
            self.win.animator().setAlphaValue_(1.0)
            NSAnimationContext.endGrouping()
            return
        visible = self.win.isVisible()
        if not visible:
            self.win.setFrame_display_(frame, True)
            self.win.setAlphaValue_(0.0)
            self.win.orderFrontRegardless()
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(0.14)
        if visible:
            self.win.animator().setFrame_display_(frame, True)  # (glides to the next control)
        self.win.animator().setAlphaValue_(1.0)
        NSAnimationContext.endGrouping()
        self.win.contentView().setNeedsDisplay_(True)

    def hide(self):
        with self.lock:
            self.want, self.target = None, None
        self.shown = None
        if self.win is not None:
            self.win.orderOut_(None)
