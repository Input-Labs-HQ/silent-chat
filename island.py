"""Silent chat as a Dynamic Island: a black capsule that grows out of the MacBook's notch.

  hidden      nothing but the notch
  compact     the notch widens a little: a live activity on the left (a little wave as you mouth or speak,
              dots while it reads your lips or Claude works, a green wave while it speaks), a small symbol on the right
  expanded    it drops down into a rounded card with the reply (sized to the text), and a text box when you pin it
  playing     album art, song, artist and ⏮ ⏯ ⏭ you can click, after you play / skip a song

Pure black, so it reads as part of the notch; the size springs smoothly between states. On a screen without a notch
it hangs from the top centre the same way. Drawn natively (SF fonts, SF Symbols), click-through except its controls.
"""
import math
import textwrap
import threading
import time

import objc
from AppKit import (NSAttributedString, NSBackingStoreBuffered, NSBezierPath, NSColor, NSFont, NSFontAttributeName,
                    NSForegroundColorAttributeName, NSImage, NSMakeRect, NSPanel, NSParagraphStyleAttributeName,
                    NSMutableParagraphStyle, NSScreen, NSStatusWindowLevel, NSTextField, NSView,
                    NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorFullScreenAuxiliary,
                    NSWindowCollectionBehaviorIgnoresCycle, NSWindowCollectionBehaviorStationary,
                    NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel)
from Foundation import NSURL, NSObject

WING = 40          # compact: how far it reaches past each side of the notch
EXP_W = 400        # expanded width
PAD = 18
TEXT_FONT = 13.0
FIELD_H = 30
ART = 52
SYMBOL = {"mouth": "mouth", "voice": "waveform", "listen": "ear", "loading": "eye", "working": "sparkles",
          "speaking": "speaker.wave.2.fill"}


def notch():
    """(screen, notch rect (x0, x1) in AppKit x, notch height). The built-in screen's notch, or a pretend one."""
    screens = NSScreen.screens()
    for s in screens:
        if s.safeAreaInsets().top > 0:
            l, r = s.auxiliaryTopLeftArea(), s.auxiliaryTopRightArea()
            return s, (l.origin.x + l.size.width, r.origin.x), s.safeAreaInsets().top
    s = NSScreen.mainScreen() or screens[0]
    f = s.frame()
    mid = f.origin.x + f.size.width / 2
    return s, (mid - 90, mid + 90), 24.0


def _white(a):
    return NSColor.colorWithCalibratedWhite_alpha_(1.0, a)


def _rgb(r, g, b, a=1.0):
    return NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, a)


COLORS = {"mouth": _white(0.95), "voice": _rgb(0.47, 0.78, 1.0), "listen": _white(0.9), "loading": _white(0.8),
          "working": _rgb(1.0, 0.75, 0.3), "speaking": _rgb(0.55, 0.92, 0.55)}


class IslandView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(IslandView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.owner = None
        return self

    def isFlipped(self):
        return True

    def acceptsFirstMouse_(self, event):
        return True

    def mouseDown_(self, event):
        p = self.convertPoint_fromView_(event.locationInWindow(), None)
        if self.owner is not None:
            self.owner.click(p.x, p.y)

    def drawRect_(self, rect):
        if self.owner is not None:
            self.owner.draw(self.bounds())


class _IslandFieldTarget(NSObject):
    def submit_(self, field):
        text = str(field.stringValue()).strip()
        field.setStringValue_("")
        if text and self.owner.on_text:
            self.owner.on_text(text)


def _symbol(name, size=13, color=None):
    img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, None)
    if img is None:
        return None
    try:
        from AppKit import NSImageSymbolConfiguration
        cfg = NSImageSymbolConfiguration.configurationWithPointSize_weight_(size, 0.3)
        if color is not None:
            cfg = cfg.configurationByApplyingConfiguration_(
                NSImageSymbolConfiguration.configurationWithHierarchicalColor_(color))
        img = img.imageWithSymbolConfiguration_(cfg) or img
    except Exception:
        pass
    return img


class Island:
    def __init__(self, on_text=None, on_control=None):
        from overlay import KeyPanel
        self.on_text, self.on_control = on_text, on_control
        self.screen, (self.nx0, self.nx1), self.nh = notch()
        self.w, self.h, self.vw, self.vh = self.nx1 - self.nx0, self.nh, 0.0, 0.0
        self.goal = (self.w, self.h)
        self.state = {"ind": None, "amp": 0.0, "lines": [], "playing": None, "field": False}
        self.art, self.art_url, self.controls, self.shown = None, None, {}, False
        self.panel = KeyPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 10, 10), NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered, False)
        self.panel.setFloatingPanel_(True)
        self.panel.setLevel_(NSStatusWindowLevel + 2)  # above the menu bar, like the notch itself
        self.panel.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces |
                                          NSWindowCollectionBehaviorFullScreenAuxiliary |
                                          NSWindowCollectionBehaviorStationary | NSWindowCollectionBehaviorIgnoresCycle)
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(NSColor.clearColor())
        self.panel.setHasShadow_(False)
        self.panel.setHidesOnDeactivate_(False)
        self.view = IslandView.alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10))
        self.view.owner = self
        self.panel.setContentView_(self.view)
        self.field = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 10, 10))
        self.field.setBezeled_(False)
        self.field.setDrawsBackground_(False)
        self.field.setFocusRingType_(1)
        self.field.setTextColor_(NSColor.whiteColor())
        self.field.setFont_(NSFont.systemFontOfSize_(TEXT_FONT))
        self.field.setPlaceholderAttributedString_(NSAttributedString.alloc().initWithString_attributes_(
            "Type to Claude, then Return", {NSForegroundColorAttributeName: _white(0.4),
                                            NSFontAttributeName: NSFont.systemFontOfSize_(TEXT_FONT)}))
        self.field_target = _IslandFieldTarget.alloc().init()
        self.field_target.owner = self
        self.field.setTarget_(self.field_target)
        self.field.setAction_("submit:")
        self.field.setHidden_(True)
        self.view.addSubview_(self.field)

    # ------------------------------------------------------------------ layout
    def _text_attrs(self, color):
        para = NSMutableParagraphStyle.alloc().init()
        para.setLineSpacing_(2)
        return {NSFontAttributeName: NSFont.systemFontOfSize_weight_(TEXT_FONT, 0.0),
                NSForegroundColorAttributeName: color, NSParagraphStyleAttributeName: para}

    def _text_height(self, lines, width):
        h = 0.0
        for line in lines:
            s = NSAttributedString.alloc().initWithString_attributes_(line[1], self._text_attrs(_white(1)))
            h += s.boundingRectWithSize_options_((width, 2000), 1 << 0).size.height + 6  # (uses line fragment origin)
        return h

    def update(self, ind=None, amp=0.0, lines=(), playing=None, field=False, visible=True):
        """Every frame. ind: the live activity (or None). lines: [(who, text)] to show (the last exchange).
        playing: {'title', 'artist', 'state', 'art'} for the now-playing card, or None. field: show the text box."""
        lines = [tuple(x) for x in lines if x[1]][-4:]  # (who, text) or (who, text, struck_at)
        self.state.update(ind=ind, amp=amp, lines=lines, playing=playing, field=field)
        nw = self.nx1 - self.nx0
        if not visible or (ind is None and not lines and not playing and not field):
            goal = (nw, self.nh)
        elif not lines and not playing and not field:
            goal = (nw + 2 * WING, self.nh)  # compact
        else:
            h = self.nh + 8
            if playing:
                h += ART + 14
            if lines:
                h += min(220.0, self._text_height(lines, EXP_W - 2 * PAD)) + 8
            if field:
                h += FIELD_H + 6
            goal = (EXP_W, h + 10)
        self.goal = goal
        if playing and playing.get("art") and playing["art"] != self.art_url:
            self._load_art(playing["art"])
        self._step()

    def _step(self):
        """A little spring towards the goal size, then place the window flush with the top of the screen."""
        k, d = 0.22, 0.62
        self.vw = self.vw * d + (self.goal[0] - self.w) * k
        self.vh = self.vh * d + (self.goal[1] - self.h) * k
        self.w, self.h = self.w + self.vw, self.h + self.vh
        settled = abs(self.goal[0] - self.w) < 0.5 and abs(self.goal[1] - self.h) < 0.5 and abs(self.vw) + abs(self.vh) < 0.5
        if settled:
            self.w, self.h, self.vw, self.vh = self.goal[0], self.goal[1], 0.0, 0.0
        nw = self.nx1 - self.nx0
        hidden = settled and self.w <= nw + 0.5 and self.h <= self.nh + 0.5
        if hidden:
            if self.shown:
                self.panel.orderOut_(None)
                self.shown = False
                self.field.setHidden_(True)
            return
        f = self.screen.frame()
        cx = (self.nx0 + self.nx1) / 2
        w, h = max(nw, self.w), max(self.nh, self.h)
        self.panel.setFrame_display_(NSMakeRect(cx - w / 2, f.origin.y + f.size.height - h, w, h), False)
        self.view.setFrame_(NSMakeRect(0, 0, w, h))
        show_field = self.state["field"] and settled
        if show_field:
            self.field.setFrame_(NSMakeRect(PAD, h - FIELD_H - 10, w - 2 * PAD, FIELD_H - 8))
        if show_field != (not self.field.isHidden()):
            self.field.setHidden_(not show_field)
        self.panel.ov_typing = show_field
        self.view.setNeedsDisplay_(True)
        if not self.shown:
            self.panel.orderFrontRegardless()
            self.shown = True

    def hit(self, x, y, margin=10):
        """Is the mouse (AppKit screen coordinates) over the island, or the bare notch when it's tucked away?"""
        if self.shown:
            f = self.panel.frame()
            return (f.origin.x - margin <= x <= f.origin.x + f.size.width + margin and
                    f.origin.y - margin <= y <= f.origin.y + f.size.height + 2)
        top = self.screen.frame().origin.y + self.screen.frame().size.height
        return self.nx0 - 8 <= x <= self.nx1 + 8 and top - self.nh <= y <= top + 2

    def has_text(self):
        return bool(str(self.field.stringValue()).strip())

    def unfocus(self):
        """Give the keyboard back to the app you were in."""
        self.panel.ov_typing = False
        if self.panel.isKeyWindow():
            self.panel.makeFirstResponder_(None)
            self.panel.resignKeyWindow()

    def focus_field(self):
        if self.state["field"] and self.shown:
            self.panel.ov_typing = True
            self.panel.makeKeyAndOrderFront_(None)
            self.panel.makeFirstResponder_(self.field)

    def rect(self):
        """Where it is now, in screen points, y down from the main screen's top (like gaze points)."""
        main_h = NSScreen.screens()[0].frame().size.height
        f = self.panel.frame()
        return f.origin.x, main_h - f.origin.y - f.size.height, f.size.width, f.size.height

    def _load_art(self, url):
        self.art_url = url

        def go():
            img = NSImage.alloc().initWithContentsOfURL_(NSURL.URLWithString_(url))
            if img is not None and self.art_url == url:
                self.art = img
        threading.Thread(target=go, daemon=True).start()

    # ------------------------------------------------------------------ drawing
    def draw(self, b):
        w, h, r = b.size.width, b.size.height, min(22.0, b.size.height / 2)
        nw = self.nx1 - self.nx0
        # the black shape: flat on top (it hangs from the screen edge), rounded at the bottom, little inward
        # curves at the top corners like the notch
        p = NSBezierPath.bezierPath()
        ear = 6.0
        p.moveToPoint_((0, 0))
        p.lineToPoint_((w, 0))
        p.curveToPoint_controlPoint1_controlPoint2_((w - ear, ear), (w - ear * 0.4, 0), (w - ear, ear * 0.4))
        p.lineToPoint_((w - ear, h - r))
        p.curveToPoint_controlPoint1_controlPoint2_((w - ear - r, h), (w - ear, h - r * 0.45), (w - ear - r * 0.45, h))
        p.lineToPoint_((ear + r, h))
        p.curveToPoint_controlPoint1_controlPoint2_((ear, h - r), (ear + r * 0.45, h), (ear, h - r * 0.45))
        p.lineToPoint_((ear, ear))
        p.curveToPoint_controlPoint1_controlPoint2_((0, 0), (ear, ear * 0.4), (ear * 0.4, 0))
        p.closePath()
        NSColor.blackColor().set()
        p.fill()
        st, t = self.state, time.monotonic()
        wing = (w - nw) / 2
        if st["ind"] and wing > 14:  # live activity, left of the notch; its symbol right of it
            self._indicator(st["ind"], st["amp"], t, ear + max(14.0, wing / 2), self.nh / 2)
            sym = _symbol(SYMBOL.get(st["ind"], "circle"), 12, COLORS.get(st["ind"]))
            if sym is not None:
                sz = sym.size()
                x = w - ear - max(14.0, wing / 2) - sz.width / 2
                sym.drawInRect_fromRect_operation_fraction_respectFlipped_hints_(
                    NSMakeRect(x, self.nh / 2 - sz.height / 2, sz.width, sz.height), NSMakeRect(0, 0, 0, 0), 2,
                    0.95, True, None)
        if w < EXP_W * 0.8 or h < self.nh + 20:
            self.controls = {}
            return
        y = self.nh + 8
        alpha = min(1.0, max(0.0, (h - self.nh - 16) / 30))  # content fades in as it opens
        if st["playing"]:
            y = self._playing(st["playing"], y, w, alpha)
        else:
            self.controls = {}
        for line in st["lines"]:
            who, text, struck = line[0], line[1], (line[2] if len(line) > 2 else None)
            col = _white(0.55 * alpha) if who in ("You", "Typed") else (
                _rgb(0.6, 0.85, 1.0, alpha) if who == "Notification" else _white(0.95 * alpha))
            s = NSAttributedString.alloc().initWithString_attributes_(text, self._text_attrs(col))
            th = s.boundingRectWithSize_options_((w - 2 * PAD, 2000), 1).size.height
            if y + th > h - (FIELD_H + 8 if st["field"] else 6):
                break
            s.drawWithRect_options_(NSMakeRect(PAD, y, w - 2 * PAD, th), 1)
            if struck is not None:  # a misread / not for Claude: a line draws through it, left to right
                self._strike(text, y, w, th, max(0.0, min(1.0, (t - struck) / 0.35)), alpha)
            y += th + 6

    def _strike(self, text, y, w, th, progress, alpha):
        """Strike through each wrapped line of text, the line growing with progress (0..1)."""
        width = w - 2 * PAD
        n = max(1, round(th / (TEXT_FONT * 1.2 + 2)))
        full = NSAttributedString.alloc().initWithString_attributes_(text, self._text_attrs(_white(1))).size().width
        remaining = full * progress
        _white(0.75 * alpha).set()
        for i in range(n):
            seg = min(width, remaining)
            if seg <= 0:
                break
            ly = y + (i + 0.55) * th / n
            p = NSBezierPath.bezierPath()
            p.moveToPoint_((PAD, ly))
            p.lineToPoint_((PAD + seg, ly))
            p.setLineWidth_(1.3)
            p.stroke()
            remaining -= width

    def _indicator(self, kind, amp, t, cx, cy):
        col = COLORS.get(kind, _white(0.9))
        col.set()
        if kind in ("mouth", "voice", "speaking"):
            a = (2.0 + 1.2 * math.sin(2 * math.pi * 1.1 * t)) if kind == "speaking" else 0.8 + 4.5 * max(0.0, min(1.0, amp))
            p = NSBezierPath.bezierPath()
            n = 40
            for i in range(n + 1):
                u = i / n
                x = cx - 13 + 26 * u
                y = cy - a * math.sin(2 * math.pi * (2.5 * u - 1.5 * t)) * math.sin(math.pi * u)
                (p.moveToPoint_ if i == 0 else p.lineToPoint_)((x, y))
            p.setLineWidth_(1.6)
            p.setLineCapStyle_(1)
            p.setLineJoinStyle_(1)
            p.stroke()
        elif kind == "listen":
            r = 2.4 + 0.8 * math.sin(2 * math.pi * 1.2 * t)
            NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(cx - r, cy - r, 2 * r, 2 * r)).fill()
        else:  # loading / working: three dots, a wave running through them
            for k in range(3):
                dy = 2.2 * max(0.0, math.sin(2 * math.pi * (1.4 * t - k * 0.18)))
                NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(cx + (k - 1) * 7 - 1.8, cy - dy - 1.8, 3.6, 3.6)).fill()

    def _playing(self, pl, y, w, alpha):
        art = NSMakeRect(PAD, y, ART, ART)
        clip = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(art, 10, 10)
        if self.art is not None and pl.get("art") == self.art_url:
            from AppKit import NSGraphicsContext
            NSGraphicsContext.saveGraphicsState()
            clip.addClip()
            self.art.drawInRect_fromRect_operation_fraction_respectFlipped_hints_(art, NSMakeRect(0, 0, 0, 0), 2,
                                                                                  alpha, True, None)
            NSGraphicsContext.restoreGraphicsState()
        else:
            _white(0.12 * alpha).set()
            clip.fill()
        tx = PAD + ART + 12
        for text, size, weight, col, dy in ((pl.get("title", ""), 14, 0.4, _white(alpha), 6),
                                            (pl.get("artist", ""), 12, 0.0, _white(0.6 * alpha), 26)):
            s = NSAttributedString.alloc().initWithString_attributes_(
                text, {NSFontAttributeName: NSFont.systemFontOfSize_weight_(size, weight),
                       NSForegroundColorAttributeName: col})
            s.drawWithRect_options_(NSMakeRect(tx, y + dy, w - tx - 120, 20), 1 | (1 << 5))  # (truncates)
        self.controls = {}
        cx = w - PAD - 96
        for i, (name, sym) in enumerate((("previous", "backward.fill"),
                                         ("toggle", "pause.fill" if pl.get("state") == "playing" else "play.fill"),
                                         ("next", "forward.fill"))):
            img = _symbol(sym, 15 if name == "toggle" else 13, _white(alpha))
            x = cx + i * 34
            if img is not None:
                sz = img.size()
                img.drawInRect_fromRect_operation_fraction_respectFlipped_hints_(
                    NSMakeRect(x + 14 - sz.width / 2, y + ART / 2 - sz.height / 2, sz.width, sz.height),
                    NSMakeRect(0, 0, 0, 0), 2, alpha, True, None)
            self.controls[name] = (x, y + ART / 2 - 16, x + 28, y + ART / 2 + 16)
        return y + ART + 14

    def click(self, x, y):
        for name, (x0, y0, x1, y1) in self.controls.items():
            if x0 <= x <= x1 and y0 <= y <= y1 and self.on_control:
                self.on_control(name)
                return

    def close(self):
        self.panel.orderOut_(None)
