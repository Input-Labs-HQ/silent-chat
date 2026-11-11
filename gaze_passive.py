"""The moments silent chat can guess where you're looking, found while you work. This is what replaces calibrating.

gaze.py turns the camera's view of your eyes into a point on the screen, and needs to be told now and then where
you really were looking. Teacher watches for moments that say so, and hands each to gaze.py as a "hint":

  click     you clicked something: your eyes rested on it as the cursor closed in. Of the steady looks in the
            second before the click, the one the model puts nearest the click is taken.
  land      the cursor came to rest somewhere without a click, and your eyes were near it
  caret     you're typing and your eyes are near the text caret (where the caret is comes from Accessibility)
  appear    something small popped up (a notification, the island, a dialog) and your eyes jumped to it
  pursuit   your eyes followed the cursor while you dragged or steered it
  centre    (a new profile's first seconds) you're using the Mac, so your gaze is somewhere around mid-screen

Only clicks are taken at face value. The rest are believed only when the model already puts your gaze roughly
there, so they sharpen it but can't drag it away; gaze.py then weighs every kind by how it has really done.

Privacy: a hint is twelve numbers about your eyes and head plus a point on the screen. No pictures, nothing you
type (only *that* a key was pressed, and where the caret is), nothing that's on the screen. It stays on this Mac.
"""
import collections
import math
import queue
import threading

import numpy as np

SPREAD = 70.0  # points: gaze that stays within this of where it settled is one look


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Teacher:
    """Feed it every camera frame (frame), every click (click), key press (key) and pop-up (appeared); take what
    it found with hints(). Times are time.monotonic(); points are screen points, y down from the main screen's
    top. caret: a function -> (x, y, w, h) of the text caret or None (called on its own thread)."""

    def __init__(self, caret=None):
        self.hist = collections.deque(maxlen=150)  # (t, features or None, predicted gaze or None, cursor): ~5 s
        self.out, self.pending, self.results = [], [], queue.Queue()
        self.clicks, self.keys = collections.deque(maxlen=8), collections.deque(maxlen=12)
        self.last = collections.defaultdict(lambda: -99.0)  # per kind: when it last made a hint
        self.made = collections.deque(maxlen=6)  # the last few hints: (kind, t, point)
        self.caret, self.asking = caret, False
        self.move, self.centred = None, 0  # the cursor movement in progress; centre hints given so far
        self.radius, self.cold, self.screen = 160.0, True, None  # (set by the caller: gaze's error, new profile?, the screen)

    # ------------------------------------------------------------------ what comes in

    def frame(self, t, feats, pred, cursor):
        """One camera frame: the gaze features (None in a blink / without a face), where the model puts the gaze
        (unclamped), and where the cursor is."""
        prev = self.hist[-1] if self.hist else None
        self.hist.append((t, feats, pred, cursor))
        if prev is not None and cursor is not None and prev[3] is not None:
            step = _dist(cursor, prev[3])
            if step / max(1e-3, t - prev[0]) > 60.0:  # moving
                if self.move is None:
                    self.move = {"t0": prev[0], "from": prev[3], "checked": t}
                self.move["last"] = t
            elif self.move is not None and t - self.move["last"] > 0.25:  # it has come to rest
                if _dist(cursor, self.move["from"]) >= 120.0:
                    self.pending.append((self.move["last"] + 0.9, "land", self.move["last"], cursor))
                self.move = None
            if self.move is not None and t - self.move["t0"] >= 0.9 and t - self.move["checked"] >= 0.3:
                self.move["checked"] = t
                self._pursuit(t)
        if (self.cold and feats is not None and self.screen and self.centred < 10 and t - self.last["centre"] > 2.0
                and (self.move is not None or (self.keys and t - self.keys[-1] < 2.0))):
            q = self.screen  # ten of these together say no more than "around the middle, give or take a third"
            self.centred, self.last["centre"] = self.centred + 1, t
            self.out.append({"x": feats, "y": (q[0] + q[2] / 2, q[1] + q[3] * 0.45), "src": "centre", "t": t,
                             "sig": (0.42 * q[2], 0.42 * q[3])})
        while not self.results.empty():
            self.pending.append(self.results.get())
        due = [p for p in self.pending if p[0] <= t]
        if due:
            self.pending = [p for p in self.pending if p[0] > t]
            for _, kind, *rest in due:
                getattr(self, "_" + kind)(*rest)

    def click(self, t, x, y):
        self.clicks.append((t, x, y))
        self.pending.append((t + 0.12, "click", t, (x, y)))  # (judged a moment later: the look may outlast it)

    def key(self, t):
        """A key was pressed (not which). While you type, ask now and then where the caret is."""
        self.keys.append(t)
        if (self.caret is None or self.asking or t - self.last["asked"] < 2.0
                or sum(t - k < 1.5 for k in self.keys) < 3):
            return
        self.asking, self.last["asked"] = True, t

        def ask():
            try:
                r = self.caret()
            except Exception:
                r = None
            if r is not None:
                self.results.put((t, "caret", t, r))
            self.asking = False
        threading.Thread(target=ask, daemon=True).start()

    def appeared(self, t, rect):
        """Something small popped up at rect (x, y, w, h): did the eyes jump to it? (judged 1.6 s later)"""
        self.pending.append((t + 1.6, "appear", t, rect))

    def hints(self):
        out, self.out = self.out, []
        return out

    # ------------------------------------------------------------------ reading the last few seconds

    def looks(self, t0, t1, spread=SPREAD):
        """The steady looks between t0 and t1, oldest first: [{"t0", "t1", "x": median features, "p": the median
        point the model puts it at, "n": frames}]. A look ends when the gaze moves more than `spread` away or is
        lost for over 0.15 s; shorter than 0.1 s (3 frames) doesn't count."""
        out, cur = [], []

        def close():
            if len(cur) >= 3 and cur[-1][0] - cur[0][0] >= 0.1:
                out.append({"t0": cur[0][0], "t1": cur[-1][0], "x": np.median(np.array([c[1] for c in cur]), 0),
                            "p": tuple(np.median(np.array([c[2] for c in cur]), 0)), "n": len(cur)})
            cur.clear()

        for t, f, p, _ in self.hist:
            if t < t0 or t > t1 or f is None or p is None:
                continue
            if cur and (t - cur[-1][0] > 0.15 or _dist(p, np.median(np.array([c[2] for c in cur]), 0)) > spread):
                close()
            cur.append((t, f, p))
        close()
        return out

    def _trace(self, t0, t1):
        return [[t - t1, *f] for t, f, _, _ in self.hist if t0 <= t <= t1 and f is not None]

    # ------------------------------------------------------------------ the kinds of hint

    def _click(self, t, xy):
        if any(k == "click" and t - tt < 2.5 and _dist(p, xy) < 25 for k, tt, p in self.made):
            return  # (a double-click, or the same button again: the same look, counted once)
        cands = [c for c in self.looks(t - 0.9, t + 0.1) if c["t1"] >= t - 0.6]
        if not cands:
            return
        best = min(cands, key=lambda c: _dist(c["p"], xy))
        self._made("click", t, xy, best["x"], trace=self._trace(t - 1.2, t + 0.1))

    def _land(self, t, xy):
        if t - self.last["land"] < 1.5 or any(abs(c[0] - t) < 1.2 and _dist(c[1:], xy) < 40 for c in self.clicks):
            return  # (a click there says the same, better)
        if any(c is not None and _dist(c, xy) > 25 for tt, _, _, c in self.hist if t <= tt <= t + 0.4):
            return  # (it moved on at once: not a rest)
        self._near("land", t, xy, self.looks(t - 0.4, t + 0.5))

    def _caret(self, t, rect):
        x, y, w, h = rect
        if t - self.last["caret"] < 3.0:
            return
        self._near("caret", t, (x + w / 2, y + h / 2), self.looks(t - 0.7, t + 0.2), sig=(190.0, 140.0))

    def _appear(self, t, rect):
        x, y, w, h = rect
        target = (x + w / 2, y + h / 2)
        new = [c for c in self.looks(t + 0.1, t + 1.6) if c["t0"] >= t + 0.15]  # a look that began after it showed
        before = self.looks(t - 0.6, t + 0.1)
        if not new or t - self.last["appear"] < 2.0:
            return
        best = min(new, key=lambda c: _dist(c["p"], target))
        d = _dist(best["p"], target)
        if d > max(1.5 * self.radius, math.hypot(w, h) / 2 + self.radius):
            return
        if before and _dist(before[-1]["p"], target) - d < 80:
            return  # (the eyes didn't move towards it: they were there already, or went elsewhere)
        self._made("appear", t, target, best["x"], sig=(max(100.0, w / 3), max(70.0, h / 3)))

    def _pursuit(self, t):
        rows = [r for r in self.hist if r[0] >= t - 0.9 and r[1] is not None and r[2] is not None and r[3] is not None]
        if len(rows) < 18 or t - self.last["pursuit"] < 0.4:
            return
        cur, gaze = np.array([r[3] for r in rows]), np.array([r[2] for r in rows])
        steps = np.linalg.norm(np.diff(cur, axis=0), axis=1)
        if steps.sum() < 180 or (steps > 1.0).mean() < 0.85:
            return  # (the cursor has to keep moving, and get somewhere)
        ax = int(np.argmax(np.ptp(cur, axis=0)))  # along the way the cursor is going
        c, g = cur[:, ax], gaze[:, ax]
        if np.std(g) < 1e-6 or np.corrcoef(c, g)[0, 1] < 0.85 or not 0.5 <= np.polyfit(c, g, 1)[0] <= 2.0:
            return
        for part in np.array_split(np.arange(len(c)), 3):  # all the way along, not one jump (eyes first, cursor after)
            dc, dg = c[part[-1]] - c[part[0]], g[part[-1]] - g[part[0]]
            if abs(dc) >= 40 and (dc * dg <= 0 or abs(dg) < 0.3 * abs(dc)):
                return
        if np.median(np.linalg.norm(gaze - cur, axis=1)) > 2.0 * self.radius:
            return
        mid = [r for r in rows if t - 0.55 <= r[0] <= t - 0.35]
        if len(mid) >= 3:
            self._made("pursuit", t, mid[len(mid) // 2][3], np.median(np.array([r[1] for r in mid]), 0))

    def _near(self, kind, t, xy, cands, sig=None):
        """A hint of a kind that's only believed when the model already puts the gaze near it."""
        if not cands:
            return
        best = min(cands, key=lambda c: _dist(c["p"], xy))
        if _dist(best["p"], xy) <= 1.5 * self.radius:
            self._made(kind, t, xy, best["x"], sig=sig)

    def _made(self, kind, t, xy, x, sig=None, trace=None):
        self.last[kind] = t
        self.made.append((kind, t, xy))
        self.out.append({"x": np.asarray(x, float), "y": (float(xy[0]), float(xy[1])), "src": kind, "t": t,
                         "sig": sig, "trace": trace})


# ----------------------------------------------------------------------------- what macOS can tell us

def caret_rect():
    """Where the text caret is: (x, y, w, h) in screen points, y down from the main screen's top, or None. Asks
    Accessibility for the caret's position only, never the text; password fields are skipped."""
    import ApplicationServices as AX
    err, el = AX.AXUIElementCopyAttributeValue(AX.AXUIElementCreateSystemWide(), "AXFocusedUIElement", None)
    if err != 0 or el is None:
        return None
    AX.AXUIElementSetMessagingTimeout(el, 0.3)  # (a busy app can't hold this up)
    err, sub = AX.AXUIElementCopyAttributeValue(el, "AXSubrole", None)
    if err == 0 and sub == "AXSecureTextField":
        return None
    for selected, bounds in (("AXSelectedTextRange", "AXBoundsForRange"),
                             ("AXSelectedTextMarkerRange", "AXBoundsForTextMarkerRange")):  # (plain fields, web pages)
        err, rng = AX.AXUIElementCopyAttributeValue(el, selected, None)
        if err != 0 or rng is None:
            continue
        err, v = AX.AXUIElementCopyParameterizedAttributeValue(el, bounds, rng, None)
        if err != 0 or v is None:
            continue
        try:
            r = AX.AXValueGetValue(v, AX.kAXValueCGRectType, None)[1]
            x, y, w, h = r.origin.x, r.origin.y, r.size.width, r.size.height
        except Exception:
            continue
        if 6 <= h <= 90 and w < 600 and (x, y) != (0, 0):  # (a caret or a short selection, not a whole paragraph)
            return float(x), float(y), float(w), float(h)
    return None


def front_window():
    """(window id, (x, y, w, h)) of the window in front, or None."""
    import Quartz
    for w in Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly
                                               | Quartz.kCGWindowListExcludeDesktopElements, Quartz.kCGNullWindowID) or []:
        if w.get("kCGWindowLayer") == 0 and w.get("kCGWindowAlpha", 1) > 0:
            b = w["kCGWindowBounds"]
            return int(w["kCGWindowNumber"]), (float(b["X"]), float(b["Y"]), float(b["Width"]), float(b["Height"]))
    return None


class WindowWatch:
    """Notices a small window coming to the front that wasn't there a moment ago (a dialog, a sheet, a picker):
    your eyes usually go to it. Checks once a second on its own thread; big windows say too little to be worth it.
    seen() -> [(time.monotonic() it was noticed, rect)]."""

    def __init__(self, every=1.0, skip=None):
        self.every, self.skip, self.found, self.known = every, skip or (lambda: False), queue.Queue(), {}
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        import time
        last = None
        while True:
            time.sleep(self.every)
            if self.skip():
                continue
            try:
                w = front_window()
            except Exception:
                w = None
            now = time.monotonic()
            if w is None:
                continue
            wid, (x, y, ww, hh) = w
            if wid != last and now - self.known.get(wid, -99) > 30 and 150 <= ww <= 720 and 80 <= hh <= 520:
                self.found.put((now - self.every / 2, (x, y, ww, hh)))
            last, self.known[wid] = wid, now
            for k in [k for k, v in self.known.items() if now - v > 600]:
                del self.known[k]

    def seen(self):
        out = []
        while not self.found.empty():
            out.append(self.found.get())
        return out
