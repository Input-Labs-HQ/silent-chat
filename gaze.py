"""Where you're looking on screen: the camera's iris and head pose, steadied by AirPods head motion, calibrated to you.

Webcam eye tracking is coarse (a few centimetres), so silent chat doesn't click where it thinks you look: it uses
the gaze point to pick the nearest button / link / field (snapping), and tells Claude what you were looking at
when you said "this". `/gaze calibrate` measures the real error on fresh points and saves it with the model.

Features per camera frame: each iris's position inside its eye (corner-to-corner axis), eyelid opening (vertical
gaze moves the lids), head yaw / pitch and head position. Head direction: the camera's pose is absolute but jittery;
AirPods' is precise but drifts. They're fused: slow part from the camera, fast part from the AirPods.
"""
import collections
import json
import math
import random
import threading
import time
from pathlib import Path

import numpy as np

SAMPLES = Path(__file__).resolve().parent / "data" / "silent_chat" / "gaze_samples.npz"
CLICKS = Path(__file__).resolve().parent / "data" / "silent_chat" / "gaze_clicks.npz"
R_EYE = dict(outer=33, inner=133, upper=159, lower=145, iris=468)
L_EYE = dict(outer=263, inner=362, upper=386, lower=374, iris=473)
NAMES = ["rh", "rv", "ro", "lh", "lv", "lo", "yaw", "pitch", "tx", "ty", "tz"]


def _eye(P, e):
    o, i = P[e["outer"], :2], P[e["inner"], :2]
    c, ax = (o + i) / 2, i - o
    w = float(np.linalg.norm(ax)) or 1.0
    ax = ax / w
    ay = np.array([-ax[1], ax[0]])
    d = P[e["iris"], :2] - c
    return d @ ax / w, d @ ay / w, float(np.linalg.norm(P[e["upper"], :2] - P[e["lower"], :2])) / w


def head_angles(matrix):
    r = np.asarray(matrix)[:3, :3]
    return (math.degrees(math.atan2(r[0, 2], r[2, 2])), math.degrees(math.asin(max(-1, min(1, -r[1, 2])))))


class HeadFusion:
    """Head yaw / pitch: the camera's slow (absolute) part + the AirPods' fast (precise) part. The AirPods' axes and
    signs are learned on the fly from how both move together; until that's clear, the camera alone (smoothed)."""

    def __init__(self, tau=1.5):
        self.tau, self.cam, self.ap_slow, self.t = tau, None, None, None
        self.hist = collections.deque(maxlen=600)  # (t, cam yaw, cam pitch, ap yaw, ap pitch)
        self.gain = (0.0, 0.0)
        self.fitted = 0.0

    def update(self, t, cam, ap=None):
        """cam: (yaw, pitch) from the camera; ap: (yaw, pitch) from the AirPods or None -> fused (yaw, pitch)."""
        dt = 0.033 if self.t is None else max(1e-3, min(0.5, t - self.t))
        self.t = t
        a = 1 - math.exp(-dt / self.tau)
        c = np.array(cam, float)
        if self.cam is None:
            self.cam, self.cam_fast = c.copy(), c.copy()
        self.cam += a * (c - self.cam)
        self.cam_fast += (1 - math.exp(-dt / 0.12)) * (c - self.cam_fast)  # (camera only: a light smoothing)
        if ap is None:
            self.ap_slow = None
            return tuple(self.cam_fast)
        ap = np.array(ap, float)
        if self.ap_slow is None:
            self.ap_slow = ap.copy()
        self.ap_slow += a * (ap - self.ap_slow)
        self.hist.append((t, *cam, *ap))
        if t - self.fitted > 2 and len(self.hist) > 90:
            self._fit()
            self.fitted = t
        if self.gain == (0.0, 0.0):  # (AirPods' axes not worked out yet)
            return tuple(self.cam_fast)
        hp = ap - self.ap_slow
        g = np.array(self.gain)
        return tuple(np.where(g != 0, self.cam + g * hp, self.cam_fast))

    def _fit(self):
        h = np.array(self.hist)[:, 1:]
        k = np.ones(6) / 6  # smooth, then compare changes over ~1/3 s: camera noise would swamp frame-to-frame ones
        sm = np.stack([np.convolve(h[:, j], k, mode="valid") for j in range(4)], 1)
        d = sm[10:] - sm[:-10]
        gains = []
        for c, k in ((0, 2), (1, 3)):  # camera yaw vs AirPods yaw, camera pitch vs AirPods pitch
            x, y = d[:, k], d[:, c]
            if x.std() < 0.05 or y.std() < 0.05:
                gains.append(0.0)
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            g = float((x * y).sum() / (x * x).sum())
            gains.append(max(-1.5, min(1.5, g)) if abs(r) > 0.5 else 0.0)
        self.gain = tuple(gains)


def features(landmarks, matrix, head):
    """One frame -> feature vector (None without a face)."""
    if landmarks is None or matrix is None or len(landmarks) < 478:
        return None
    P = np.asarray(landmarks)
    rh, rv, ro = _eye(P, R_EYE)
    lh, lv, lo = _eye(P, L_EYE)
    t = np.asarray(matrix)[:3, 3]
    return np.array([rh, rv, ro, lh, lv, lo, head[0], head[1], t[0], t[1], t[2]], float)


def _expand(Z):
    """Standardised features + a few curvature terms (gaze isn't linear in iris position near the edges)."""
    eh, ev = (Z[:, 0] + Z[:, 3]) / 2, (Z[:, 1] + Z[:, 4]) / 2
    yaw, pitch = Z[:, 6], Z[:, 7]
    extra = np.stack([eh * eh, ev * ev, eh * ev, eh * yaw, ev * pitch], 1)
    return np.hstack([np.ones((len(Z), 1)), Z, extra])


# Ridge penalty per expanded column: [offset, 6 eye, yaw, pitch, tx, ty, tz, 5 curvature]. Head *position* is held
# almost still while calibrating, so a free fit gives it a huge gain (1 cm of head bob = a third of the screen) that
# throws the point off screen as soon as you move. Physically 1 cm of head movement moves your gaze only ~1 cm (about
# 50 pt), less than the error, so it's left out; curvature terms are shrunk (they blow up outside the calibrated range).
PENALTY = np.array([0] + [1] * 6 + [1, 1] + [1e5, 1e5, 1e5] + [30] * 5, float)  # (1e5: effectively left out;
# on the first real calibration that cut fresh-point error 100 -> 80 pt and head-position sensitivity ~220 pt/cm -> 0)
CLIP_SD = 2.5  # features are clipped to the calibrated range, +-this many sd: no wild extrapolation


class Model:
    def __init__(self, mean, std, coef, lo=None, hi=None):
        self.mean, self.std, self.coef = np.asarray(mean), np.asarray(std), np.asarray(coef)
        self.lo = np.asarray(lo) if lo is not None else -CLIP_SD * np.ones(len(self.mean))
        self.hi = np.asarray(hi) if hi is not None else CLIP_SD * np.ones(len(self.mean))

    def predict(self, X):
        X = np.atleast_2d(X)
        return _expand(np.clip((X - self.mean) / self.std, self.lo, self.hi)) @ self.coef

    @classmethod
    def fit(cls, X, Y, groups, lams=(0.03, 0.1, 0.3, 1, 3, 10, 30, 100)):
        """Ridge regression; the strength is chosen by leave-one-point-out cross-validation. -> (model, cv error)."""
        mean, std = X.mean(0), X.std(0) + 1e-6
        Z = (X - mean) / std
        lo, hi = Z.min(0) - CLIP_SD * 0.4, Z.max(0) + CLIP_SD * 0.4
        A = _expand(Z)
        best = None
        for lam in lams:
            errs = []
            for g in np.unique(groups):
                tr, te = groups != g, groups == g
                c = _ridge(A[tr], Y[tr], lam)
                errs.append(np.linalg.norm(np.median(A[te] @ c, 0) - Y[te][0]))
            e = float(np.median(errs))
            if best is None or e < best[0]:
                best = (e, lam)
        return cls(mean, std, _ridge(A, Y, best[1]), lo, hi), best[0]

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "coef": self.coef.tolist(),
                "lo": self.lo.tolist(), "hi": self.hi.tolist()}


def _ridge(A, Y, lam):
    I = np.diag(PENALTY[:A.shape[1]])  # (the offset isn't shrunk)
    return np.linalg.solve(A.T @ A + lam * I, A.T @ Y)


class OneEuro:
    """Smooths the gaze point: steady when you fixate, quick when your eyes jump."""

    def __init__(self, min_cutoff=0.35, beta=0.003):
        self.min_cutoff, self.beta, self.x, self.dx, self.t = min_cutoff, beta, None, 0.0, None

    @staticmethod
    def _a(cutoff, dt):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, t, x):
        x = np.asarray(x, float)
        if self.x is None:
            self.x, self.t = x, t
            return x
        dt = max(1e-3, t - self.t)
        self.t = t
        dx = (x - self.x) / dt
        self.dx = self.dx + self._a(1.0, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        self.x = self.x + self._a(cutoff, dt) * (x - self.x)
        return self.x


# ----------------------------------------------------------------------------- screens

def screens():
    """[(AppKit frame, Quartz rect (x, y, w, h) with y down from the main screen's top, mm per point)]."""
    from AppKit import NSScreen
    import Quartz
    out = []
    main_h = NSScreen.screens()[0].frame().size.height
    for s in NSScreen.screens():
        f = s.frame()
        did = s.deviceDescription()["NSScreenNumber"]
        mm = Quartz.CGDisplayScreenSize(did)
        out.append((f, (f.origin.x, main_h - f.origin.y - f.size.height, f.size.width, f.size.height),
                    (mm.width / f.size.width) if mm.width else 0.0, int(did)))
    return out


def screen_under_mouse():
    from AppKit import NSEvent
    m = NSEvent.mouseLocation()
    for s in screens():
        f = s[0]
        if f.origin.x <= m.x < f.origin.x + f.size.width and f.origin.y <= m.y < f.origin.y + f.size.height:
            return s
    return screens()[0]


_VIEW = None


def _view_class():
    global _VIEW
    if _VIEW is not None:
        return _VIEW
    from AppKit import (NSBezierPath, NSColor, NSFont, NSFontAttributeName, NSForegroundColorAttributeName,
                        NSMakeRect, NSString, NSView)

    class GazeDotView(NSView):
        def isFlipped(self):
            return True  # y down, like screen (Quartz) coordinates

        def drawRect_(self, rect):
            st = getattr(self, "st", {})
            b = self.bounds()
            if st.get("dim"):
                NSColor.colorWithCalibratedWhite_alpha_(0.04, st["dim"]).set()
                NSBezierPath.fillRect_(b)
            for (x, y, r, kind) in st.get("dots", []):
                if kind == "target":
                    NSColor.whiteColor().set()
                    NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(x - r, y - r, 2 * r, 2 * r)).fill()
                    NSColor.colorWithSRGBRed_green_blue_alpha_(1, 0.25, 0.3, 1).set()
                    NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(x - 3, y - 3, 6, 6)).fill()
                else:  # the live gaze point: a ring
                    NSColor.colorWithSRGBRed_green_blue_alpha_(0.4, 0.8, 1, 0.9).set()
                    p = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(x - r, y - r, 2 * r, 2 * r))
                    p.setLineWidth_(3)
                    p.stroke()
            if st.get("text"):
                attrs = {NSFontAttributeName: NSFont.systemFontOfSize_weight_(22, 0.3),
                         NSForegroundColorAttributeName: NSColor.whiteColor()}
                s = NSString.stringWithString_(st["text"])
                size = s.sizeWithAttributes_(attrs)
                s.drawAtPoint_withAttributes_(((b.size.width - size.width) / 2, b.size.height * 0.42), attrs)

    _VIEW = GazeDotView
    return _VIEW


class ScreenLayer:
    """A click-through full-screen layer above everything on one screen: calibration dots or the live gaze ring."""

    def __init__(self, screen):
        from AppKit import (NSBackingStoreBuffered, NSColor, NSScreenSaverWindowLevel, NSWindow,
                            NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorFullScreenAuxiliary,
                            NSWindowStyleMaskBorderless)
        self.screen = screen
        f = screen[0]
        self.win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            f, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
        self.win.setLevel_(NSScreenSaverWindowLevel)
        self.win.setOpaque_(False)
        self.win.setBackgroundColor_(NSColor.clearColor())
        self.win.setIgnoresMouseEvents_(True)
        self.win.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces |
                                        NSWindowCollectionBehaviorFullScreenAuxiliary)
        self.view = _view_class().alloc().initWithFrame_(((0, 0), (f.size.width, f.size.height)))
        self.view.st = {}
        self.win.setContentView_(self.view)
        self.win.setFrame_display_(f, True)
        self.win.orderFrontRegardless()

    def to_local(self, x, y):
        q = self.screen[1]
        return x - q[0], y - q[1]

    def show(self, **st):
        self.view.st = st
        self.view.setNeedsDisplay_(True)

    def close(self):
        self.win.orderOut_(None)


# ----------------------------------------------------------------------------- calibration

class Calibration:
    """Shows dots one at a time on the screen under the mouse. 20 to learn from, then 9 new ones to measure the real
    error (never used for fitting). ~60 s. step() each camera frame; .result when done."""

    SHOW, SETTLE = 1.7, 0.7  # seconds per dot; samples are taken after the eyes have landed

    def __init__(self, seed=None):
        self.screen = screen_under_mouse()
        _, (qx, qy, qw, qh), self.mm, _ = self.screen
        rng = random.Random(seed)
        fit = [(0.06 + 0.88 * i / 4, 0.08 + 0.84 * j / 3) for i in range(5) for j in range(4)]
        test = [(0.17 + 0.33 * i + rng.uniform(-0.06, 0.06), 0.2 + 0.3 * j + rng.uniform(-0.06, 0.06))
                for i in range(3) for j in range(3)]
        rng.shuffle(fit)
        rng.shuffle(test)
        self.points = [("fit", qx + u * qw, qy + v * qh) for u, v in fit] + \
                      [("test", qx + u * qw, qy + v * qh) for u, v in test]
        self.layer = ScreenLayer(self.screen)
        self.i, self.t0, self.samples, self.result, self.intro = 0, None, [], None, None

    def step(self, now, feats):
        """feats: this frame's feature vector or None (blink / no face). False when finished."""
        if self.intro is None:
            self.intro = now
        if now - self.intro < 3.0:  # instructions first
            self.layer.show(dim=0.88, text="Look at each dot until it moves. Keep your head natural.  (\\ cancels)")
            return True
        if self.i >= len(self.points):
            return False
        kind, x, y = self.points[self.i]
        if self.t0 is None:
            self.t0 = now
        el = now - self.t0
        r = 14 - 7 * min(1.0, el / self.SHOW)  # shrinks: draws the eyes to its centre
        lx, ly = self.layer.to_local(x, y)
        label = "measuring accuracy …" if kind == "test" and el < 0.6 and self.points[self.i - 1][0] == "fit" else ""
        self.layer.show(dim=0.88, dots=[(lx, ly, r, "target")], text=label)
        if el >= self.SETTLE and feats is not None:
            self.samples.append((self.i, kind, x, y, feats))
        if el >= self.SHOW:
            self.i, self.t0 = self.i + 1, None
            if self.i >= len(self.points):
                self.finish()
                return False
        return True

    def finish(self, save=True):
        if self.layer is not None:
            self.layer.close()
        fit = [s for s in self.samples if s[1] == "fit"]
        test = [s for s in self.samples if s[1] == "test"]
        if len({s[0] for s in fit}) < 12 or len({s[0] for s in test}) < 5:
            self.result = {"error": "Not enough clear looks at the dots (face not visible, or eyes closed)."}
            return
        try:  # (kept, so the model can be re-tuned without calibrating again: refit())
            if not save:
                raise OSError
            np.savez(SAMPLES, X=np.array([s[4] for s in self.samples]), Y=np.array([[s[2], s[3]] for s in self.samples]),
                     g=np.array([s[0] for s in self.samples]), test=np.array([s[1] == "test" for s in self.samples]))
        except OSError:
            pass
        X, Y = np.array([s[4] for s in fit]), np.array([[s[2], s[3]] for s in fit])
        X, Y, g = _clean(X, Y, np.array([s[0] for s in fit]))
        model, cv = Model.fit(X, Y, g)
        # the real test: points it never saw. Per frame (raw), and per look (median over the look, what "this" uses)
        per_frame, per_look = [], []
        for i in sorted({s[0] for s in test}):
            rows = [s for s in test if s[0] == i]
            P = model.predict(np.array([s[4] for s in rows]))
            target = np.array([rows[0][2], rows[0][3]])
            per_frame += list(np.linalg.norm(P - target, axis=1))
            per_look.append(float(np.linalg.norm(np.median(P, 0) - target)))
        mm = self.mm
        self.result = {"model": model.to_dict(), "screen": self.screen[1], "display": self.screen[3],
                       "mm_per_pt": mm, "cv_error_pt": cv, "n_fit": len(X),
                       "look_error_pt": float(np.median(per_look)), "look_error_mean_pt": float(np.mean(per_look)),
                       "look_error_max_pt": float(np.max(per_look)), "frame_error_pt": float(np.median(per_frame)),
                       "within_100pt": float(np.mean(np.array(per_look) < 100)), "looks": per_look,
                       "date": time.strftime("%Y-%m-%d %H:%M")}

    def cancel(self):
        self.layer.close()


def refit(path):
    """Fit again from the last calibration's saved frames (after changing the model), without calibrating again.
    Rewrites gaze.json and logs the new result. -> result dict."""
    z = np.load(SAMPLES)
    c = Calibration.__new__(Calibration)
    d = json.loads(Path(path).read_text())
    scr = next((s for s in screens() if s[3] == d.get("display")), None) or (None, tuple(d["screen"]), d["mm_per_pt"], d.get("display"))
    c.screen, c.mm, c.layer = scr, d["mm_per_pt"], None
    c.samples = [(int(i), "test" if t else "fit", float(y[0]), float(y[1]), x) for x, y, i, t in zip(z["X"], z["Y"], z["g"], z["test"])]
    c.finish(save=False)
    r = c.result
    if "model" in r:
        r["refit_of"] = d.get("date")
        Path(path).write_text(json.dumps(r))
        with open(Path(path).with_name("gaze_eval.log"), "a") as f:
            f.write(json.dumps({k: v for k, v in r.items() if k != "model"}) + "\n")
    return r


def _clean(X, Y, g):
    """Drop frames far from the rest of their dot's frames (blinks, glances away)."""
    keep = np.ones(len(X), bool)
    for i in np.unique(g):
        m = g == i
        med = np.median(X[m], 0)
        mad = np.median(np.abs(X[m] - med), 0) + 1e-6
        keep[m] = (np.abs(X[m] - med) / mad < 6).all(1)
    return X[keep], Y[keep], g[keep]


def describe(r):
    """The measured accuracy in words."""
    if "error" in r:
        return r["error"]
    mm = r["mm_per_pt"]
    cm = (lambda pt: f"{pt * mm / 10:.1f} cm") if mm else (lambda pt: f"{pt:.0f} pt")
    return (f"typical error {cm(r['look_error_pt'])} ({r['look_error_pt']:.0f} pt) on 9 fresh points, worst "
            f"{cm(r['look_error_max_pt'])}; {r['within_100pt']:.0%} of looks within 100 pt")


# ----------------------------------------------------------------------------- live

class Gaze:
    """Live gaze from each camera frame. .point(t0, t1) -> median gaze (x, y) in screen points over that time."""

    def __init__(self, path):
        self.path, self.model, self.info = Path(path), None, {}
        self.head, self.filter = HeadFusion(), OneEuro()
        self.hist = collections.deque(maxlen=300)  # (t, x, y)
        self.last_feats, self.calib, self.layer, self.show_until = None, None, None, 0.0
        self.lens, self.lens_on, self.warp_on = None, False, False  # the glass circle that follows your gaze
        # free calibration from your clicks: you look at what you click, so each click is a labelled example
        self.feat_hist = collections.deque(maxlen=60)  # (t, features) for the last ~2 s
        self.clicks, self.refit_at, self.click_note = [], 0, None
        try:
            z = np.load(CLICKS)
            self.clicks = [(f, float(x), float(y)) for f, (x, y) in zip(z["X"], z["Y"])]
            self.refit_at = len(self.clicks)
        except (OSError, KeyError, ValueError):
            pass
        try:
            d = json.loads(self.path.read_text())
            self.model, self.info = Model(**d["model"]), d
        except (OSError, ValueError, KeyError):
            pass

    @property
    def ready(self):
        return self.model is not None

    def update(self, t, landmarks, matrix, blend, airpods=None):
        """airpods: (t, pitch, yaw, roll) degrees, or None."""
        self.last_feats = None
        if landmarks is None or matrix is None:
            return self._lens(t, None)
        if (blend.get("eyeBlinkLeft", 0) + blend.get("eyeBlinkRight", 0)) / 2 > 0.45:
            return self._lens(t, None)  # (eyes closed: no gaze)
        cam = head_angles(matrix)
        ap = (airpods[2], airpods[1]) if airpods else None
        head = self.head.update(t, cam, ap)
        f = features(landmarks, matrix, head)
        self.last_feats = f
        if f is not None:
            self.feat_hist.append((t, f))
        if self.calib is not None or self.model is None or f is None:
            return self._lens(t, None)
        px, py = self.model.predict(f)[0]
        q = self.info.get("screen") or (0, 0, 1e5, 1e5)
        m = 0.05 * min(q[2], q[3])  # (kept on the calibrated screen, give or take a little)
        px, py = min(max(px, q[0] - m), q[0] + q[2] + m), min(max(py, q[1] - m), q[1] + q[3] + m)
        x, y = self.filter(t, (px, py))
        self.hist.append((t, float(x), float(y)))
        self._lens(t, (float(x), float(y)))
        if self.layer is not None:
            if t > self.show_until:
                self.layer.close()
                self.layer = None
            else:
                lx, ly = self.layer.to_local(x, y)
                self.layer.show(dots=[(lx, ly, 22, "ring")])

    def _lens(self, t, p):
        if not self.lens_on or self.calib is not None or self.model is None:
            return self.hide_lens()
        if self.lens is None:
            self.lens = GlassLens(hold=max(40.0, 0.5 * self.radius), warp=self.warp_on and Warper.allowed())
        self.lens.update(t, p)

    def add_click(self, t, x, y):
        """A click at screen point (x, y), y down. Uses where your eyes were just before it (0.05-0.45 s) as a new
        calibration example, if it was on the calibrated screen and roughly where gaze said you were looking (a
        blind click, e.g. while reading something else, is skipped). Every 10 new clicks it refits and keeps the
        new model only if it's more accurate on clicks it didn't train on. -> a note when it improved, else None."""
        if self.model is None or self.calib is not None:
            return None
        q = self.info.get("screen")
        if not q or not (q[0] <= x <= q[0] + q[2] and q[1] <= y <= q[1] + q[3]):
            return None
        fs = [f for tt, f in self.feat_hist if t - 0.45 <= tt <= t - 0.05]
        if len(fs) < 4:
            return None
        f = np.median(np.array(fs), 0)
        if np.linalg.norm(self.model.predict(f)[0] - np.array([x, y])) > max(300.0, 3 * self.radius):
            return None
        self.clicks.append((f, float(x), float(y)))
        self.clicks = self.clicks[-400:]
        if len(self.clicks) >= 20 and len(self.clicks) - self.refit_at >= 10:
            self.refit_at = len(self.clicks)
            return self.refit_with_clicks()
        return None

    def refit_with_clicks(self):
        """Calibration frames + your clicks -> a new model, judged on the newest quarter of the clicks."""
        try:
            np.savez(CLICKS, X=np.array([c[0] for c in self.clicks]), Y=np.array([c[1:] for c in self.clicks]))
            z = np.load(SAMPLES)
        except (OSError, KeyError, ValueError):
            return None
        n_test = max(5, len(self.clicks) // 4)
        train, test = self.clicks[:-n_test], self.clicks[-n_test:]
        fit = ~z["test"]
        X, Y, g = _clean(z["X"][fit], z["Y"][fit], z["g"][fit])
        cX = np.array([c[0] for c in train])
        cY = np.array([c[1:] for c in train])
        cg = 1000 + np.arange(len(train))
        rep = 4  # (a click is one look; a calibration dot is ~20 frames: weight clicks up a little)
        X2 = np.vstack([X] + [cX] * rep)
        Y2 = np.vstack([Y] + [cY] * rep)
        g2 = np.concatenate([g] + [cg] * rep)
        new, _ = Model.fit(X2, Y2, g2)
        tX, tY = np.array([c[0] for c in test]), np.array([c[1:] for c in test])
        old_err = float(np.median(np.linalg.norm(self.model.predict(tX) - tY, axis=1)))
        new_err = float(np.median(np.linalg.norm(new.predict(tX) - tY, axis=1)))
        entry = {"date": time.strftime("%Y-%m-%d %H:%M"), "clicks": len(self.clicks), "old_click_error_pt": old_err,
                 "new_click_error_pt": new_err, "kept": new_err < old_err * 0.95}
        with open(self.path.with_name("gaze_eval.log"), "a") as fh:
            fh.write(json.dumps({"click_refit": entry}) + "\n")
        if not entry["kept"]:
            return None
        self.model = new
        self.info.update(model=new.to_dict(), look_error_pt=min(self.info.get("look_error_pt", new_err), new_err),
                         click_refits=self.info.get("click_refits", 0) + 1, click_error_pt=new_err)
        self.path.write_text(json.dumps(self.info))
        mm = self.info.get("mm_per_pt") or 0
        return (f"Gaze tuned from your clicks: {old_err:.0f} -> {new_err:.0f} pt"
                + (f" ({new_err * mm / 10:.1f} cm)" if mm else "") + " on clicks it hadn't seen.")

    def hide_lens(self):
        if self.lens is not None:
            self.lens.close()
            self.lens = None

    def point(self, t0, t1):
        pts = [(x, y) for t, x, y in self.hist if t0 <= t <= t1]
        if len(pts) < 3:
            return None
        return tuple(np.median(np.array(pts), 0))

    def show(self, seconds=20):
        """Draw a ring where it thinks you're looking, for a while (to see how well it works)."""
        if self.model is None:
            return
        scr = next((s for s in screens() if s[3] == self.info.get("display")), screens()[0])
        if self.layer is None:
            self.layer = ScreenLayer(scr)
        self.show_until = time.monotonic() + seconds

    def start_calibration(self):
        self.calib = Calibration()

    def step_calibration(self, now):
        """Each frame while calibrating. Returns the result dict when done (and saves the model), else None."""
        if self.calib is None:
            return None
        if self.calib.step(now, self.last_feats):
            return None
        r, self.calib = self.calib.result, None
        if "model" in r:
            self.path.write_text(json.dumps(r))
            self.model, self.info = Model(**r["model"]), r
            self.filter, self.hist = OneEuro(), collections.deque(maxlen=300)
            with open(self.path.with_name("gaze_eval.log"), "a") as f:
                f.write(json.dumps({k: v for k, v in r.items() if k != "model"}) + "\n")
        return r

    def cancel_calibration(self):
        if self.calib is not None:
            self.calib.cancel()
            self.calib = None

    @property
    def radius(self):
        """How far off it typically is (screen points): the snapping radius."""
        return float(self.info.get("look_error_pt", 150))


# ----------------------------------------------------------------------------- the glass indicator

_LENS_VIEW = None


def _lens_view_class():
    """Barely there: a soft refraction edge (a faint light band with a fainter shadow just outside, no hard line),
    like the rim of a drop of water on the screen, plus a ripple that spreads and fades each time your gaze lands
    somewhere new. Clear in the middle, so you read straight through it. (macOS's own Liquid Glass material can't
    see other apps' windows, so it frosts everything behind it into a grey disc that hides text; this draws the
    look instead.)"""
    global _LENS_VIEW
    if _LENS_VIEW is not None:
        return _LENS_VIEW
    from AppKit import NSBezierPath, NSColor, NSMakeRect, NSView

    class GlassLensView(NSView):
        def drawRect_(self, rect):
            b = self.bounds()
            c = (b.size.width / 2, b.size.height / 2)
            a = getattr(self, "alpha", 1.0)
            R = getattr(self, "radius", 26.0)
            w = lambda v, al: NSColor.colorWithCalibratedWhite_alpha_(v, max(0.0, al))
            if a > 0.01:  # the resting edge: a faint light band fading to nothing on both sides (thin rings)
                for i in range(10):
                    f = i / 9
                    rr = R * (0.72 + 0.3 * f)
                    al = 0.10 * a * math.exp(-((f - 0.6) / 0.22) ** 2)
                    p = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(c[0] - rr, c[1] - rr, 2 * rr, 2 * rr))
                    p.setLineWidth_(R * 0.033 + 0.2)
                    w(1, al).set()
                    p.stroke()
            for age in getattr(self, "ripples", []):  # spreading rings, 0 -> 1 over their life
                r = R * (0.35 + 1.15 * age)
                fade = (1 - age) ** 1.6 * a
                for k, (dr, al, lw) in enumerate(((0.0, 0.16, 1.4), (-5.0, 0.07, 1.0))):
                    rr = r + dr
                    if rr <= 2:
                        continue
                    p = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(c[0] - rr, c[1] - rr, 2 * rr, 2 * rr))
                    p.setLineWidth_(lw)
                    w(1, al * fade).set()
                    p.stroke()

    _LENS_VIEW = GlassLensView
    return _LENS_VIEW


class Warper:
    """The real glass bend: a background thread copies the small patch of screen under the lens (everything below
    its own window, so not the lens itself), bends it with Core Image (a gentle bulge, strongest in the middle and
    fading to nothing at the edge, so it blends in seamlessly), and hands the result to the lens to show. Needs
    Screen Recording permission (without it macOS only gives the wallpaper, so it stays off). ~10 ms per frame at
    ~24 fps on one background thread; nothing is saved or sent anywhere."""

    def __init__(self, size, radius, fps=24):
        import Quartz
        self.Q, self.size, self.radius, self.every = Quartz, size, radius, 1.0 / fps
        self.ctx = Quartz.CIContext.contextWithOptions_(None)
        self.filter = Quartz.CIFilter.filterWithName_("CIBumpDistortion")
        self.rect, self.win_id, self.scale, self.image, self.running = None, 0, 0.0, None, True
        self.frames, self.busy = 0, 0.0
        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def allowed(ask=False):
        import Quartz
        if Quartz.CGPreflightScreenCaptureAccess():
            return True
        if ask:
            Quartz.CGRequestScreenCaptureAccess()  # (macOS shows its prompt once; the answer applies after a restart)
        return False

    def _run(self):
        Q = self.Q
        while self.running:
            t0 = time.monotonic()
            rect, win_id, scale = self.rect, self.win_id, self.scale
            if rect is not None and win_id and scale > 0.001:
                try:
                    img = Q.CGWindowListCreateImage(Q.CGRectMake(*rect), Q.kCGWindowListOptionOnScreenBelowWindow,
                                                    win_id, Q.kCGWindowImageBestResolution)
                    if img is not None:
                        w, h = Q.CGImageGetWidth(img), Q.CGImageGetHeight(img)
                        px = w / rect[2]
                        ci = Q.CIImage.imageWithCGImage_(img)
                        f = self.filter
                        f.setValue_forKey_(ci, "inputImage")
                        f.setValue_forKey_(Q.CIVector.vectorWithX_Y_(w / 2, h / 2), "inputCenter")
                        f.setValue_forKey_(self.radius * px, "inputRadius")
                        f.setValue_forKey_(scale, "inputScale")
                        self.image = self.ctx.createCGImage_fromRect_(f.outputImage(), ci.extent())
                        self.frames += 1
                except Exception:
                    pass
            used = time.monotonic() - t0
            self.busy = 0.95 * self.busy + 0.05 * used
            time.sleep(max(0.005, self.every - used))

    def stop(self):
        self.running = False


class GlassLens:
    """A faint glass edge that drifts to where you're looking, with a soft ripple when your gaze lands somewhere new.
    Steady while you read: it only moves once your gaze has clearly moved on (more than `hold` points away), then
    glides there; it fades out when your eyes are closed or away. Its own small click-through window, so moving it
    costs next to nothing."""

    RIPPLE_S = 0.9

    def __init__(self, radius=26.0, hold=45.0, warp=False):
        from AppKit import (NSBackingStoreBuffered, NSColor, NSStatusWindowLevel, NSWindow,
                            NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorFullScreenAuxiliary,
                            NSWindowCollectionBehaviorStationary, NSWindowStyleMaskBorderless)
        self.hold, self.size = hold, int(radius * 3.4)
        size = self.size
        self.win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0, 0), (size, size)), NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
        self.win.setLevel_(NSStatusWindowLevel)
        self.win.setOpaque_(False)
        self.win.setHasShadow_(False)
        self.win.setBackgroundColor_(NSColor.clearColor())
        self.win.setIgnoresMouseEvents_(True)
        self.win.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces |
                                        NSWindowCollectionBehaviorFullScreenAuxiliary |
                                        NSWindowCollectionBehaviorStationary)
        self.view = _lens_view_class().alloc().initWithFrame_(((0, 0), (size, size)))
        self.view.radius, self.view.alpha, self.view.ripples = radius, 0.0, []
        self.win.setContentView_(self.view)
        self.warper, self.warp_layer, self.radius = None, None, radius
        if warp:  # the bent screen patch sits under the drawn edge, clipped to a circle a bit wider than the bend
            from Quartz import CALayer
            self.view.setWantsLayer_(True)
            wl = CALayer.layer()
            d = radius * 2.3
            wl.setFrame_(((size / 2 - d / 2, size / 2 - d / 2), (d, d)))
            wl.setCornerRadius_(d / 2)
            wl.setMasksToBounds_(True)
            wl.setContentsScale_(self.win.backingScaleFactor())
            wl.setActions_({"contents": None, "position": None, "bounds": None})  # (no implicit animation lag)
            self.view.layer().insertSublayer_atIndex_(wl, 0)
            self.warp_layer, self.warp_d = wl, d
            self.warper = Warper(size, radius * 1.1)
        self.pos, self.goal, self.seen, self.alpha, self.shown = None, None, 0.0, 0.0, False
        self.ripples = []  # start times

    def update(self, t, point):
        """Each frame: point = the gaze (x, y) in screen points (y down from the main screen's top), or None."""
        from AppKit import NSScreen
        if point is not None:
            self.seen = t
            if self.goal is None or math.hypot(point[0] - self.goal[0], point[1] - self.goal[1]) > self.hold:
                self.goal = point  # gaze moved on: go there, and ripple when it arrives
                self.ripples.append(t + 0.18)
            else:  # small wobble: settle slowly towards it, so it doesn't jitter while you read
                self.goal = (self.goal[0] + 0.03 * (point[0] - self.goal[0]), self.goal[1] + 0.03 * (point[1] - self.goal[1]))
        if self.goal is None:
            return
        self.pos = self.goal if self.pos is None else (self.pos[0] + 0.22 * (self.goal[0] - self.pos[0]),
                                                       self.pos[1] + 0.22 * (self.goal[1] - self.pos[1]))
        target = 1.0 if t - self.seen < 0.6 else 0.0  # (blinks don't make it flicker)
        self.alpha += 0.15 * (target - self.alpha)
        self.ripples = [s for s in self.ripples if t - s < self.RIPPLE_S][-2:]
        if self.alpha < 0.02:
            if self.shown:
                self.win.orderOut_(None)
                self.shown = False
            return
        main_h = NSScreen.screens()[0].frame().size.height
        self.win.setFrameOrigin_((self.pos[0] - self.size / 2, main_h - self.pos[1] - self.size / 2))
        ripples = [(t - s) / self.RIPPLE_S for s in self.ripples if t >= s]
        if self.warper is not None:
            d = self.warp_d
            self.warper.rect = (self.pos[0] - d / 2, self.pos[1] - d / 2, d, d)
            self.warper.win_id = self.win.windowNumber() if self.shown else 0
            pulse = max([math.sin(math.pi * min(1.0, a * 1.6)) * (1 - a) for a in ripples] + [0.0])
            self.warper.scale = (0.10 + 0.12 * pulse) * self.alpha  # a gentle bulge; a little more as it lands
            if self.warper.image is not None:
                self.warp_layer.setContents_(self.warper.image)
                self.warp_layer.setOpacity_(min(1.0, self.alpha * 1.2))
        if ripples or self.view.ripples or abs(self.view.alpha - self.alpha) > 0.03:
            self.view.alpha, self.view.ripples = self.alpha, ripples
            self.view.setNeedsDisplay_(True)
        if not self.shown:
            self.win.orderFrontRegardless()
            self.shown = True

    def close(self):
        self.win.orderOut_(None)
        self.shown, self.goal, self.pos = False, None, None
        if self.warper is not None:
            self.warper.stop()
