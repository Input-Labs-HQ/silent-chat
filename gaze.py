"""Where you're looking on screen: the camera's iris and head pose, steadied by AirPods head motion. It calibrates
itself while you work.

Webcam eye tracking is coarse (a couple of centimetres), so silent chat doesn't click where it thinks you look: it
uses the gaze point to pick the nearest button / link / field (snapping), and tells Claude what you were looking at
when you said "this".

There is no calibration step. A geometric model (below) turns the eyes and head into a point on the screen; what
it has to find out about you is where your "straight ahead" lands, and that it learns from the moments it can guess
where you're looking: you click something, the cursor comes to rest, you type at the caret, something pops up and
your eyes jump to it, your eyes follow what you drag (gaze_passive.py finds those moments). Every one is a noisy
hint (in about half of clicks your eyes are somewhere else), so each kind is weighted by how reliable it has
proved, far-off ones are ignored, and what it learned is only kept while it predicts your *next* clicks better than
the model it would replace. `/gaze calibrate` is still there as "Recalibrate": 20 dots, then 9 fresh ones to
measure the error. It's optional, and the one thing hints can't do: measure how far *your* eyes move.
eval_gaze.py measures all of this on the saved clicks and dots.

Features per camera frame: each iris's position inside its eye (corner-to-corner axis), eyelid opening (vertical
gaze moves the lids), head yaw / pitch / roll and head position. Head direction: the camera's pose is absolute but
jittery; AirPods' is precise but drifts. They're fused: slow part from the camera, fast part from the AirPods.
"""
import collections
import json
import math
import os
import random
import threading
import time
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data" / "silent_chat"
SAMPLES = DATA / "gaze_samples.npz"  # the last explicit calibration's frames
CLICKS = DATA / "gaze_clicks.npz"    # (older versions: clicks only; read once, to carry them over)
PASSIVE = DATA / "gaze_passive.npz"  # every hint it learned from: stays on this Mac
R_EYE = dict(outer=33, inner=133, upper=159, lower=145, iris=468)
L_EYE = dict(outer=263, inner=362, upper=386, lower=374, iris=473)
NAMES = ["rh", "rv", "ro", "lh", "lv", "lo", "yaw", "pitch", "tx", "ty", "tz", "roll"]


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
    # roll: how far the line through both eyes is tilted in the picture. The iris offsets above are measured along
    # each eye's own axis, so when your head is tilted they have to be turned back before they mean left / down.
    across = (P[L_EYE["outer"], :2] + P[L_EYE["inner"], :2]) / 2 - (P[R_EYE["outer"], :2] + P[R_EYE["inner"], :2]) / 2
    roll = math.degrees(math.atan2(across[1], across[0]))
    return np.array([rh, rv, ro, lh, lv, lo, head[0], head[1], t[0], t[1], t[2], roll], float)


# ----------------------------------------------------------------------------- the model
#
# Where the gaze meets the screen, in cm from the camera (x towards the picture's right, y down), head upright:
#     x = c0 + c1*tx + D*(c2*yaw + c3*pitch + A)
#     y = d0 + d1*ty + D*(d2*pitch + d3*yaw + B)
# D is the head's distance, (tx, ty) its position, and A / B how far the eyes are turned sideways / down inside the
# head: each a weighted sum of the iris offsets and lid openings, plus a constant. Everything the eyes and head
# contribute is an angle, so it's multiplied by the distance: lean back and the same eye movement covers more
# screen. 22 weights: 4 offsets (c0, d0 and the constants in A and B: where "straight ahead" is) and 18 gains.
# With the head tilted (roll), both brackets are worked out in the head's own frame and turned back by the roll,
# so turning the camera's picture turns the answer with it (checked on real frames turned +-15 degrees).
#
# The two kinds are learned differently, and that is the point of this design (eval_gaze.py has the numbers):
#   offsets  change whenever you shift in your chair, tilt the screen or the light changes, and a few hints pin
#            them down, so every hint moves them a little: a Kalman filter over the four of them (recursive least
#            squares that starts from a prior and lets them drift).
#   gains    only change with your eyes and the camera. Clicks can't teach them: in about half of clicks your
#            eyes are somewhere else, which pulls every gain towards zero (the learned gaze range shrank by ~30 %
#            vertically, and its error on clean calibration dots doubled). So gains come from the starting
#            weights below, and from Recalibrate, whose dots are clean.

CENTRE = (-0.055, -0.07, 0.31)  # typical resting (both irises' inward offset, vertical iris offset, lid opening)
N = 22
OFFSETS = [0, 4, 11, 15]  # c0, A's constant, d0, B's constant
HEAD = [1, 2, 3, 12, 13, 14]
MINOR = [6, 7, 8, 9, 10, 18, 19, 20, 21]  # the eyes' smaller terms (differences between the eyes, cross-talk)


def _rows(X):
    """Feature rows (n, 11 or 12) -> H (n, 2, 22): per frame, the two rows that multiply the weights."""
    X = np.atleast_2d(np.asarray(X, float))
    rh, rv, ro, lh, lv, lo, yaw, pitch, tx, ty, tz = X[:, :11].T
    roll = np.radians(X[:, 11]) if X.shape[1] > 11 else np.zeros(len(X))  # (older saved rows have no roll)
    D = -tz
    eh, vg, ev = (rh - lh) / 2, (rh + lh) / 2 - CENTRE[0], (rv - lv) / 2 - CENTRE[1]  # both eyes together
    op, vd, od = (ro + lo) / 2 - CENTRE[2], (rv + lv) / 2, (ro - lo) / 2              # ... and their differences
    c, s = np.cos(roll), np.sin(roll)
    # the head's turn as the head itself would call it (sideways, nodding): its two gains differ a lot, because a
    # nod changes how the eyes look to the camera far more than a turn does, and that happens in the head's frame
    ps, ph = np.radians(yaw), np.radians(pitch)
    ps, ph = D * (c * ps + s * ph), D * (c * ph - s * ps)
    one, zero = np.ones(len(X)), np.zeros(len(X))
    A = D * np.stack([one, eh, vg, ev, op, vd, od])
    B = D * np.stack([one, ev, op, eh, vg, vd, od])
    Hx = np.stack([one, tx, c * ps, c * ph, *(c * A), zero, zero, -s * ph, -s * ps, *(-s * B)], 1)
    Hy = np.stack([zero, zero, s * ps, s * ph, *(s * A), one, ty, c * ph, c * ps, *(c * B)], 1)
    return np.stack([Hx, Hy], 1)


# The starting weights, measured on one person with a MacBook Pro's camera (eval_gaze.py --prior): the eye gains
# from clean calibration dots, the head terms from clicks made in many postures. The head terms belong to the
# camera and MediaPipe more than to the person (MediaPipe assumes a wider lens than a Mac has, so it puts the head
# too close and sees a turn where there's only a sideways move: that's why 1 cm of head movement counts for less
# than 1 cm). The eye gains are personal to some degree: Recalibrate measures yours.
#                       c0    tx    yaw   pitch | A: 1     eh    vg    ev    op    vd    od
PRIOR_MEAN = np.array([-0.33, 0.40, 1.20, -0.10,   -0.02, 4.15, 0.15, 0.02, 0.16, 0.06, -0.25,
                       #  d0    ty   pitch  yaw  | B: 1     ev    op    eh    vg    vd    od
                       0.45, -0.50, 0.40, 0.40,    0.06, 6.63, -1.90, -0.17, 0.19, 0.07, 0.09])
OFFSET_SD = np.array([1.5, 0.06, 1.5, 0.15])        # how unsure a new profile is of each offset (cm, rad, cm, rad)
OFFSET_DRIFT = np.array([0.02, 0.0025, 0.02, 0.0025])  # how far each may wander per ~20 s: little. On clean dots
# slow beats fast (56 pt at this rate, 81 at 3x, 102 at 6x): averaging over many noisy clicks matters more than
# chasing the latest ones, and with the head terms right the offsets hardly move from day to day.
# For a calibration: how far each weight may move from the starting weights (the head barely moves during one,
# so the head terms stay; the minor eye terms are held close, or 20 dots overfit them and it shows in other
# postures: 59 -> 52 pt on the same sitting's dots but 94 -> 117 pt on them from offsets learned elsewhere).
CALIBRATION_SD = np.array([3.0, 1e-4, 1e-4, 1e-4,  0.12, 2.4, 0.25, 0.25, 0.25, 0.25, 0.25,
                           3.0, 1e-4, 1e-4, 1e-4,  0.30, 5.0, 2.4, 0.25, 0.25, 0.25, 0.25])


class Filter:
    """The weights, and how sure it is of the four offsets (a Kalman filter over them)."""

    def __init__(self, w=None, P=None, wide=1.0):
        self.w = PRIOR_MEAN.copy() if w is None else np.asarray(w, float)
        self.cap = (OFFSET_SD * wide) ** 2  # (wide: a screen the camera isn't built into, offsets anyone's guess)
        self.P = np.diag(self.cap) if P is None else np.asarray(P, float)

    def copy(self):
        f = Filter(self.w.copy(), self.P.copy())
        f.cap = self.cap.copy()
        return f

    def drift(self, steps=1.0):
        """Time passed: be a little less sure of the offsets (never less sure than a new profile)."""
        d = self.P.diagonal()
        self.P[np.diag_indices(4)] = d + np.minimum(OFFSET_DRIFT ** 2 * steps, np.maximum(0.0, self.cap - d))

    def reopen(self):
        """Forget where "straight ahead" was: after it has been wrong about click after click."""
        self.P = np.diag(self.cap)

    def distance(self, H, z, R):
        """How surprising a hint is: 0.83 is typical when everything is as believed, 3 is far out."""
        Ho = H[:, OFFSETS]
        nu = z - H @ self.w
        S = Ho @ self.P @ Ho.T + R
        return float(np.sqrt(nu @ np.linalg.solve(S, nu) / 2)), nu, S

    def update(self, H, z, R, gate=2.5, soft=1.0):
        """Learn from one hint (H: its two rows, z: the point in cm, R: how far off such hints usually are): it
        moves the offsets. Surprising hints count less, very surprising ones not at all. -> (surprise, used)."""
        r, nu, S = self.distance(H, z, R)
        if r > gate:
            return r, False
        Ho = H[:, OFFSETS]
        if r > soft:
            S = Ho @ self.P @ Ho.T + R * (r / soft) ** 2
        K = np.linalg.solve(S, Ho @ self.P).T
        self.w[OFFSETS] = self.w[OFFSETS] + K @ nu
        P = self.P - K @ Ho @ self.P
        self.P = (P + P.T) / 2
        return r, True

    def to_dict(self):
        return {"w": [round(float(v), 6) for v in self.w], "P": np.round(self.P, 9).tolist(),
                "cap": [round(float(v), 6) for v in self.cap]}

    @classmethod
    def from_dict(cls, d):
        f = cls(d["w"], d["P"])
        f.cap = np.asarray(d.get("cap", OFFSET_SD ** 2), float)
        return f


def fit_gains(H, z, mean=None, sd=CALIBRATION_SD, noise=0.7):
    """All 22 weights from clean looks (calibration dots): least squares pulled towards the starting weights.
    H: (n, 2, 22) rows, z: (n, 2) points in cm, noise: how far off a look at a dot is (cm)."""
    H, z = H.reshape(-1, N), z.reshape(-1)
    L = np.diag(1.0 / np.asarray(sd) ** 2)
    return np.linalg.solve(H.T @ H / noise ** 2 + L, H.T @ z / noise ** 2 + L @ (PRIOR_MEAN if mean is None else mean))


SOURCES = {  # kind of hint: how far (pt) the point it names usually is from where you were really looking
    "dot": 35.0,       # a calibration dot you were asked to look at
    "click": 90.0,     # something you clicked (your eyes rested on it as the cursor closed in)
    "pursuit": 120.0,  # your eyes following the cursor while you drag or steer it
    "appear": 150.0,   # something popped up and your eyes jumped to it (a notification, the island, a dialog)
    "land": 170.0,     # the cursor came to rest with your eyes near it, no click
    "caret": 190.0,    # typing, eyes near the text caret
    "centre": 300.0,   # (a new profile only) while you're using the Mac your gaze is somewhere around mid-screen
}
KINDS = list(SOURCES)
TRUSTED = {"dot", "click"}  # the cautious model learns from these only; clicks are also the judge
WINDOW, EVERY = 30, 10      # judged on the last 30 clicks, every 10 clicks


class SelfCalibration:
    """One screen's gaze model. Every hint moves its offsets, and it guards itself against learning the wrong
    thing:

    fast    learns from every kind of hint
    safe    learns from clicks and calibration dots only
    anchor  a frozen copy of the last model that earned its place

    Each click is first predicted by all three (so the errors are on clicks none of them has learned from), then
    learned. Every 10 clicks the last 30 errors are compared: the better learner goes live; it replaces the anchor
    only if it beats it, and if the anchor beats both learners they are rolled back to it. `notes` collects what
    was decided (for the log)."""

    def __init__(self, geom):
        self.geom = dict(geom)
        wide = 1.0 if geom.get("builtin", True) else 4.0
        self.fast, self.safe, self.anchor = Filter(wide=wide), Filter(wide=wide), Filter(wide=wide)
        self.cap = self.fast.cap.copy()
        self.live, self.errs, self.judged, self.t, self.misses = "fast", collections.deque(maxlen=WINDOW), 0, None, 0
        self.n = {k: [0, 0] for k in KINDS}  # per kind: hints seen, hints used
        self.trust = {k: 1.0 for k in KINDS}  # per kind: its distance above, scaled by how it has really done
        self.calibrated, self.notes, self.kept = None, [], 0

    # -- geometry: screen points (y down from the main screen's top) <-> cm from the camera
    def _cm(self, p):
        k = self.geom["mm_per_pt"] / 10.0
        cam = self.geom["cam"]
        return np.array([-(p[0] - cam[0]) * k, (p[1] - cam[1]) * k])

    def _pt(self, cm):
        k = 10.0 / self.geom["mm_per_pt"]
        cam = self.geom["cam"]
        cm = np.atleast_2d(cm)
        return np.stack([cam[0] - cm[:, 0] * k, cam[1] + cm[:, 1] * k], 1)

    def model(self, name=None):
        return getattr(self, name or self.live)

    def predict(self, X, name=None):
        """Feature rows -> gaze points (n, 2) in screen points."""
        return self._pt(_rows(X) @ self.model(name).w)

    def contains(self, x, y):
        q = self.geom["screen"]
        return q[0] <= x <= q[0] + q[2] and q[1] <= y <= q[1] + q[3]

    @property
    def used(self):
        return sum(v[1] for k, v in self.n.items() if k != "centre")

    @property
    def warm(self):
        """Has it learned enough to be worth acting on? (a calibration, or a handful of hints it believed)"""
        return self.calibrated is not None or self.used >= 8

    @property
    def radius(self):
        """How far off it typically is (screen points): the snapping radius. From its error on recent clicks
        before it learned from them; about half of clicks are made with the eyes elsewhere, so it's the lower
        third of those errors that says how good the gaze is (on clean dots it measured ~50 pt when this said
        ~85)."""
        col = {"fast": 0, "safe": 1, "anchor": 2}[self.live]
        recent = [e[col] for e in self.errs]
        if len(recent) >= 8:
            r = float(np.percentile(recent, 30))
        elif self.calibrated:
            r = float(self.calibrated["look_error_pt"]) * 1.3
        else:
            r = 160.0 if self.warm else 220.0
        if self.calibrated and time.time() - self.calibrated.get("at", 0) < 600:
            r = min(r, float(self.calibrated["look_error_pt"]) * 1.3)  # (just calibrated: believe the measurement)
        return float(min(300.0, max(50.0, r)))

    def learn(self, x, y, src, t=None, sig=None):
        """One hint: features x while you were (probably) looking at screen point y. t: seconds (any clock that
        keeps running), for how much the model may have drifted since the last hint. sig: (x, y) spread in points
        when this hint knows better than its kind's usual. -> (the live model's error on it in points, before
        learning from it; whether it was believed)."""
        H, z = _rows(x)[0], self._cm(y)
        s = sig or (SOURCES[src], SOURCES[src])
        k = self.geom["mm_per_pt"] / 10.0 * self.trust[src]
        R = np.diag([(s[0] * k) ** 2, (s[1] * k) ** 2])
        y = np.asarray(y, float)
        err = float(np.linalg.norm(self._pt(H @ self.model().w)[0] - y))
        steps = 1.0 if t is None or self.t is None else min(30.0, max(0.0, (t - self.t) / 20.0))
        self.t = t if t is not None else self.t
        if src == "click":  # the judge: scored by all three before any of them learns from it
            self.errs.append([float(np.linalg.norm(self._pt(H @ m.w)[0] - y))
                              for m in (self.fast, self.safe, self.anchor)])
            self.judged += 1
        gate, soft = (1e9, 1e9) if src == "dot" else (2.5, 1.0)
        self.fast.drift(steps)
        self.safe.drift(steps)
        r, used = self.fast.update(H, z, R, gate, soft)
        if src in TRUSTED:
            self.safe.update(H, z, R, gate, soft)
        self.n[src][0] += 1
        self.n[src][1] += used
        if src not in ("dot", "centre") and self.used >= 25:  # how this kind of hint really does, slowly
            surprise = max(-1.0, min(1.5, math.log(max(r, 0.05) / 0.83)))  # (0 when it's as far off as expected)
            self.trust[src] = float(min(2.5, max(0.7, self.trust[src] * math.exp(0.04 * surprise))))
        if src == "click":
            self.misses = 0 if used else self.misses + 1
            if self.misses >= 5:  # wrong about five clicks in a row: you've moved, or it has lost the plot
                self.fast.reopen()
                self.safe.reopen()
                self.misses = 0
                self.notes.append({"reopened": True, "clicks": self.judged})
            if self.judged % EVERY == 0 and len(self.errs) >= 12:
                self._referee()
        return err, used

    def _referee(self):
        med = dict(zip(("fast", "safe", "anchor"), np.median(np.array(self.errs), 0)))
        best = "fast" if med["fast"] <= med["safe"] * 1.03 else "safe"
        note = {"clicks": self.judged, **{k + "_pt": round(float(v), 1) for k, v in med.items()}}
        if med["fast"] > med["safe"] * 1.15:  # the other kinds of hint are leading it astray: drop what they taught
            self.fast = self.safe.copy()
            for k in KINDS:
                if k not in TRUSTED:
                    self.trust[k] = min(2.5, self.trust[k] * 1.3)
            note["fast_reset"] = True
        if med[best] < med["anchor"] * 0.95:  # it earned its place
            self.anchor, self.live, self.kept = getattr(self, best).copy(), best, self.kept + 1
            note["kept"] = True
        elif med["anchor"] < med[best] * 0.87:  # learning made it worse than the copy that learned nothing
            self.fast, self.safe, self.live = self.anchor.copy(), self.anchor.copy(), "fast"
            for f in (self.fast, self.safe):
                f.drift(30.0)
            note["rolled_back"] = True
        else:
            self.live = best
        note["live"] = self.live
        self.notes.append(note)

    def calibrate(self, fit, test):
        """An explicit calibration. fit / test: per dot (its frames' features (n, 12), x, y). The gains and
        offsets are fitted to the fit dots; the test dots are predicted with that (the real error), and then
        used as well. -> result."""
        def solve(dots_):
            X = np.array([np.median(X, 0) for X, _, _ in dots_])
            return fit_gains(_rows(X), np.array([self._cm((x, y)) for _, x, y in dots_]))

        probe = Filter(solve(fit))
        per_frame, per_look = [], []
        for X, x, y in test:
            P = self._pt(_rows(X) @ probe.w)
            per_frame += list(np.linalg.norm(P - np.array([x, y]), axis=1))
            per_look.append(float(np.linalg.norm(np.median(P, 0) - np.array([x, y]))))
        w = solve(list(fit) + list(test))
        sure = np.diag((OFFSET_SD / 4) ** 2)  # (just measured: the offsets are known, until time passes)
        self.fast, self.safe, self.anchor = (Filter(w.copy(), sure.copy()) for _ in range(3))
        for f in (self.fast, self.safe, self.anchor):
            f.cap = self.cap.copy()
        self.live, self.misses = "fast", 0
        self.n["dot"] = [self.n["dot"][0] + len(fit) + len(test)] * 2
        self.calibrated = {
            "look_error_pt": float(np.median(per_look)), "look_error_mean_pt": float(np.mean(per_look)),
            "look_error_max_pt": float(np.max(per_look)), "frame_error_pt": float(np.median(per_frame)),
            "within_100pt": float(np.mean(np.array(per_look) < 100)), "looks": per_look, "n_fit": len(fit),
            "date": time.strftime("%Y-%m-%d %H:%M"), "at": time.time()}
        return {**self.calibrated, "screen": list(self.geom["screen"]), "display": self.geom.get("display"),
                "mm_per_pt": self.geom["mm_per_pt"]}

    def to_dict(self):
        return {"geom": self.geom, "fast": self.fast.to_dict(), "safe": self.safe.to_dict(),
                "anchor": self.anchor.to_dict(), "live": self.live, "errs": [[round(v, 1) for v in e] for e in self.errs],
                "judged": self.judged, "t": self.t, "n": self.n, "trust": self.trust, "calibrated": self.calibrated,
                "kept": self.kept}

    @classmethod
    def from_dict(cls, d, geom=None):
        c = cls(geom or d["geom"])
        c.fast, c.safe, c.anchor = (Filter.from_dict(d[k]) for k in ("fast", "safe", "anchor"))
        c.cap = c.fast.cap.copy()
        c.live, c.judged, c.t, c.kept = d.get("live", "fast"), d.get("judged", 0), d.get("t"), d.get("kept", 0)
        c.errs.extend(d.get("errs", []))
        c.n.update({k: list(v) for k, v in d.get("n", {}).items() if k in c.n})
        c.trust.update({k: float(v) for k, v in d.get("trust", {}).items() if k in c.trust})
        c.calibrated = d.get("calibrated")
        return c


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




def geometry(scr, every=None):
    """Where the camera is relative to a screen (one of screens()): in the notch of a built-in display, or just
    above one without a notch. A screen the camera isn't built into gets the built-in display's camera, wherever
    the Displays arrangement puts it (or, with no built-in display, a webcam sitting on top of it): only a rough
    guess, so that screen's model starts wide open and finds its own offsets."""
    import Quartz
    from AppKit import NSScreen

    def camera(s):
        ns = next((n for n in NSScreen.screens() if int(n.deviceDescription()["NSScreenNumber"]) == s[3]), None)
        top = ns.safeAreaInsets().top if ns is not None else 0.0
        return [s[1][0] + s[1][2] / 2, s[1][1] + (top / 2 if top > 0 else -6.0 / (s[2] or 0.25))]

    _, q, mm, did = scr
    mm = mm or 0.25
    builtin = bool(Quartz.CGDisplayIsBuiltin(did))
    if builtin:
        cam = camera(scr)
    else:
        b = next((s for s in (every or screens()) if Quartz.CGDisplayIsBuiltin(s[3])), None)
        cam = camera(b) if b is not None else [q[0] + q[2] / 2, q[1] - 8.0 / mm]
    return {"screen": [float(v) for v in q], "display": int(did), "mm_per_pt": float(mm),
            "cam": [float(cam[0]), float(cam[1])], "builtin": builtin}


# ----------------------------------------------------------------------------- calibration (optional)

class Calibration:
    """Shows dots one at a time on the screen under the mouse. 20 to learn from, then 9 new ones to measure the real
    error (predicted before they're learned). ~60 s. step() each camera frame; .fit / .test (or .error) when done."""

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
        self.i, self.t0, self.samples, self.intro = 0, None, [], None
        self.fit, self.test, self.error = None, None, None

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
        s = self.samples
        if len({i for i, k, *_ in s if k == "fit"}) < 12 or len({i for i, k, *_ in s if k == "test"}) < 5:
            self.error = "Not enough clear looks at the dots (face not visible, or eyes closed)."
            return
        X, Y = np.array([r[4] for r in s]), np.array([[r[2], r[3]] for r in s])
        g, test = np.array([r[0] for r in s]), np.array([r[1] == "test" for r in s])
        if save:  # (kept, so the accuracy can be measured again after the model changes: eval_gaze.py)
            try:
                np.savez(SAMPLES, X=X, Y=Y, g=g, test=test)
            except OSError:
                pass
        self.fit, self.test = dots(X[~test], Y[~test], g[~test]), dots(X[test], Y[test], g[test])

    def cancel(self):
        self.layer.close()


def dots(X, Y, g):
    """Calibration frames -> per dot (its frames' features, x, y), without blinks and glances away."""
    X, Y, g = _clean(X, Y, g)
    return [(X[g == i], float(Y[g == i][0][0]), float(Y[g == i][0][1])) for i in np.unique(g)]


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


def carry_over(geom, samples=SAMPLES, clicks=CLICKS, date=None):
    """A model from what an older version saved (its last calibration's frames, then its clicks, in that order),
    so nobody starts again from nothing. -> SelfCalibration, or None when there's nothing saved."""
    cal, found = SelfCalibration(geom), False
    try:
        z = np.load(samples)
        t = z["test"]
        cal.calibrate(dots(z["X"][~t], z["Y"][~t], z["g"][~t]), dots(z["X"][t], z["Y"][t], z["g"][t]))
        cal.calibrated.update(at=0.0, date=date or cal.calibrated["date"])  # (measured then, not now)
        found = True
    except (OSError, KeyError, ValueError, IndexError):
        pass
    try:
        z = np.load(clicks)
        for x, y in zip(z["X"], z["Y"]):
            if cal.contains(*y):
                cal.learn(x, y, "click")
                found = True
    except (OSError, KeyError, ValueError):
        pass
    cal.notes.clear()
    return cal if found else None


# ----------------------------------------------------------------------------- live

class SampleLog:
    """Every hint it learned from, kept on this Mac (never sent anywhere) so the accuracy can be measured later
    (eval_gaze.py) and the model rebuilt after a change: the features, the point, the kind of hint, when, on which
    display, how far off the model was just before learning it, and whether it believed it. Clicks also keep the
    second of features before them (gaze_traces.npz), to tune which moment before a click to trust."""

    KEEP, TRACES, LENGTH = 6000, 1200, 40

    def __init__(self, path):
        self.path, self.rows, self.traces, self.dirty = Path(path), [], [], False
        try:
            z = np.load(self.path)
            self.rows = [(x, y, int(s), float(t), int(d), float(e), bool(u)) for x, y, s, t, d, e, u in
                         zip(z["X"], z["Y"], z["src"], z["t"], z["display"], z["err"], z["used"])]
        except (OSError, KeyError, ValueError):
            pass
        try:
            z = np.load(self.path.with_name("gaze_traces.npz"))
            self.traces = list(zip(z["T"], z["t"], z["Y"]))
        except (OSError, KeyError, ValueError):
            pass

    def add(self, x, y, src, t, display, err, used, trace=None):
        x = np.asarray(x, np.float32)
        self.rows = self.rows[-self.KEEP + 1:] + [(x, np.asarray(y, np.float32), KINDS.index(src), t, display, err, used)]
        if trace is not None and len(trace):
            T = np.full((self.LENGTH, len(NAMES) + 1), np.nan, np.float32)
            tr = np.asarray(trace, np.float32)[-self.LENGTH:]
            T[:len(tr)] = tr
            self.traces = self.traces[-self.TRACES + 1:] + [(T, t, np.asarray(y, np.float32))]
        self.dirty = True

    def save(self):
        if not self.dirty or not self.rows:
            return
        try:
            r = self.rows
            np.savez(self.path, X=np.array([a[0] for a in r]), Y=np.array([a[1] for a in r]),
                     src=np.array([a[2] for a in r], np.int8), t=np.array([a[3] for a in r]),
                     display=np.array([a[4] for a in r], np.int64), err=np.array([a[5] for a in r], np.float32),
                     used=np.array([a[6] for a in r]), kinds=np.array(KINDS))
            if self.traces:
                np.savez(self.path.with_name("gaze_traces.npz"), T=np.array([a[0] for a in self.traces]),
                         t=np.array([a[1] for a in self.traces]), Y=np.array([a[2] for a in self.traces]))
            self.dirty = False
        except OSError:
            pass


class Gaze:
    """Live gaze from each camera frame. .point(t0, t1) -> median gaze (x, y) in screen points over that time.
    .learn(hint) is how it calibrates itself (the hints come from gaze_passive.Teacher)."""

    def __init__(self, path, log=PASSIVE):
        self.path = Path(path)
        self.head, self.filter = HeadFusion(), OneEuro()
        self.hist = collections.deque(maxlen=300)  # (t, x, y)
        self.last_feats, self.raw, self.calib, self.layer, self.show_until = None, None, None, None, 0.0
        self.lens, self.lens_on, self.warp_on = None, False, False  # the glass circle that follows your gaze
        self.learning = True  # Settings > "Learn gaze as I work"
        self.cals, self.cal = {}, None  # a model per display; the one for the screen you're on
        self.log = SampleLog(log) if log else None
        self.checked, self.saved, self.dirty = -99.0, time.monotonic(), False
        try:
            d = json.loads(self.path.read_text())
        except (OSError, ValueError):
            d = {}
        for key, c in (d.get("cals") or {}).items():
            try:
                self.cals[key] = SelfCalibration.from_dict(c)
            except (KeyError, ValueError, TypeError):
                pass
        self.old = d if "model" in d and not self.cals else None  # (an older version's calibration: carried over)

    @property
    def ready(self):
        return self.cal is not None

    @property
    def warm(self):
        """Learned enough to act on (peeking at the notch, telling Claude what you looked at)."""
        return self.cal is not None and self.cal.warm

    @property
    def radius(self):
        """How far off it typically is (screen points): the snapping radius."""
        return self.cal.radius if self.cal is not None else 220.0

    def _screen(self, t):
        """The model for the screen you're on (the one with the mouse), made the first time it's needed."""
        if t - self.checked < 2.0:
            return self.cal
        self.checked = t
        try:
            every = screens()
            scr = screen_under_mouse()
            key, geom = str(scr[3]), geometry(scr, every)
        except Exception:
            return self.cal
        cal = self.cals.get(key)
        if cal is None:
            cal = (carry_over(geom, date=self.old.get("date")) if self.old and self.old.get("display") == scr[3]
                   else None) or SelfCalibration(geom)
            self.cals[key], self.old, self.dirty = cal, None, True
        elif cal.geom != geom:  # (resolution or arrangement changed: the weights are in cm, so they still hold)
            cal.geom = geom
        if cal is not self.cal:
            self.cal, self.filter = cal, OneEuro()
        return cal

    def update(self, t, landmarks, matrix, blend, airpods=None):
        """airpods: (t, pitch, yaw, roll) degrees, or None."""
        self.last_feats = self.raw = None
        if landmarks is None or matrix is None:
            return self._lens(t, None)
        if (blend.get("eyeBlinkLeft", 0) + blend.get("eyeBlinkRight", 0)) / 2 > 0.45:
            return self._lens(t, None)  # (eyes closed: no gaze)
        cam = head_angles(matrix)
        ap = (airpods[2], airpods[1]) if airpods else None
        head = self.head.update(t, cam, ap)
        f = features(landmarks, matrix, head)
        self.last_feats = f
        cal = self._screen(t)
        if self.calib is not None or cal is None or f is None:
            return self._lens(t, None)
        px, py = cal.predict(f)[0]
        self.raw = (float(px), float(py))  # (as predicted: may be off the screen, when you look away)
        q = cal.geom["screen"]
        m = 0.05 * min(q[2], q[3])  # (kept on the screen, give or take a little)
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
        if self.dirty and t - self.saved > 60:
            self.save()

    def _lens(self, t, p):
        if not self.lens_on or self.calib is not None or self.cal is None:
            return self.hide_lens()
        if self.lens is None:
            self.lens = GlassLens(hold=max(40.0, 0.5 * self.radius), warp=self.warp_on and Warper.allowed())
        self.lens.update(t, p)

    def learn(self, hint):
        """A moment it can guess where you were looking: {"x": features, "y": (x, y) screen point, "src": the kind
        (SOURCES), "sig": (x, y) spread in points or None, "trace": the features leading up to it or None}.
        -> a note when what it learned earned its place (or was thrown out), else None."""
        if not self.learning or self.calib is not None:
            return None
        cal = next((c for c in self.cals.values() if c.contains(*hint["y"])), None)
        if cal is None or (hint["src"] == "centre" and cal.warm):
            return None
        err, used = cal.learn(hint["x"], hint["y"], hint["src"], t=time.time(), sig=hint.get("sig"))
        if self.log is not None:
            self.log.add(hint["x"], hint["y"], hint["src"], time.time(), cal.geom.get("display", 0), err, used,
                         hint.get("trace"))
        self.dirty = True
        return self._notes(cal)

    def _notes(self, cal):
        out = None
        for n in cal.notes:
            try:
                with open(self.path.with_name("gaze_eval.log"), "a") as fh:
                    fh.write(json.dumps({"self_calibration": {"date": time.strftime("%Y-%m-%d %H:%M"), **n}}) + "\n")
            except OSError:
                pass
            mm = cal.geom["mm_per_pt"]
            if n.get("kept"):
                e = n[n["live"] + "_pt"]
                out = (f"Gaze tuned itself from what you clicked and looked at: {n['anchor_pt']:.0f} -> {e:.0f} pt "
                       f"({e * mm / 10:.1f} cm) on clicks it hadn't seen.")
            elif n.get("rolled_back"):
                out = "Gaze: what it learned lately made it worse on your clicks, so it went back to the last good model."
        cal.notes.clear()
        return out

    def save(self):
        """The models and the hint log, to disk (also called every minute while there's something new)."""
        self.saved, self.dirty = time.monotonic(), False
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"v": 2, "date": time.strftime("%Y-%m-%d %H:%M"),
                                       "cals": {k: c.to_dict() for k, c in self.cals.items()}}))
            os.replace(tmp, self.path)
        except OSError:
            pass
        if self.log is not None:
            self.log.save()

    def reset(self):
        """Forget what it learned about the eyes in front of this screen (someone else is using the Mac, or it has
        gone wrong): back to the starting weights. The hint log is kept."""
        if self.cal is not None:
            key = next(k for k, c in self.cals.items() if c is self.cal)
            self.cal = self.cals[key] = SelfCalibration(self.cal.geom)
            self.filter, self.dirty = OneEuro(), True
            self.save()

    def status(self):
        """One line: how well it's doing and what it has learned from."""
        c = self.cal
        if c is None:
            return "waiting to see your face"
        mm = c.geom["mm_per_pt"]
        seen = ", ".join(f"{v[1]} {k}" for k, v in c.n.items() if v[1] and k != "centre")
        cal = f"calibrated {c.calibrated['date']}: {describe({**c.calibrated, 'mm_per_pt': mm})}" if c.calibrated \
            else "never calibrated (it doesn't need it)"
        return (f"typically within ~{c.radius:.0f} pt ({c.radius * mm / 10:.1f} cm); learned from {seen or 'nothing yet'}"
                f"; {c.kept} refits kept; {cal}")

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
        if self.cal is None:
            return
        scr = next((s for s in screens() if s[3] == self.cal.geom.get("display")), screens()[0])
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
        calib, self.calib = self.calib, None
        if calib.error:
            return {"error": calib.error}
        key = str(calib.screen[3])
        cal = self.cals.get(key)
        if cal is None:
            cal = self.cals[key] = SelfCalibration(geometry(calib.screen))
        r = cal.calibrate(calib.fit, calib.test)
        cal.notes.clear()
        self.cal, self.filter, self.hist = cal, OneEuro(), collections.deque(maxlen=300)
        self.save()
        try:
            with open(self.path.with_name("gaze_eval.log"), "a") as f:
                f.write(json.dumps(r) + "\n")
        except OSError:
            pass
        return r

    def cancel_calibration(self):
        if self.calib is not None:
            self.calib.cancel()
            self.calib = None


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
