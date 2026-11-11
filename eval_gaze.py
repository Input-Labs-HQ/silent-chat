"""Measure gaze accuracy offline, on the hints saved while you worked. No camera needed.

The data (data/silent_chat/, never leaves this Mac):
  gaze_clicks.npz    clicks an older version saved: the features just before each click + the click point, in order
  gaze_samples.npz   the last explicit calibration's frames: 20 dots to fit, 9 fresh ones to test
  gaze_passive.npz   every hint the self-calibrating model has learned from since (clicks, landings, the caret ...)

Everything is held out by time: a click is predicted by a model that has only seen what came before it, and only
then learned from. A click is a noisy label (in roughly half of them the eyes are 150+ pt from the click point), so
errors on clicks are an upper bound on the real error; the calibration dots are clean, but only one sitting.

Measured 2026-11-10: 170 clicks over two days and one calibration (median / 90th percentile in points; 1 cm = 50 pt):
  before   one calibration, then refits from clicks: 59 / 145 on its own 9 fresh dots in the same sitting, but
           183 / 268 on the 90 clicks made after it (221 / 354 without the refits).
  after    gains from that calibration's 20 fit dots, offsets learned from clicks: 35 / 168 on the 9 fresh dots,
           142 / 278 on the same 90 clicks, and the clean test: those 9 dots with the offsets learned from clicks
           alone, starting from nothing, at 17 moments over both days: 50 / 84 (87, 66, 54 after 10, 20, 30 clicks).
  never calibrated, with eye gains 25 % off the starting weights: 112-143 on those dots.
  What didn't work: letting clicks retune the gains too (a 22-weight Kalman filter). It looked better on clicks
  (107 on the later ones) and was twice as bad on the dots (137 instead of ~60): half of all clicks are made
  looking elsewhere, and that pulls every gain towards zero. Clicks reward a model for shrinking; dots don't.

  .venv/bin/python eval_gaze.py            before / after (also appended to data/silent_chat/gaze_eval.log)
  .venv/bin/python eval_gaze.py --prior    remake the starting weights (PRIOR_MEAN in gaze.py) from the saved hints
  .venv/bin/python eval_gaze.py --passive  the hints logged by the running app: per kind, how many and how far off
"""
import argparse
import json
import time

import numpy as np

import gaze

LOG = gaze.DATA / "gaze_eval.log"
# What plain geometry says before anyone has been measured: 1 cm of head movement moves the gaze 1 cm, the head's
# turn counts in full, and the eye turns ~2.6 rad per eye-width of iris movement (eye ~29 mm wide, iris ~11 mm
# from the eye's centre of rotation). Used to check how much the fitted starting weights owe to the test data.
TEXTBOOK = np.zeros(gaze.N)
TEXTBOOK[[1, 2, 5, 12, 13, 16]] = [1.0, 1.0, 2.6, -1.0, 1.0, 2.6]


def stats(e):
    e = np.asarray(e, float)
    return (f"{np.median(e):4.0f} / {np.percentile(e, 90):4.0f}   (n={len(e)})" if len(e) else "   - /    -   (n=0)")


def geom_of(path=gaze.DATA / "gaze.json"):
    """The screen the saved hints are from (falls back to a 14-inch MacBook Pro)."""
    g = {"screen": [0.0, 0.0, 1512.0, 982.0], "display": 1, "mm_per_pt": 0.19921568328259037, "cam": [756.0, 16.0],
         "builtin": True}
    try:
        d = json.loads(path.read_text())
        cals = list((d.get("cals") or {}).values())
        if cals:
            return next((c["geom"] for c in cals if c["geom"].get("builtin")), cals[0]["geom"])
        q, mm = d["screen"], d["mm_per_pt"]
        g.update(screen=q, display=d.get("display", 1), mm_per_pt=mm, cam=[q[0] + q[2] / 2, q[1] + 16.0])
    except (OSError, ValueError, KeyError):
        pass
    return g


def load():
    """-> (click features, click points, calibration dict or None, how many clicks came before the calibration)."""
    c = np.load(gaze.CLICKS)
    try:
        z = np.load(gaze.SAMPLES)
        cal = {k: z[k] for k in ("X", "Y", "g", "test")}
    except OSError:
        cal = None
    before = 0
    try:  # the log says how many clicks had been collected when each calibration happened
        seen = 0
        for line in LOG.read_text().splitlines():
            r = json.loads(line)
            if "click_refit" in r:
                seen = r["click_refit"]["clicks"]
            elif "look_error_pt" in r and "refit_of" not in r:
                before = seen + 5  # (between two refits: halfway)
    except (OSError, ValueError):
        pass
    before = min(len(c["X"]), max(0, before))
    return c["X"], c["Y"], cal, (before // 10 * 10 if cal is not None else 0)


# ----------------------------------------------------------------------------- before: the model this replaces

class Ridge:
    """The previous gaze model, as it shipped: ridge regression on standardised features + 5 curvature terms,
    head position effectively left out, features clipped to the calibrated range. Fitted on the calibration's
    frames; every 10 clicks refitted with the clicks and kept if better on the newest quarter of them."""

    PENALTY = np.array([0] + [1] * 6 + [1, 1] + [1e5, 1e5, 1e5] + [30] * 5, float)

    def __init__(self, X, Y, groups, lams=(0.03, 0.1, 0.3, 1, 3, 10, 30, 100)):
        X = X[:, :11]
        self.mean, self.std = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mean) / self.std
        self.lo, self.hi = Z.min(0) - 1.0, Z.max(0) + 1.0
        A = self._expand(Z)
        best = None
        for lam in lams:
            errs = []
            for g in np.unique(groups):
                tr, te = groups != g, groups == g
                errs.append(np.linalg.norm(np.median(A[te] @ self._solve(A[tr], Y[tr], lam), 0) - Y[te][0]))
            if best is None or np.median(errs) < best[0]:
                best = (float(np.median(errs)), lam)
        self.coef = self._solve(A, Y, best[1])

    @staticmethod
    def _expand(Z):
        eh, ev = (Z[:, 0] + Z[:, 3]) / 2, (Z[:, 1] + Z[:, 4]) / 2
        return np.hstack([np.ones((len(Z), 1)), Z, np.stack([eh * eh, ev * ev, eh * ev, eh * Z[:, 6], ev * Z[:, 7]], 1)])

    def _solve(self, A, Y, lam):
        return np.linalg.solve(A.T @ A + lam * np.diag(self.PENALTY), A.T @ Y)

    def predict(self, X):
        X = np.atleast_2d(X)[:, :11]
        return self._expand(np.clip((X - self.mean) / self.std, self.lo, self.hi)) @ self.coef


def before(cX, cY, cal, start):
    """The old pipeline on the clicks after its calibration -> (errors with the calibration frozen, errors with
    its click refits as it ran, its error on its own 9 fresh dots)."""
    fit = ~cal["test"]
    X, Y, g = gaze._clean(cal["X"][fit], cal["Y"][fit], cal["g"][fit])
    model = frozen = Ridge(X, Y, g)
    looks = [np.linalg.norm(np.median(frozen.predict(cal["X"][(cal["g"] == i) & cal["test"]]), 0) - cal["Y"][cal["g"] == i][0])
             for i in np.unique(cal["g"][cal["test"]])]
    errs, at = [], start
    for i in range(start, len(cX)):
        errs.append(float(np.linalg.norm(model.predict(cX[i])[0] - cY[i])))
        n = i + 1
        if n >= 20 and n - at >= 10:
            at = n
            k = max(5, n // 4)
            tr = slice(0, n - k)
            new = Ridge(np.vstack([X] + [cX[tr, :11]] * 4), np.vstack([Y] + [cY[tr]] * 4),
                        np.concatenate([g] + [1000 + np.arange(n - k)] * 4))
            te = slice(n - k, n)
            old_e = np.median(np.linalg.norm(model.predict(cX[te]) - cY[te], axis=1))
            new_e = np.median(np.linalg.norm(new.predict(cX[te]) - cY[te], axis=1))
            if new_e < old_e * 0.95:
                model = new
    return np.linalg.norm(frozen.predict(cX[start:]) - cY[start:], axis=1), np.array(errs), np.array(looks)


# ----------------------------------------------------------------------------- after: the self-calibrating model

def fresh(geom, w=None):
    """A new profile (optionally with other weights)."""
    c = gaze.SelfCalibration(geom)
    if w is not None:
        for f in (c.fast, c.safe, c.anchor):
            f.w = np.array(w, float)
    return c


def run(cal, X, Y, src="click"):
    """Feed hints in order -> each one's error just before it was learned."""
    return np.array([cal.learn(x, y, src)[0] for x, y in zip(X, Y)])


def looks(d, which=None):
    """Calibration frames -> per dot (features, point), optionally only the fit (False) or test (True) dots."""
    ids = [i for i in np.unique(d["g"]) if which is None or d["test"][d["g"] == i][0] == which]
    return (np.array([np.median(d["X"][d["g"] == i], 0) for i in ids]),
            np.array([d["Y"][d["g"] == i][0] for i in ids]))


def gains_from(geom, X, Y, mean=None):
    """Weights fitted to clean looks, as Recalibrate does."""
    c = gaze.SelfCalibration(geom)
    return gaze.fit_gains(gaze._rows(X), np.array([c._cm(y) for y in Y]), mean)


def transfer(geom, w, cX, cY, X, Y, every=10):
    """The clean test of what clicks teach: start from weights w, learn offsets from the clicks in order, and
    after every 10 clicks predict the looks (X, Y) that none of it has seen. -> (click errors, per moment the
    median error on the looks)."""
    c, errs, moments = fresh(geom, w), [], []
    for i, (x, y) in enumerate(zip(cX, cY)):
        errs.append(c.learn(x, y, "click")[0])
        if (i + 1) % every == 0:
            moments.append(float(np.median(np.linalg.norm(c.predict(X) - Y, axis=1))))
    return np.array(errs), np.array(moments)


def robust_fit(geom, X, Y, weight=None, iters=40, scale=1.5):
    """All 22 weights from noisy hints, robustly (far-off hints count little): least squares pulled, loosely,
    towards plain geometry. Good for the head terms (clicks cover many postures); it shrinks the eye gains."""
    c = gaze.SelfCalibration(geom)
    H = gaze._rows(X).reshape(-1, gaze.N)
    z = np.array([c._cm(y) for y in Y]).reshape(-1)
    u0 = np.repeat(np.ones(len(X)) if weight is None else weight, 2)
    sd = np.array([3.0, 0.6, 0.8, 0.3, 0.12, 2.4, 2.0, 1.2, 0.8, 2.0, 1.2,
                   3.0, 0.6, 0.8, 0.4, 0.3, 5.0, 2.4, 1.0, 1.6, 2.4, 1.2])
    L = np.diag(1.0 / sd ** 2)
    w = TEXTBOOK.copy()
    for _ in range(iters):
        u = u0 / (1 + ((z - H @ w) / scale / 1.5) ** 2)
        w = np.linalg.solve((H * u[:, None]).T @ H / scale ** 2 + L, (H * u[:, None]).T @ z / scale ** 2 + L @ TEXTBOOK)
    return w


def starting_weights(geom, cX, cY, cal):
    """How PRIOR_MEAN in gaze.py was made: head terms from a robust fit to everything (clicks span many postures),
    then the eye gains and offsets from the clean calibration dots with those head terms held."""
    dX, dY = looks(cal)
    pad = lambda A: np.pad(A, ((0, 0), (0, len(gaze.NAMES) - A.shape[1])))
    w = robust_fit(geom, np.vstack([pad(cX), pad(dX)]), np.vstack([cY, dY]),
                   np.r_[np.ones(len(cX)), 3 * np.ones(len(dX))])  # (a dot is a clean label: it counts for 3)
    mean = TEXTBOOK.copy()
    mean[gaze.HEAD] = np.round(w[gaze.HEAD], 1)
    mean[17] = 0.0
    return gains_from(geom, dX, dY, mean)


def paired(a, b, block=10, n=2000, seed=0):
    """How often median(b) < median(a) when the clicks are resampled in blocks (clicks come in bursts)."""
    rng = np.random.default_rng(seed)
    blocks = [np.arange(i, min(i + block, len(a))) for i in range(0, len(a), block)]
    wins = 0
    for _ in range(n):
        idx = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        wins += np.median(b[idx]) < np.median(a[idx])
    return wins / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior", action="store_true", help="print starting weights fitted to all saved hints")
    ap.add_argument("--passive", action="store_true", help="summarise the hints the running app has logged")
    args = ap.parse_args()
    geom = geom_of()
    if args.passive:
        return passive(geom)
    cX, cY, cal, start = load()
    if args.prior:
        w = starting_weights(geom, cX, cY, cal)
        np.set_printoptions(precision=2, suppress=True, linewidth=150)
        print(f"PRIOR_MEAN, from {len(cX)} clicks (head terms) and {len(np.unique(cal['g']))} calibration dots "
              "(eye gains)")
        print("  x: c0, tx, yaw, pitch | A: 1, eh, vg, ev, op, vd, od\n   ", w[:11])
        print("  y: d0, ty, pitch, yaw | B: 1, ev, op, eh, vg, vd, od\n   ", w[11:])
        return
    out, mm = {}, geom["mm_per_pt"]

    def row(key, label, e):
        out[key] = {"median_pt": round(float(np.median(e)), 1), "p90_pt": round(float(np.percentile(e, 90)), 1),
                    "n": len(e)}
        print(f"  {label:74s} {stats(e)}")

    print(f"Gaze error in screen points, median / 90th percentile (1 cm = {10 / mm:.0f} pt on this screen). "
          f"{len(cX)} saved clicks, about {start} of them before the calibration.\n")
    frozen, ran, old_looks = before(cX, cY, cal, start)
    print("BEFORE: one explicit calibration, then refits from clicks")
    row("before_dots", "its 9 fresh dots, same sitting", old_looks)
    row("before_frozen", f"the {len(frozen)} clicks after it, calibration alone", frozen)
    row("before", f"the {len(ran)} clicks after it, with its click refits (as it ran)", ran)

    fX, fY = looks(cal, False)
    tX, tY = looks(cal, True)
    w = gains_from(geom, fX, fY)  # (what Recalibrate would have set from the 20 fit dots)
    print("\nAFTER: gains from one calibration, offsets learned from every click after predicting it")
    row("after_dots", "its 9 fresh dots, same sitting", np.linalg.norm(fresh(geom, w).predict(tX) - tY, axis=1))
    c = fresh(geom, w)
    run(c, cX[:start], cY[:start])
    e = run(c, cX[start:], cY[start:])
    row("after", f"the same {len(e)} clicks as above", e)
    print(f"      better than before in {paired(ran, e):.0%} of block-resamples of those clicks")
    w0 = w.copy()
    w0[gaze.OFFSETS] = 0.0  # (forget the offsets that sitting had: they must come from the clicks alone)
    e, moments = transfer(geom, w0, cX, cY, tX, tY)
    row("after_all", f"all {len(e)} clicks, over both days", e)
    row("after_transfer", f"the 9 fresh dots again, offsets from clicks only, at {len(moments)} moments over both days",
        moments)
    print("      (the clean test: a click is a noisy label, a dot isn't. Each moment is the median over the 9 dots,")
    print("       after every 10 clicks; after the first 10 / 20 / 30 clicks: "
          f"{moments[0]:.0f} / {moments[1]:.0f} / {moments[2]:.0f} pt)")

    print("\nAFTER, never calibrated: the starting weights as shipped (they were measured on these same dots, so the")
    print("dots can't test them; a stranger's eyes are stood in for by gains 25 % off)")
    e = run(fresh(geom), cX, cY)
    row("fresh_all", f"a new profile, all {len(e)} clicks", e)
    row("fresh_first20", "... its first 20 clicks", e[:20])
    row("fresh_rest", "... after the first 20", e[20:])
    eye = [i for i in range(gaze.N) if i not in gaze.OFFSETS + gaze.HEAD]
    for s in (0.75, 1.25):
        w2 = w0.copy()
        w2[eye] *= s
        e, moments = transfer(geom, w2, cX, cY, tX, tY)
        row(f"stranger_{s}", f"eye gains x{s}, never calibrated: the 9 fresh dots at {len(moments)} moments", moments)
    out["date"] = time.strftime("%Y-%m-%d %H:%M")
    try:
        with open(LOG, "a") as f:
            f.write(json.dumps({"offline_eval": out}) + "\n")
    except OSError:
        pass


def passive(geom):
    """What the running app has logged: per kind of hint, and whether the kinds beyond clicks earn their keep."""
    try:
        z = np.load(gaze.PASSIVE)
    except OSError:
        print("No hints logged yet (data/silent_chat/gaze_passive.npz): run Silent Chat for a while first.")
        return
    kinds = [str(k) for k in z["kinds"]]
    src, err, used, t = z["src"], z["err"], z["used"], z["t"]
    hours = (t.max() - t.min()) / 3600 if len(t) > 1 else 0
    print(f"{len(src)} hints over {hours:.1f} h. Error = the live model's, just before it learned from the hint.\n")
    print(f"  {'kind':10s} {'hints':>6s} {'believed':>9s}   median / p90 (pt)")
    for i, k in enumerate(kinds):
        m = src == i
        if m.any():
            print(f"  {k:10s} {m.sum():6d} {used[m].mean():9.0%}   {stats(err[m])}")
    click = src == kinds.index("click")
    if click.sum() >= 20:
        n = click.sum()
        print(f"\n  clicks, first half then second half: {stats(err[click][:n // 2])}  ->  {stats(err[click][n // 2:])}")
        try:
            w = gains_in_use()
        except (OSError, ValueError, KeyError, StopIteration):
            w = None
        both, only = fresh(geom, w), fresh(geom, w)
        e_both, e_only = [], []
        for x, y, s, tt in zip(z["X"], z["Y"], src, t):
            if not both.contains(*y):
                continue
            k = kinds[s]
            e = both.learn(x, y, k, t=float(tt))[0]
            if k in ("click", "dot"):
                e2 = only.learn(x, y, k, t=float(tt))[0]
                if k == "click":
                    e_both.append(e)
                    e_only.append(e2)
        print("\n  replayed from new offsets, error on each click before learning it:")
        print(f"    learning from every kind of hint   {stats(e_both)}")
        print(f"    learning from clicks only          {stats(e_only)}")


def gains_in_use(path=gaze.DATA / "gaze.json"):
    """The weights in use, with the offsets forgotten (so a replay keeps a calibration's gains and re-learns only
    where "straight ahead" is)."""
    d = json.loads(path.read_text())
    c = next(iter(d["cals"].values()))
    w = np.array(c[c.get("live", "fast")]["w"], float)
    w[gaze.OFFSETS] = 0.0
    return w


if __name__ == "__main__":
    main()
