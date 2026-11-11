"""`/gaze check`: measure how far off gaze really is, on every screen, without changing anything.

Nine dots per screen (a 3x3 grid), one screen after another, ~1.7 s each (~35 s for two screens). While you look
at each dot it keeps the camera frames (JPEG, on this Mac only), the features, every screen model's prediction
and how bright and contrasty your eyes are in the picture. Nothing is learned from it.

eval_gaze_check.py replays the saved frames to compare things offline (e.g. brightening the picture before the
face tracker in the dark) against the same ground truth. Delete data/silent_chat/gaze_check/ to throw them away.
"""
import json
import queue
import threading
import time

import cv2
import numpy as np

import gaze

OUT = gaze.DATA / "gaze_check"
SHOW, SETTLE = 1.8, 0.7
GRID = [(u, v) for v in (0.12, 0.5, 0.88) for u in (0.08, 0.5, 0.92)]
EYES = [33, 133, 159, 145, 263, 362, 386, 374, 468, 473]


def eye_light(frame_bgr, landmarks):
    """(mean, contrast) of the grey levels around both eyes (0-255), or (None, None)."""
    if landmarks is None:
        return None, None
    P = np.asarray(landmarks)[EYES, :2]
    (x0, y0), (x1, y1) = P.min(0), P.max(0)
    pad = 0.25 * (x1 - x0)
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = (int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(w, x1 + pad)), int(min(h, y1 + pad)))
    if x1 - x0 < 8 or y1 - y0 < 4:
        return None, None
    g = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float(g.mean()), float(g.std())


class GazeCheck:
    def __init__(self, gazer):
        self.gazer = gazer
        mouse = gaze.screen_under_mouse()
        self.screens = [mouse] + [s for s in gaze.screens() if s[3] != mouse[3]]
        self.points = [(s, s[1][0] + u * s[1][2], s[1][1] + v * s[1][3]) for s in self.screens for u, v in GRID]
        self.dir = OUT / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.i, self.t0, self.intro, self.layer, self.rows, self.n = 0, None, None, None, [], 0
        self.q = queue.Queue(maxsize=120)
        threading.Thread(target=self._writer, daemon=True).start()
        self.done = None  # the summary, when finished

    def _writer(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            path, frame = item
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

    def _layer_for(self, scr):
        if self.layer is None or self.layer.screen[3] != scr[3]:
            if self.layer is not None:
                self.layer.close()
            self.layer = gaze.ScreenLayer(scr)
        return self.layer

    def step(self, now, frame, landmarks):
        """Each camera frame. False when finished (self.done has the summary)."""
        if self.intro is None:
            self.intro = now
        if self.i >= len(self.points):
            return False
        scr, x, y = self.points[self.i]
        layer = self._layer_for(scr)
        if now - self.intro < 2.5:
            layer.show(dim=0.85, text="Gaze check: look at each dot until it moves. Nothing is learned.")
            return True
        if self.t0 is None:
            self.t0 = now
        el = now - self.t0
        lx, ly = layer.to_local(x, y)
        layer.show(dim=0.85, dots=[(lx, ly, 14 - 7 * min(1.0, el / SHOW), "target")])
        f = self.gazer.last_feats
        if el >= SETTLE and f is not None and frame is not None:
            preds = {k: [float(v) for v in c.predict(f)[0]] for k, c in self.gazer.cals.items()}
            mean, contrast = eye_light(frame, landmarks)
            name = f"{self.n:05d}.jpg"
            try:
                self.q.put_nowait((self.dir / name, frame.copy()))
            except queue.Full:
                name = None
            self.rows.append({"i": self.i, "display": int(scr[3]), "x": x, "y": y, "t": now, "f": [float(v) for v in f],
                              "preds": preds, "live": list(self.gazer.hist[-1][1:]) if self.gazer.hist else None,
                              "light": mean, "contrast": contrast, "frame": name})
            self.n += 1
        if el >= SHOW:
            self.i, self.t0 = self.i + 1, None
            if self.i >= len(self.points):
                self.finish()
                return False
        return True

    def finish(self):
        if self.layer is not None:
            self.layer.close()
        self.q.put(None)
        (self.dir / "check.json").write_text(json.dumps({"date": time.strftime("%Y-%m-%d %H:%M"), "rows": self.rows,
                                                         "screens": [[list(s[1]), s[2], s[3]] for s in self.screens]}))
        self.done = summary(self.rows)

    def cancel(self):
        if self.layer is not None:
            self.layer.close()
        self.q.put(None)


def summary(rows):
    """Per screen: the median error of each dot's median prediction (own screen's model, and the live point)."""
    out = []
    for d in sorted({r["display"] for r in rows}):
        rs = [r for r in rows if r["display"] == d]
        own, live = [], []
        for i in sorted({r["i"] for r in rs}):
            dr = [r for r in rs if r["i"] == i]
            t = np.array([dr[0]["x"], dr[0]["y"]])
            p = [r["preds"].get(str(d)) for r in dr if r["preds"].get(str(d))]
            if p:
                own.append(float(np.linalg.norm(np.median(np.array(p), 0) - t)))
            lv = [r["live"] for r in dr if r["live"]]
            if lv:
                live.append(float(np.linalg.norm(np.median(np.array(lv), 0) - t)))
        light = np.median([r["light"] for r in rs if r["light"] is not None] or [0])
        out.append(f"screen {d}: its model {np.median(own) if own else float('nan'):.0f} pt, live point "
                   f"{np.median(live) if live else float('nan'):.0f} pt (median over {len(own)} dots); "
                   f"eye brightness {light:.0f}/255")
    return "; ".join(out)
