"""Head motion for silent chat: AirPods (via the "Silent Chat Motion" helper app) or the camera's head pose.

Used for nod-to-confirm / shake-to-cancel, and as the head part of gaze. With AirPods in, their motion sensors are
used (precise, fast). Otherwise the camera's head pose (MediaPipe) is used.
"""
import collections
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

HELPER = Path.home() / "Applications" / "Silent Chat Motion.app"


class AirPodsMotion:
    """Reads the helper's JSON lines from a named pipe. .latest() -> (t, pitch, yaw, roll) in degrees, or None."""

    def __init__(self, fifo):
        self.fifo, self.samples, self.status = Path(fifo), collections.deque(maxlen=400), "starting"
        self.on_sample = None  # callback(t, pitch, yaw)
        if not HELPER.exists():
            self.status = "helper not installed"
            return
        if not self.fifo.exists():
            os.mkfifo(self.fifo, 0o600)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            subprocess.run(["open", "-g", "-a", str(HELPER), "--args", "--fifo", str(self.fifo)], capture_output=True)
            try:
                with open(self.fifo) as f:  # waits for the helper to connect
                    for line in f:
                        self._line(line)
            except OSError:
                pass
            self.status = "helper stopped"
            time.sleep(5)  # (relaunch it)

    def _line(self, line):
        try:
            d = json.loads(line)
        except ValueError:
            return
        if "status" in d:
            self.status = d["status"] if d["status"] != "started" else (
                "waiting for AirPods" if d.get("available") else "no headphone motion on this Mac")
            if d["status"] == "error":
                self.status = f"error: {d.get('message', '')}"
            return
        if "pitch" in d:
            t = time.monotonic()
            p, y, r = (math.degrees(d[k]) for k in ("pitch", "yaw", "roll"))
            self.samples.append((t, p, y, r))
            self.status = "AirPods"
            if self.on_sample:
                self.on_sample(t, p, y)

    def active(self):
        return bool(self.samples) and time.monotonic() - self.samples[-1][0] < 0.5

    def helper_ok(self):
        """The helper is running and this Mac has headphone motion (so active() means "AirPods in your ears")."""
        return self.status in ("AirPods", "waiting for AirPods")

    def latest(self):
        return self.samples[-1] if self.active() else None


def _legs(x, h):
    """How many monotonic swings of at least h the signal makes (a nod down-and-up = 2, a shake = 3+)."""
    legs, up, lo, hi = 0, None, x[0], x[0]
    for v in x[1:]:
        if up is None:
            if v - lo >= h:
                up, hi, legs = True, v, 1
            elif hi - v >= h:
                up, lo, legs = False, v, 1
            lo, hi = min(lo, v), max(hi, v)
        elif up:
            if v > hi:
                hi = v
            elif hi - v >= h:
                up, lo, legs = False, v, legs + 1
        else:
            if v < lo:
                lo = v
            elif v - lo >= h:
                up, hi, legs = True, v, legs + 1
    return legs


class NodShake:
    """Nod (yes) / shake (no) from (t, pitch, yaw) degrees. Camera head pose is noisier than AirPods, so it needs
    bigger movements."""

    def __init__(self, nod=6.0, shake=8.0, window=1.6, smooth=1):
        self.nod, self.shake, self.window, self.smooth = nod, shake, window, smooth
        self.buf = collections.deque()

    @classmethod
    def for_camera(cls):
        return cls(nod=9.0, shake=12.0, smooth=5)  # noisier, so smoothed and bigger movements only

    def add(self, t, pitch, yaw):
        self.buf.append((t, pitch, yaw))
        while self.buf and t - self.buf[0][0] > self.window:
            self.buf.popleft()

    def clear(self):
        self.buf.clear()

    def _smoothed(self, x):
        k = self.smooth
        if k <= 1 or len(x) < k:
            return x
        return [sum(x[max(0, i - k + 1):i + 1]) / len(x[max(0, i - k + 1):i + 1]) for i in range(len(x))]

    def detect(self):
        """'nod', 'shake' or None (and clears itself after a detection)."""
        if len(self.buf) < 8:
            return None
        p = self._smoothed([s[1] for s in self.buf])
        y = self._smoothed([s[2] for s in self.buf])
        pr, yr = max(p) - min(p), max(y) - min(y)
        result = None
        if pr >= self.nod and yr < 0.7 * pr and _legs(p, self.nod * 0.6) >= 2:
            result = "nod"
        elif yr >= self.shake and pr < 0.7 * yr and _legs(y, self.shake * 0.5) >= 3:
            result = "shake"
        if result:
            self.buf.clear()
        return result
