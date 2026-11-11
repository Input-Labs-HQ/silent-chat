"""Before an action runs: do it now, or ask for a nod first?

Every local action (an instant command, a saved shortcut, a press of a control, what the small local model
decided) passes through `assess`. Harmless, reversible things run at once: scrolling, pausing, turning the volume
down, showing or hiding silent chat. Anything you'd notice if it went wrong asks first: playing, skipping or
turning up audio, opening apps, tabs and web searches, closing things, pressing a control, running a shortcut.

How it started matters as much as what it is:
  typed          you typed it: it runs (no reading involved)
  voice          you said it out loud (Whisper heard it): runs, except a shortcut that sends something
  mouthed        a deliberate start (holding Option, a long blink): runs when the lip-reader was clearly sure,
                 asks when its first two guesses were close
  handsfree      nobody pressed anything: anything consequential always asks

`NodPrompt` is the asking: a nod runs it, a shake, looking away or ~4 s of nothing cancels (the default). Only a
nod that starts after the question appears counts, and if your head was already bobbing (music), a nod has to be
clearly bigger than the bobbing was.

Measured on data/silent_chat/handsfree.log (Nov 8-11): 118 hands-free actions ran on their own. Of the 75 that
were instant commands (71 of them 'NEXT', mostly misreadings while a video talked), 53 would now ask for a nod,
21 no longer count as a command at all (HF_SURE in silent_chat.py: the reading has to clearly beat the next one),
and 1 (a pause) would still run at once. Of 3 deliberate mouthed local commands, 2 would have asked (both 'NEXT'
with close guesses).
"""
import collections
import json
import re
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "silent_chat"
LOG = DATA / "gate.log"
SURE = 3.0  # lip-reader: how far ahead its first guess must be of the next different one to count as sure

SAFE_SPOTIFY = {"pause"}
AUDIO_SPOTIFY = {"play", "next", "previous"}


def describe(kind, args, label=""):
    """(level, question). level: 'safe' | 'audio' | 'act'."""
    if kind == "action":  # silent chat's own switches (typing mode, hide ...)
        return "safe", ""
    if kind == "shortcut":
        return "act", f"Run your shortcut “{label}”?"
    if kind == "press":
        return "act", f"Press “{label}”?"
    a = list(args or [])
    if not a:
        return "act", "Do that?"
    if a[0] == "spotify":
        what = a[1] if len(a) > 1 else ""
        if what in SAFE_SPOTIFY:
            return "safe", ""
        if what == "volume":
            v = a[2] if len(a) > 2 else ""
            if v == "down":
                return "safe", ""
            return "audio", "Turn the music up?" if v == "up" else f"Set the volume to {v}?"
        return "audio", {"play": "Play music?", "next": "Skip to the next song?",
                         "previous": "Go back a song?"}.get(what, "Change the music?")
    if a[0] == "scroll":
        return "safe", ""
    if a[0] == "open":
        return "act", f"Open {a[1] if len(a) > 1 else 'that'}?"
    if a[0] == "url":
        q = re.search(r"[?&]q=([^&]+)", a[1] if len(a) > 1 else "")
        from urllib.parse import unquote_plus
        return "act", (f"Search the web for “{unquote_plus(q.group(1))}”?" if q else "Open that page?")
    if a[0] == "key":
        k = a[1] if len(a) > 1 else ""
        return "act", {"cmd+w": "Close this tab?", "cmd+t": "Open a new tab?", "cmd+[": "Go back?",
                       "cmd+r": "Reload the page?"}.get(k, f"Press {k}?")
    if a[0] == "press":
        return "act", f"Press “{a[1] if len(a) > 1 else 'that'}”?"
    return "act", "Do that?"


def how_started(part):
    """typed / voice / mouthed / handsfree, from a message part."""
    if part.get("how"):
        return part["how"]
    if part.get("handsfree"):
        return "handsfree"
    if part.get("spoken") or part.get("voice"):
        return "voice"
    if part.get("typed") and not part.get("hyps"):
        return "typed"
    return "mouthed"


def margin(hyps):
    """How far the lip-reader's first guess is ahead of the next guess that reads differently (log-prob)."""
    if not hyps:
        return 0.0
    top = hyps[0][0].strip().upper()
    for h, s in hyps[1:]:
        if h.strip().upper() != top:
            return float(hyps[0][1] - s)
    return 99.0


def assess(kind, args, part, label="", sends=False):
    """-> (decision, question): decision 'run' or 'ask'."""
    level, q = describe(kind, args, label)
    how = how_started(part)
    if level == "safe" or how == "typed":
        return "run", q
    if how == "voice":
        return ("ask", q) if sends else ("run", q)
    if how == "mouthed":
        return ("run", q) if margin(part.get("hyps")) >= SURE and not sends else ("ask", q)
    return "ask", q  # hands-free: never acts on anything consequential without a nod


def log(event, **kv):
    try:
        with open(LOG, "a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, **kv}) + "\n")
    except OSError:
        pass


class Motion:
    """The last few seconds of head movement (pitch, yaw in degrees), from the camera or the AirPods."""

    def __init__(self, keep=4.0):
        self.keep, self.buf = keep, collections.deque()

    def add(self, t, pitch, yaw):
        self.buf.append((t, pitch, yaw))
        while self.buf and t - self.buf[0][0] > self.keep:
            self.buf.popleft()

    def since(self, t0):
        return [s for s in self.buf if s[0] >= t0]


def _smooth(x, k):
    if k <= 1 or len(x) < k:
        return list(x)
    return [sum(x[max(0, i - k + 1):i + 1]) / len(x[max(0, i - k + 1):i + 1]) for i in range(len(x))]


def _swing(samples, k):
    """Peak-to-peak (pitch, yaw) of smoothed samples."""
    if len(samples) < 3:
        return 0.0, 0.0
    p, y = _smooth([s[1] for s in samples], k), _smooth([s[2] for s in samples], k)
    return max(p) - min(p), max(y) - min(y)


class NodPrompt:
    """One question waiting for a nod. step() each frame -> 'nod' | 'shake' | 'timeout' | 'away' | None."""

    WAIT, REACT = 4.0, 0.25  # s to answer; movement in the first REACT s doesn't count (it began before)

    def __init__(self, now, motion, camera=True):
        self.t0, self.motion, self.camera = now, motion, camera
        self.nod, self.shake, self.k = (9.0, 12.0, 5) if camera else (6.0, 8.0, 1)
        before = [s for s in motion.buf if now - 3.0 <= s[0] <= now]
        bob_p, bob_y = _swing(before, self.k)  # already moving (music, talking): answers must be clearly bigger
        self.bob = (bob_p, bob_y)
        self.nod = max(self.nod, 1.6 * bob_p)
        self.shake = max(self.shake, 1.6 * bob_y)
        self.away_since = None

    def step(self, now, face_seen=True, facing=True):
        if now - self.t0 > self.WAIT:
            return "timeout"
        if not (face_seen and facing) and self.camera:
            self.away_since = self.away_since or now
            if now - self.away_since > 1.2:
                return "away"
        else:
            self.away_since = None
        s = self.motion.since(self.t0 + self.REACT)
        if len(s) < 6:
            return None
        p, y = _smooth([v[1] for v in s], self.k), _smooth([v[2] for v in s], self.k)
        pr, yr = max(p) - min(p), max(y) - min(y)
        if pr >= self.nod and yr < 0.7 * pr and _legs(p, self.nod * 0.6) >= 2:
            return "nod"
        if yr >= self.shake and pr < 0.7 * yr and _legs(y, self.shake * 0.5) >= 3:
            return "shake"
        return None


def _legs(x, step):
    """How many times the signal changes direction by at least `step` (a nod is down and back up: 2)."""
    legs, anchor, direction = 0, x[0], 0
    for v in x[1:]:
        d = v - anchor
        if direction >= 0 and d <= -step:
            legs, direction, anchor = legs + 1, -1, v
        elif direction <= 0 and d >= step:
            legs, direction, anchor = legs + 1, 1, v
        elif (direction > 0 and v > anchor) or (direction < 0 and v < anchor):
            anchor = v
    return legs
