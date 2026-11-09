"""When silent chat shows itself, when it talks, and when it gets out of the way.

  * The panel appears only while you're using it (listening, thinking, a reply to read, a notification), then
    fades out after you've had time to read it, unless your eyes are still on it. \\ pins it open.
  * Replies are spoken only when that makes sense: never without headphones in (text instead); with them, when you
    spoke out loud, or mouthed with VOICE on; never over a video you're watching.
  * While you talk to it out loud and it answers, Spotify is turned down, and back up afterwards.
"""
import re
import threading
import time

VIDEO_TITLE = re.compile(r"youtube|netflix|twitch|prime video|disney\+|hulu|vimeo|crunchyroll|plex|max\.com|"
                         r"\bwatch\b|tiktok", re.I)
VIDEO_APPS = {"QuickTime Player", "IINA", "VLC", "TV", "Netflix", "Infuse", "Plex"}


def watching(front_line):
    """'Front: App - 'title'' (the screen summary's first line) -> True if it looks like a video in front."""
    if not front_line:
        return False
    m = re.match(r"Front: (.+?)(?: - (.*)|$| \()", front_line)
    app, title = (m.group(1), m.group(2) or "") if m else ("", front_line)
    return app in VIDEO_APPS or bool(VIDEO_TITLE.search(title))


def should_speak(headphones, spoke_aloud, voice_on, video):
    """Speak this reply? No headphones: never (it's shown instead). A video in front: never (don't talk over it).
    Otherwise yes if you spoke out loud, or if VOICE is on for mouthed / typed requests."""
    if not headphones or video:
        return False
    return spoke_aloud or voice_on


REQUEST = re.compile(r"\b(play|pause|skip|open|close|search|find|send|message|text|reply|call|turn|set|show|tell|"
                     r"read|write|type|make|start|stop|put|move|go|look|check|add|remind|translate|summari[sz]e|"
                     r"click|scroll|translat\w*|i want|i'd like|"
                     r"what|what's|how|who|when|where|why|can you|could you|please|hey|claude)\b", re.I)
FILLER = {"ok", "okay", "yes", "yeah", "no", "so", "um", "uh", "hmm", "thank you", "thanks", "right", "cool", "nice"}


def plausible_request(hyps, app_names=()):
    """A free, local check on a hands-free lip-reading: could this be a request for Claude (worth a model call),
    or is it chatter / chewing / a misfire? Looks for a request word or an app name in the top guesses, at least
    three words, and the top two guesses roughly agreeing (misfires tend to produce scattered guesses)."""
    texts = [h[0].lower().strip() for h in (hyps or [])[:3] if h and h[0]]
    if not texts:
        return False
    top = texts[0]
    words = re.findall(r"[a-z']+", top)
    if len(words) < 2 or top.strip(" .!?") in FILLER:
        return False
    names = {a.lower() for a in app_names if a}
    has_target = any(n in top for n in names)
    if not (REQUEST.search(top) or has_target):
        return False
    if len(texts) > 1:
        a, b = set(words), set(re.findall(r"[a-z']+", texts[1]))
        if len(a & b) / max(1, len(a | b)) < 0.25 and not has_target:
            return False
    return True


def reading_time(text):
    """Seconds to leave a reply up: ~4 words a second, at least 4 s, at most 25 s."""
    return min(25.0, max(4.0, 2.0 + len(text.split()) / 4.0))


class Ducker:
    """Turns Spotify down while you talk to silent chat out loud (and while it answers), then back up."""

    def __init__(self, level=0.3):
        self.level, self.saved, self.busy = level, None, False

    def _osa(self, script):
        import subprocess
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else ""

    def _bg(self, fn):
        if self.busy:
            return
        self.busy = True

        def go():
            try:
                fn()
            except Exception:
                pass
            self.busy = False
        threading.Thread(target=go, daemon=True).start()

    def duck(self):
        if self.saved is not None:
            return

        def go():
            from AppKit import NSRunningApplication
            if not NSRunningApplication.runningApplicationsWithBundleIdentifier_("com.spotify.client"):
                return
            st = self._osa('tell application "Spotify" to return (player state as string) & "|" & sound volume')
            if not st.startswith("playing|"):
                return
            vol = int(st.split("|")[1])
            self.saved = vol
            self._osa(f'tell application "Spotify" to set sound volume to {max(5, int(vol * self.level))}')
        self._bg(go)

    def restore(self):
        if self.saved is None:
            return
        vol, self.saved = self.saved, None
        self._bg(lambda: self._osa(f'tell application "Spotify" to set sound volume to {vol}'))


class Presence:
    """Decides, each frame, whether the panel should be visible."""

    def __init__(self):
        self.show_until, self.last_key = 0.0, None

    def content(self, now, key, text=""):
        """Call with something identifying what's displayed: when it changes, the panel stays up to read it."""
        if key != self.last_key:
            self.last_key = key
            if text:
                self.show_until = max(self.show_until, now + reading_time(text))

    def poke(self, now, seconds):
        self.show_until = max(self.show_until, now + seconds)

    def visible(self, now, busy, pinned, gaze_on_panel):
        """busy: listening / thinking / speaking / waiting for an answer."""
        if busy or pinned:
            return True
        if gaze_on_panel and now < self.show_until + 15:  # still reading it: stay (up to 15 s more)
            self.show_until = max(self.show_until, now + 1.5)
        return now < self.show_until
