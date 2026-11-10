#!/usr/bin/env python
"""Talk to Claude with mouth movements only (prototype).

Hold OPTION (in any app) while you silently mouth a message, then let go. Or use face gestures (FACE button):
  long blink (~0.5 s)   start listening; open your eyes and mouth; it stops after ~1 s without lip movement
  smile (TYPE mode)     delete dictated words: the first after ~0.7 s, then one per second while you smile
  eyebrow raise (TYPE)  press Enter to send (or a wink, with --send wink)

The overlay floats above every app (including full-screen ones), can be dragged and snaps to the screen edge. It
shows a topographic map of your lips (not video) so you can see the camera is tracking you, plus the conversation.
Claude's reply streams in and is spoken sentence by sentence (VOICE button: spoken or text only). Hold Option or
blink long while Claude is talking to interrupt it.

It learns from you:
  * Correct it naturally ("no, I said Nissan") or type `/fix <what you said>` in the console. The clip is relabelled,
    and the word and the misreading go into your personal memory, which Claude is given.
  * Clips you confirmed (or didn't correct) are used to fine-tune the lip-reader on your mouth. This runs
    automatically when the chat closes and enough new clips exist; see train_personal.py.

Usually it runs as the menu bar app "Silent Chat" (~/Applications, opens at login; build it with make_app.py).
Console commands (menu bar icon > Open Console, or silent_chat.command): type a message to send it as text,
`/fix <text>`, `/learn <word, word>`, `/voice`, `/face`, `/type`, `/mic`, `/hide`, `/show`, `/restart`, `/quit`.

It can also run straight from Terminal (Terminal then needs the camera etc. permissions):
  .venv/bin/python silent_chat.py --speak
  .venv/bin/python silent_chat.py --confirm --speak     # older mode: SPACE in a camera window + pick 1/2/3
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import local_commands
import improve
import island
import local_router
import presence
import privacy
from AppKit import NSEvent, NSScreen
import shortcuts
import vsr
from learning import DATA, ROOT, Labels, PersonalMemory
from llm_correct import format_nbest, run_claude

# the app macOS asks about for permissions: Silent Chat.app when started from it, else Terminal
HOST = "Silent Chat" if os.environ.get("SILENT_CHAT_APP") else "Terminal"

CALIBRATION = DATA / "activity_calibration.json"
LIPREADER_NOTES = (
    "The lip-reader only sees the lips, so it confuses words that look alike (p/b/m, f/v, t/d/n/s/z/th, k/g, "
    "ch/j/sh, similar vowels) and gets word boundaries wrong. Its decoder was trained on TED talks, so when unsure "
    "it drifts into fluent but unrelated TED-style phrases. Hypotheses come best first with log-probability scores "
    "(closer to zero = more confident). Use the words and lip shapes the hypotheses agree on, undo look-alike "
    "confusions, and use the conversation as context. If you're given words this user uses and past misreadings, "
    "use them: a hypothesis that resembles a past misreading probably means the same thing again. "
    "Numbers are the least reliable part: '-teen' and '-ty' look almost identical (thirteen/thirty, fifteen/fifty), "
    "the first part of a compound number often drops out ('twenty-three' comes out as '3'), and digits with similar "
    "lip shapes get swapped (two/ten/eight, five/nine). Numbers may come out as digits or words. The user may say "
    "numbers digit by digit ('two two zero' = 220, 'two three gauge' = 23 gauge) and may spell with the NATO "
    "alphabet (Alpha, Bravo, Charlie, ...; expect garbled versions such as 'ALFA' or 'BROTHER' for Bravo); turn "
    "those into the number or the letters. Single spoken letters are nearly unreadable. When a number or spelling "
    "matters to your answer and the hypotheses disagree or it looks implausible, say the value you're assuming in "
    "a few words (e.g. 'Assuming 23-gauge: ...') so the user can correct you."
)
APP_CONTROL = (
    "You can also control this app. Each message starts with its current state. If the user clearly asks to change "
    "how the app works, add one line per change, 'ACTION: <action>', using only these actions: "
    "'type on' / 'type off' (TYPE mode: what they mouth is typed into the app in front of them instead of being "
    "sent to you), 'voice on' / 'voice off' (your replies spoken aloud), 'face on' / 'face off' (face gestures: a "
    "long blink starts listening; in TYPE mode a smile deletes words and an eyebrow raise presses Enter), "
    "'mic on' / 'mic off' (out-loud speech recognition, e.g. for correcting names), 'handsfree on' / 'handsfree off' "
    "(start listening by itself when they mouth words while facing the screen), 'wake on' / 'wake off' (always "
    "listening for 'Hey Claude' through the microphone, so they can talk to you out loud anytime, even with the "
    "overlay hidden), 'hide' (hides the overlay and turns the camera off; they bring it back with the \\ key). "
    "Never add ACTION lines unless they asked."
)
MAC_CONTROL = (
    "You can act on the user's Mac like a careful power user, through your Bash tool: ./macctl <command> in your "
    "working directory (nothing else runs). Use the cheapest way that works, in this order:\n"
    "1. Direct commands: 'open <App> [<App> ...]', 'focus <App>', 'url <https://...>', 'apps' (what's open), "
    "'place <App> left|right|full|center|top|bottom', 'arrange <App1> <App2>' (side by side; one call does "
    "everything), 'fullscreen <App> on|off', 'minimize|restore <App>', "
    "'menu <App> \"File > Export...\"' (choose a menu item; 'menu <App> <Menu>' lists it with keyboard shortcuts), "
    "'key <combo> ...' (cmd+k, return, escape, tab, down*3, cmd+shift+t) and 'type <text>' (at the cursor). Many tasks "
    "are just a shortcut plus typing: run 'skill <App>' first for that app's tips (short).\n"
    "2. The accessibility tree, for anything in a window: 'ui [App] [--find words]' lists its controls as #ids plus "
    "visible text (exact and cheap; use --find to keep it short). Then 'press #id' (buttons, links, tabs, rows; "
    "doesn't move the mouse), 'set #id <text>' (replace a field's text), 'click #id'. Read messages, lists and "
    "notifications from ui text, not screenshots.\n"
    "Each message starts with a short 'Screen when they said it' summary (front app and window, focused field, "
    "selected text, page address, main controls by label). Use it to resolve 'this' / 'here' / 'that' and to act "
    "directly (press \"Send\", set \"Search\" <text>: labels work in quotes) instead of running ui first; run ui "
    "only when what you need isn't listed. '(Screen: unchanged ...)' means as in their previous message. "
    "If there's a 'Looking at' line, it's where their eyes were as they spoke (webcam eye tracking, give or take "
    "the stated radius, so it lists the nearest controls). It's background awareness, like a friend glancing at "
    "what they're looking at: use it to understand what they mean ('what does this say', 'is that right', "
    "'reply to him', a question about something on screen), not as a pointer they aim. Don't mention it unless it "
    "helps, and never act on it alone: press or change something only when they ask, and if what they mean isn't "
    "clear from their words plus what they were looking at, ask (naming the likely control). "
    "Notifications that just arrived are listed too: 'reply ...' means reply to the latest one (open its app and "
    "conversation, type the reply, then ASK FIRST before sending).\n"
    "3. Only if ui can't see it (a canvas, image, game or odd app): 'see [App]' saves a small screenshot; Read it, "
    "then 'click x y' with pixel coordinates in that image ('see --zoom x y w h' to look closer at part of it).\n"
    "Batch steps you can predict into ONE call: ./macctl do \"focus Discord\" \"key cmd+k\" \"type alex\" "
    "\"wait 0.5\" \"key return\" (it stops at the first failure). Every action prints what has focus afterwards: "
    "trust that instead of checking again; 'wait-for <text>' waits for something to appear. If a name the "
    "lip-reader gave you is unclear, look at what's really there (ui --find, or the app's list) and pick the closest "
    "match. When you work out something non-obvious about an app, save it for next time: "
    "./macctl learn <App> \"<one-line tip>\".\n"
    "SHORTCUTS: when you've finished a task they're likely to ask for again and it worked, save it in the same "
    "turn so next time it runs instantly without you: ./macctl remember \"open my DMs with {name}\" \"focus "
    "Discord\" \"key cmd+k\" \"type {name}\" \"wait 0.5\" \"key return\". Use {blanks} for the parts that change, "
    "and start a step with ask: if it sends, posts, deletes, buys or can't be undone (it will ask them before that "
    "step). Don't save one-off or failed tasks, or ones that type private content. './macctl shortcuts' lists "
    "them; 'forget <n>' and 'rename <n> <phrase>' manage them when they ask.\n"
    "ASK FIRST and wait for a yes in their next message before anything that sends, posts, deletes, buys, submits, "
    "books, starts a machine (e.g. a 3D print), changes settings or accounts, or can't be undone. Say exactly what "
    "will happen (e.g. 'Send \"on my way\" to Alex on Discord?'). Typing a draft is fine; pressing send is not, until "
    "they say yes. Never type passwords, codes or payment details, and never sign in: they do that themselves. If a "
    "command says STOPPED, stop immediately and say so. While working, say what you're doing in a few words, not a "
    "running commentary; when done, report the result briefly. Summarise private content instead of reading it out "
    "at length.\n"
    "Spotify: './macctl spotify now' (what's playing), './macctl spotify play' (resume) or "
    "'./macctl spotify play <spotify: URI or open.spotify.com link>', './macctl spotify pause|toggle|next|previous', "
    "'./macctl spotify volume <0-100|up|down>', './macctl spotify shuffle on|off', './macctl spotify repeat on|off'. "
    "To play something specific, get its Spotify URI or link (from your spotify tools if you have them, otherwise "
    "WebSearch for its open.spotify.com link), then play that; play/pause/next already report what's playing. This all "
    "happens in the background: don't open or focus Spotify, and don't use 'spotify search' (it brings Spotify to the "
    "front) unless they ask to see results. "
    "Quote app names that contain spaces. For harmless, easily undone actions (playing or pausing music, opening an "
    "app or page, searching, scrolling) just do the most likely reading; don't ask to confirm, they can say stop. "
    "Ask first only when it's genuinely unclear what they want, or before anything that sends, deletes, buys or "
    "can't be undone. "
    "Write the HEARD line once, at the very start (never again after using a tool)."
)
LIVE_SYSTEM = (
    "You are Claude, talking with a user who speaks to you by silently mouthing words to a camera; a lip-reading "
    "model transcribes them. Each user message gives the lip-reader's top hypotheses for what they just said "
    "(sometimes in several parts if they paused mid-thought; the parts form one message), or text they typed, "
    "or what they said out loud (marked '(said out loud)': speech recognition, reliable; take it as given), "
    "possibly preceded by notes in parentheses (e.g. that they interrupted you or corrected an earlier message) and "
    "by what you've learned about them. " + LIPREADER_NOTES + " Work out what the user most likely said and reply "
    "to that.\n"
    "Always answer in this format:\n"
    "HEARD: <what you think they just said; for typed text, the text itself>\n"
    "FIX: <the corrected full text of their PREVIOUS message>  <- only when they're telling you that you misheard "
    "their previous message (e.g. \"no, I said Nissan\")\n"
    "<your reply on the next line(s)>\n"
    "Your reply may be read aloud, so keep it short (one to three sentences), plain text, no markdown or lists. "
    "If you genuinely can't tell what they said, write 'HEARD: ?' and briefly ask them to repeat or rephrase. "
    "If they interrupted you, don't repeat yourself; respond to what they just said.\n"
    + APP_CONTROL + " Put ACTION lines right after HEARD (and FIX), then confirm briefly in your reply."
)
INTERPRET_SYSTEM = (
    "You turn lip-reading output into text. The user is chatting with Claude by silently mouthing words to a "
    "camera, and a camera-only lip-reading model transcribed their next message. " + LIPREADER_NOTES +
    " Reply with up to three different interpretations of what they said, most likely first, one per line, with "
    "no numbering, quotes or commentary."
)
CHAT_SYSTEM = (
    "You are Claude, chatting with a user who talks to you by silently mouthing words to a lip-reading camera. "
    "Each of their messages was reconstructed from lip movements and confirmed by them, so it may still be "
    "slightly off; if something seems garbled, ask briefly. Your replies are shown in a small window and may be "
    "read aloud, so keep them short (one to three sentences), plain text, no markdown or lists."
)
DICTATE_SYSTEM = (
    "You turn lip-reading output into dictated text. The user is silently mouthing text to type into an app on "
    "their Mac (a message, an email, a document, a search box). Each message gives the lip-reader's top hypotheses "
    "for what they just mouthed (sometimes in parts) or what they said out loud (marked '(said out loud)': "
    "reliable, use it as given), the app they're typing into, and what has been typed there "
    "so far, possibly with what you've learned about them. " + LIPREADER_NOTES + "\n"
    "You are a transcriber, not an assistant: never answer, comment on or carry out what they say. Questions and "
    "requests are just text to type (lip-reader 'whats the weather on sunday' -> TYPE: What's the weather on "
    "Sunday?).\n"
    "Reply with exactly 'TYPE: ' followed by the text to type: what they most likely said, with natural punctuation "
    "and capitalisation that continue the text typed so far (no capital mid-sentence, don't repeat earlier text). "
    "Spoken commands become symbols: 'new line' -> a line break, 'new paragraph' -> two line breaks, 'comma', "
    "'period', 'question mark' and so on -> the punctuation. Numbers as digits where natural. No quotes or "
    "commentary. If you can't make out anything plausible, reply with exactly: ?\n"
    "If instead they are clearly telling the app itself to change mode (e.g. 'turn off typing mode', 'switch to "
    "chat', 'mic on', 'voice off', 'hide'), reply with only ACTION lines and type nothing. " + APP_CONTROL
)
# Face gestures (MediaPipe blendshape scores, 0-1)
# Tuned to your face (logged): open eyes read ~0.2-0.37; a deliberate blink ~0.78 left / ~0.65 right.
BLINK_CLOSED, BLINK_OPEN, BLINK_HOLD = 0.55, 0.42, 0.3  # long blink starts listening (most natural blinks < 0.25 s)
BLINK_EACH = 0.48                                       # ... and each eye must be at least this closed
SMILE_ON, SMILE_OFF, JAW_MAX = 0.45, 0.25, 0.15         # closed-lip smile ...
SMILE_FIRST, SMILE_EVERY = 0.7, 1.0                     # ... deletes a word after 0.7 s, then one per second
BROWS_UP, BROWS_DOWN, BROWS_HOLD = 0.5, 0.25, 0.3       # eyebrow raise sends (Enter)
WINK_CLOSED, WINK_OPEN, WINK_HOLD = 0.6, 0.3, 0.3       # ... or a wink, with --send wink
GESTURE_COOLDOWN = 1.0                                  # no gestures right after you finish mouthing
# Hands-free start: lips moving while you face the screen
HF_HOLD, HF_MIN, HF_COOLDOWN = 0.35, 0.7, 1.5  # mouthing this long starts it; shorter captures are dropped
HF_YAW, HF_PITCH = 25, (-30, 20)               # facing the screen (head pose, degrees)
CUE_WORDS = {"claude", "cloud", "clod", "clawed", "hey"}
# Instant commands by closed-set lip-reading (see eval_commands.py for how these were measured)
CLOSED_MARGIN, CLOSED_GAP = 2.0, 2.0  # eval (personal, held-out, short clips): 41% instant & right, 3% wrong, 3% false accepts
TYPE_MARK = "TYPE:"  # dictation replies must start with this; anything else is never typed
SENTENCE_END = re.compile(r"[.!?…][\"')\]]*\s")


# ----------------------------------------------------------------------------- shared helpers

def transcript(history, last=12):
    if not history:
        return "(no messages yet; this is the first one)"
    return "\n".join(f"{who}: {text}" for who, text in history[-last:])


class Speaker:
    """Speaks replies with macOS `say`, sentence by sentence as they arrive; can be cut off and switched on/off.

    device "auto" speaks through AirPods whenever they're connected to this Mac (even if they aren't the system
    default output), otherwise through the default output. Any other value matches an output device by name.
    """

    def __init__(self, enabled, rate=190, device="auto"):
        self.enabled, self.rate, self.device = enabled, rate, device
        self.proc, self.current, self.queue, self.args, self.last_output = None, None, [], None, None
        self.started, self.spoken, self.total, self.text = 0.0, 0, 0, ""

    @staticmethod
    def outputs():
        """[(id, name)] of audio outputs `say` can use (checked per reply, so AirPods can connect mid-chat)."""
        out = subprocess.run(["say", "-a", "?"], capture_output=True, text=True, timeout=5).stdout
        return [tuple(l.strip().split(None, 1)) for l in out.splitlines() if l.strip()[:1].isdigit()]

    def _route(self):
        if self.device == "default":
            return [], "the system default output"
        want = "airpods" if self.device == "auto" else self.device.lower()
        match = next(((i, n) for i, n in self.outputs() if want in n.lower()), None)
        if match:
            return ["-a", match[0]], match[1]
        hint = " (AirPods aren't connected to this Mac)" if self.device == "auto" else f" ('{self.device}' not found)"
        return [], "the system default output" + hint

    def begin_reply(self):
        self.queue, self.spoken, self.total, self.args, self.text = [], 0, 0, None, ""

    def say(self, text):
        text = text.strip()
        if self.enabled and text:
            self.queue.append(text)
            self.total += len(text.split())
            self.text += " " + text
            self.update()

    def update(self):
        """Call every frame: starts the next queued sentence when the previous one has finished."""
        if self.proc is not None and self.proc.poll() is not None and self.current:
            self.spoken += len(self.current.split())
            self.current = None
        if self.speaking or not self.queue:
            return
        if self.args is None:
            self.args, name = self._route()
            if name != self.last_output:
                print(f"Speaking through {name}.", flush=True)
                self.last_output = name
        self.current = self.queue.pop(0)
        self.proc = subprocess.Popen(["say", "-r", str(self.rate), *self.args, self.current])
        self.started = time.monotonic()

    @property
    def speaking(self):
        return self.proc is not None and self.proc.poll() is None

    @property
    def active(self):
        return self.speaking or bool(self.queue)

    def stop(self):
        """Stop talking. Returns (words spoken, words in the reply so far), or None if it wasn't talking."""
        if not self.active:
            return None
        partial = 0
        if self.speaking:
            partial = min(len(self.current.split()), int((time.monotonic() - self.started) * self.rate / 60))
            self.proc.terminate()
        said = self.spoken + partial
        self.queue, self.current = [], None
        return said, self.total


class Hotkey:
    """Global push-to-talk key: polls the system-wide modifier state, so it works whichever app is in front.

    Counts as held only when this modifier is down on its own (so shortcuts like Cmd+Option+... don't trigger it).
    """

    def __init__(self, name="option"):
        import Quartz
        self.Q, self.name = Quartz, name
        cmd, shift, ctrl, alt = (Quartz.kCGEventFlagMaskCommand, Quartz.kCGEventFlagMaskShift,
                                 Quartz.kCGEventFlagMaskControl, Quartz.kCGEventFlagMaskAlternate)
        self.mask, self.others = {
            "option": (alt, cmd | shift | ctrl),
            "right_option": (alt, cmd | shift | ctrl),
            "control": (ctrl, cmd | shift | alt),
        }[name]
        self.label = {"option": "Option", "right_option": "right Option", "control": "Control"}[name]

    def held(self):
        flags = self.Q.CGEventSourceFlagsState(self.Q.kCGEventSourceStateHIDSystemState)  # physical keys only
        if self.name == "right_option" and not flags & 0x40:  # NX_DEVICERALTKEYMASK
            return False
        return bool(flags & self.mask) and not flags & self.others


def stdin_reader(q):
    for line in sys.stdin:
        if line.strip():
            q.put(line.strip())


def fifo_reader(q, path):
    """Commands from the launcher window through a named pipe, so the chat keeps running without that window."""
    while True:
        try:
            with open(path) as f:  # waits until the launcher window sends something
                for line in f:
                    if line.strip():
                        q.put(line.strip())
        except OSError:
            time.sleep(1)


# ----------------------------------------------------------------------------- live mode (hold Option / smile)

SAVE_CLIPS = [True]  # Settings > "Save mouth clips for training" (off: clips are read, then thrown away)


# lips look the same for these sounds (visemes): a misread usually swaps within a group
VISEME = {**{c: "P" for c in "pbm"}, **{c: "F" for c in "fv"}, **{c: "T" for c in "tdnszl"}, **{c: "K" for c in "kgqcx"},
          **{c: "S" for c in "jy"}, "r": "R", "w": "W", "h": "", **{c: "A" for c in "aeiou"}}


def _visemes(word):
    w = word.lower()
    w = w.replace("ch", "j").replace("sh", "j").replace("th", "t").replace("ph", "f").replace("ck", "k")
    out = "".join(VISEME.get(c, "") for c in w)
    return "".join(c for i, c in enumerate(out) if i == 0 or c != out[i - 1])  # (repeats look like one)


def _close(a, b):
    """Edit distance between viseme strings, relative to length."""
    import difflib
    if not a or not b:
        return 1.0
    return 1 - difflib.SequenceMatcher(None, a, b).ratio()


def screen_variants(hyps, words, limit=24):
    """The top guesses with a word swapped for an on-screen word that looks alike on the lips
    ('SARA' -> 'Zara' if Zara is in the window in front)."""
    words = {w for w in words if len(w) >= 3}
    vis = {w: _visemes(w) for w in words}
    out = []
    for text, _ in hyps[:3]:
        toks = text.split()
        for i, t in enumerate(toks):
            vt = _visemes(t)
            for w, vw in vis.items():
                if (w.lower() != t.lower() and vt[:1] == vw[:1] and 0.75 <= len(vw) / max(1, len(vt)) <= 1.34
                        and _close(vt, vw) <= 0.3):
                    v = " ".join(toks[:i] + [w.upper()] + toks[i + 1:])
                    if v not in out and v != text:
                        out.append(v)
    return out[:limit]


def lipread(reader, frames, stamps, clip, io_pool, audio=None, transcriber=None, vocabulary=(), candidates=None,
            screen_words=()):
    """Captured frames -> lip-reader n-best (+ speech-to-text if the mic was on). The clip is saved in the
    background, off the critical path. candidates() -> [(phrase, action)]: also scores those phrases against the
    clip (closed-set lip-reading for instant commands); the best is in result["closed"]."""
    spoken = transcriber(audio, vocabulary) if audio is not None and transcriber is not None else ""
    video = vsr.resample_to_fps(frames, stamps)
    if SAVE_CLIPS[0]:
        io_pool.submit(vsr.write_video, video, clip, pix_fmt="bgr24")
    crops, _ = reader.mouth_crops(np.ascontiguousarray(video[..., ::-1]))
    enc = reader.encode(crops)
    hyps = reader.decode_enc(enc, nbest=10)
    result = {"clip": clip, "hyps": hyps, "typed": None, "spoken": spoken, "closed": None}
    cands = candidates() if candidates and not spoken else []
    variants = screen_variants(hyps, screen_words) if screen_words and not spoken else []
    if variants:  # an on-screen word that fits the video better than the lip-reader's own guess moves to the top
        t0 = time.perf_counter()
        scored = dict(reader.score(enc, [hyps[0][0]] + variants))
        best = max(variants, key=lambda v: scored[v])
        if scored[best] > scored[hyps[0][0]] + 0.5:
            hyps.insert(0, (best, scored[best]))
            result["biased"] = {"from": hyps[1][0], "to": best, "gain": round(scored[best] - scored[hyps[1][0]], 2),
                                "ms": round((time.perf_counter() - t0) * 1000)}
    if len(cands) >= 2:
        t0 = time.perf_counter()
        scored = reader.score(enc, [p for p, _ in cands])
        actions = dict(cands)
        result["closed"] = {"phrase": scored[0][0], "action": actions[scored[0][0]], "score": scored[0][1],
                            "second": scored[1][1], "runner_up": scored[1][0], "free": hyps[0][1],
                            "n": len(cands), "ms": round((time.perf_counter() - t0) * 1000)}
    return result


def describe(p):
    """How a message part is shown to Claude."""
    if p.get("typed"):
        return f"(typed) {p['typed']}"
    if p.get("gesture"):
        return f"({'nodded' if p['gesture'] == 'nod' else 'shook their head'}, i.e.) {p['typed']}"
    if p.get("voice"):
        return f"(said out loud after 'Hey Claude'; they may not be looking at the screen) {p['spoken']}"
    if p.get("spoken"):
        return f"(said out loud) {p['spoken']}"
    return format_nbest(p["hyps"])


def handsfree_stats(path, hours=24):
    """From the hands-free log: starts that matched a real start (Option or long blink) within 3 s, ones on their
    own (false starts), and real starts it would have missed."""
    events = []
    try:
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            stamp = datetime.strptime(" ".join(parts[:2]), "%Y-%m-%d %H:%M:%S.%f")
            if (datetime.now() - stamp).total_seconds() < hours * 3600:
                events.append((float(parts[2]), parts[4]))
    except (OSError, ValueError, IndexError):
        pass
    starts = [t for t, e in events if e in ("would-start", "auto-start")]
    real = [t for t, e in events if e == "real-start"]
    matched = [t for t in starts if any(-1.5 <= r - t <= 3 for r in real)]
    missed = [r for r in real if not any(-1.5 <= r - t <= 3 for t in starts)]
    used = sum(e == "auto-used" for _, e in events)
    dropped = sum(e == "auto-dropped" for _, e in events)
    return (f"last {hours} h: {len(starts)} hands-free starts, {len(matched)} at the same time as you started "
            f"(Option/blink), {len(starts) - len(matched)} on their own; {len(missed)} of your {len(real)} starts "
            f"it would have missed; auto captures sent {used}, dropped as too short {dropped}")


def usage_stats(path, hours=24):
    """From the usage log: tokens and model calls per message, with and without the screen summary."""
    rows = []
    try:
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            if (datetime.now() - datetime.strptime(r["at"], "%Y-%m-%d %H:%M:%S")).total_seconds() < hours * 3600:
                rows.append(r)
    except (OSError, ValueError, KeyError):
        pass
    if not rows:
        return f"no Claude messages in the last {hours} h"
    out = []
    for name, sel in (("with screen", [r for r in rows if r["screen"]]), ("without", [r for r in rows if not r["screen"]])):
        if sel:
            n = len(sel)
            avg = lambda k: sum(r[k] for r in sel) / n
            out.append(f"{name}: {n} messages, per message {avg('fresh') + avg('cache_write'):.0f} new input tokens "
                       f"+ {avg('cache_read'):.0f} cached, {avg('out'):.0f} output, {avg('api_calls'):.1f} model calls, "
                       f"{avg('tools'):.1f} tool calls, {avg('ms') / 1000:.1f} s")
    return f"last {hours} h: " + "; ".join(out)


def steps_of(cmd):
    """'./macctl do "focus Discord" "key cmd+k"' -> 'focus Discord → key cmd+k' (for the overlay)."""
    import shlex
    try:
        parts = shlex.split(cmd.split("macctl", 1)[1])
    except (ValueError, IndexError):
        return cmd
    return " → ".join(parts[1:]) if parts[:1] == ["do"] else " ".join(parts)


def read_hold(reader, transcriber, frames, stamps, clip, io_pool, audio, vocabulary):
    """Option held while silent chat is hidden: what you said out loud (Whisper), or if you only mouthed it, the
    lip-reader's hypotheses."""
    import speech
    if audio is not None and transcriber is not None and speech.voiced(audio) >= 0.15:
        spoken = transcriber(audio, vocabulary)
        if spoken:
            return {"clip": None, "hyps": None, "typed": None, "spoken": spoken}
    if len(frames) < 12:
        raise ValueError("didn't hear speech, and the camera didn't start in time to see mouthing")
    return lipread(reader, frames, stamps, clip, io_pool)


def build_message(parts, notes, memory_block, state="", screen=""):
    body = (describe(parts[0]) if len(parts) == 1 else
            "\n".join(f"Part {i}:\n{describe(p)}" for i, p in enumerate(parts, 1)))
    blocks = ([state] if state else []) + ([screen] if screen else []) + ([memory_block] if memory_block else []) + notes
    return "\n\n".join(blocks + ["The user's new message (lip-reader hypotheses, best first):\n" + body])


def run_live(args):
    from claude_session import ClaudeSession
    from overlay import Overlay

    import connectors
    found_box = {}  # which claude.ai connectors are connected: checked while the lip-reader loads
    disc = threading.Thread(target=lambda: found_box.update(connectors.discover()), daemon=True)
    if not args.no_connectors and not args.no_mac:
        disc.start()
    print("Loading the lip-reader ...", flush=True)
    reader = vsr.LipReader(args.device, beam_size=args.beam)
    which = "your personal model" if reader.ckpt == vsr.PERSONAL_CKPT else "the base model"
    activity = vsr.MouthActivity()
    cam = vsr.Recorder(args.camera, window=False, start=not args.hidden)  # hidden at login: camera stays off
    hotkey, speaker = Hotkey(args.hotkey), Speaker(args.speak, device=args.audio)
    labels, memory = Labels(), PersonalMemory()
    cal = json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else {"on": 0.04, "off": 0.024}
    ui = {"face": not args.no_face, "quit": False, "type": False, "open": True,
          "toggle": None, "mic": False, "restart": False}
    # the app's launcher: SIGTERM = quit cleanly, SIGUSR1 = show (you opened the app again)
    signal.signal(signal.SIGTERM, lambda *_: ui.update(quit=True))
    signal.signal(signal.SIGUSR1, lambda *_: ui.update(toggle=True))
    import speech
    mic, transcriber = speech.Mic(prefer=args.mic_device), {"model": None}
    settings_file = DATA / "settings.json"  # settings that should survive restarts (e.g. Hey Claude on/off)
    try:
        settings = json.loads(settings_file.read_text())
    except (OSError, ValueError):
        settings = {}

    def save_setting(key, value):
        settings[key] = value
        try:
            settings_file.write_text(json.dumps(settings))
        except OSError:
            pass

    def top_vocab():
        words = list(memory.data["vocabulary"].items())
        return [w for w, _ in sorted(words, key=lambda kv: -kv[1])[:40]]

    def wake_muted():  # don't listen to Claude's own voice, or while you're dictating / pushing to talk
        try:
            return speaker.active or time.monotonic() - voice["quiet_since"] < 0.7 or cap is not None or hold is not None
        except NameError:
            return True
    wake = speech.WakeWord(mic, lambda: transcriber["model"], vocabulary=top_vocab, muted=wake_muted)
    ui["handsfree"] = args.handsfree or settings.get("handsfree", "shadow")  # off | shadow (measure only) | on
    import headmotion
    nod_airpods, nod_camera = headmotion.NodShake(), headmotion.NodShake.for_camera()
    airpods = headmotion.AirPodsMotion(DATA / "motion.fifo") if settings.get("airpods_motion", True) else None
    if airpods is not None:
        airpods.on_sample = nod_airpods.add
    ui["question_at"], ui["nod_quiet_until"] = -99.0, 0.0
    # what's on screen goes with each message (control.context), built while the lip-reader is still working
    import control
    import notify
    control.SKIP_PIDS.update({os.getpid(), os.getppid()})
    ctx_pool = cf.ThreadPoolExecutor(1)
    screen = {"future": None, "at": -99.0, "last": None}
    ui["screen"] = settings.get("screen_context", True) and not args.no_mac
    usage_log = DATA / "usage.log"
    # new notifications: shown in the overlay, read out when voice is on, and offered to Claude for "reply ..."
    notif_q, ui["notify"], ui["notifs"] = queue.Queue(), settings.get("notifications", True), []
    notif_watch = notify.Watcher(notif_q.put)
    notif_watch.enabled = ui["notify"]
    # where you're looking (gaze.py): camera iris + head pose (+ AirPods), after /gaze calibrate
    try:  # MLX keeps freed GPU buffers around for reuse; a small cap stops that growing to gigabytes
        import mlx.core as _mx
        _mx.set_cache_limit(256 * 1024 * 1024)
    except Exception:
        pass
    import gaze as gaze_mod
    gazer = gaze_mod.Gaze(DATA / "gaze.json")
    ui["gaze"] = settings.get("gaze", True)
    ui["lens"] = gazer.lens_on = settings.get("gaze_lens", True)
    # self-improvement (improve.py): learns shortcuts and per-app tips once a day while you're idle
    imp = {"next_check": time.monotonic() + 120, "running": False, "q": queue.Queue(), "last_msg": time.monotonic()}

    def improve_now(dry=False, quiet=False):
        if imp["running"]:
            return
        imp["running"] = True

        def go():
            try:
                changes, stats = improve.run(dry=dry, force=not quiet)
            except Exception as e:
                changes, stats = [f"(failed: {e})"], {}
            imp["running"] = False
            imp["q"].put((changes, stats, dry, quiet))
        threading.Thread(target=go, daemon=True).start()
    ui["warp"] = gazer.warp_on = settings.get("gaze_warp", True)  # the lens bends the screen under it
    if ui["lens"] and ui["warp"] and not gaze_mod.Warper.allowed(ask=True):
        print("[gaze] the glass circle's bend needs Screen Recording: allow Silent Chat in System Settings > "
              "Privacy & Security > Screen & System Audio Recording, then restart. Until then it's drawn only.",
              flush=True)
    hf_log = DATA / "handsfree.log"

    def hf_note(event, detail=""):
        """Hands-free log, to measure false starts: would-start/auto-* vs real-start (Option, long blink)."""
        with open(hf_log, "a") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M:%S.%f} {time.monotonic():.2f} {ui['handsfree']} {event} {detail}\n")

    def mic_on(flag):
        if flag:
            name = mic.start()
            if transcriber["model"] is None:
                transcriber["model"] = speech.Transcriber(idle_unload=120)  # (loads on first use, ~1 s)
            return name
        if not wake.enabled:  # "Hey Claude" keeps the mic open
            mic.stop()
    import typer
    dictation = {"session": None, "turn": None, "pending": [], "pid": None, "app": "", "field": "", "last": "",
                 "words": 0, "last_clips": [], "deleted": False, "memory_sent": None}

    def on_button(name):
        if name in ("voice", "face", "type", "mic", "wake", "handsfree") and not ui.get("from_chat"):
            ui["note_change"] = name  # tell Claude on your next message (it didn't make this change)
        if name == "voice":
            speaker.enabled = ui["voice_pref"] = not ui["voice_pref"]
            if not speaker.enabled:
                speaker.stop()
            print(f"Voice {'on' if speaker.enabled else 'off (text only)'}.", flush=True)
        elif name == "face":
            ui["face"] = not ui["face"]
            print(f"Face gestures {'on' if ui['face'] else 'off'}.", flush=True)
        elif name == "close":
            ui["toggle"] = False  # hide; the toggle key (or /show) brings it back
        elif name == "type":
            ui["type"] = not ui["type"]
            if ui["type"]:
                speaker.stop()
                if dictation["session"] is None:
                    dictation["session"] = ClaudeSession(DICTATE_SYSTEM, model=MODELS["haiku"], effort=args.effort)
                if not typer.trusted(prompt=True):
                    set_notice(f"Allow {HOST} in System Settings > Privacy & Security > Accessibility so I can "
                               "type for you.")
            print(f"{'TYPE mode: what you mouth is typed into the app in front' if ui['type'] else 'CHAT mode'}.",
                  flush=True)
        elif name == "mic":
            ui["mic"] = not ui["mic"]
            if ui["mic"]:
                try:
                    used = mic_on(True)
                    set_notice(f"MIC on ({used}): speak out loud while holding {hotkey.label}, e.g. to correct a "
                               "name. Mouthing silently still works.")
                except Exception as e:
                    ui["mic"] = False
                    set_notice(f"Couldn't open the microphone ({e}). Allow {HOST} under Privacy & Security > "
                               "Microphone.")
            else:
                mic_on(False)
                set_notice("MIC off.")
            print(f"MIC {'on' if ui['mic'] else 'off'}.", flush=True)
        elif name == "wake":
            set_wake(not wake.enabled)
        elif name == "handsfree":
            set_handsfree("off" if ui["handsfree"] == "on" else "on")

    def set_handsfree(mode):
        ui["handsfree"] = mode
        save_setting("handsfree", mode)
        set_notice({"on": "Hands-free on: just start mouthing while you face the screen.",
                    "off": "Hands-free off.", "shadow": "Hands-free is measuring only (it won't start by itself)."}[mode])
        print(f"Hands-free {mode}.", flush=True)

    def set_wake(flag):
        if flag:
            try:
                mic_on(True)
            except Exception as e:
                set_notice(f"Couldn't open the microphone ({e}). Allow {HOST} under Privacy & Security > Microphone.")
                flag = False
        wake.enabled = flag
        if not flag and not (ui["mic"] and ui["open"]) and hold is None:
            mic.stop()
        save_setting("wake", flag)
        set_notice('Say "Hey Claude" anytime to talk out loud (the mic stays on).' if flag else "Hey Claude off.")
        print(f"Hey Claude {'on: listening through the mic' if flag else 'off'}.", flush=True)


    def set_notice(text):
        nonlocal notice
        notice = text

    panel = Overlay(on_button, DATA / "overlay.json", on_text=lambda t: typed.put(t))
    # ambient: the camera stays on, and the panel shows only while you're using it (presence.py); \ pins it open
    ui["ambient"], ui["pinned"] = settings.get("ambient", True), False
    if ui["ambient"]:  # (the island replaces the panel: keep it from flashing up at start)
        panel.alpha = panel.alpha_goal = 0.0
        panel.panel.setAlphaValue_(0.0)
        panel.hide()
    pres, ducker = presence.Presence(), presence.Ducker()

    def trim_gpu():
        """Give back GPU memory the lip-reader (PyTorch) and MLX cached but aren't using (often ~2 GB)."""
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        try:
            import mlx.core as _mx
            _mx.clear_cache()
        except Exception:
            pass

    def running_app_names():
        try:
            from AppKit import NSWorkspace
            return [a.localizedName() for a in NSWorkspace.sharedWorkspace().runningApplications()
                    if a.activationPolicy() == 0]
        except Exception:
            return []

    def spotify_control(name):
        """The island's now-playing buttons."""
        verb = {"previous": "previous track", "toggle": "playpause", "next": "next track"}[name]
        threading.Thread(target=lambda: subprocess.run(["osascript", "-e", f'tell application "Spotify" to {verb}'],
                                                       capture_output=True), daemon=True).start()
        now_playing_soon(0.5)

    def island_tap(what):
        """A click on the island: open it up with more of the conversation (click again to close), or a vote."""
        if what == "island":
            amb["history_open"] = not amb.get("history_open")
            isl.max_text = 440.0 if amb["history_open"] else 220.0
        elif what in ("up", "down"):
            vote_q.put(what)
    vote_q = queue.Queue()
    def sent_from_box(text):
        """Typed in the island's box: send it, give the keyboard back, and let the island close normally once the
        reply has been read, even with the mouse still over it (hovering opens it again after you leave)."""
        typed.put(text)
        amb["hover_lock"], amb["hover"] = True, False
        amb["unfocus"] = True
    isl = island.Island(on_text=sent_from_box, on_control=spotify_control, on_tap=island_tap)

    def on_escape(event):
        if event.keyCode() == 53:  # Esc: put the island away
            esc_q.put(1)
    esc_q = queue.Queue()
    from AppKit import NSEvent as _NSEv
    esc_monitor = _NSEv.addGlobalMonitorForEventsMatchingMask_handler_(1 << 10, on_escape)  # (key down)
    SAVE_CLIPS[0] = labels.enabled = settings.get("save_clips", True)
    gazer.learn_clicks = settings.get("gaze_clicks", True)
    clicks_q = queue.Queue()  # your clicks, for gaze to learn from (gaze.add_click)

    def on_click(event):
        from AppKit import NSEvent
        p = NSEvent.mouseLocation()
        clicks_q.put((time.monotonic(), p.x, NSScreen.screens()[0].frame().size.height - p.y))
    from AppKit import NSEvent as _NSEvent
    click_monitor = _NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(1 << 1, on_click)  # (left mouse down)
    # hands-free reads that might be requests: a small local model decides (local_router.py), so Claude only gets
    # the real ones and simple ones run instantly
    router, routed = local_router.Router(), queue.Queue()
    guard = privacy.Guard(router)  # (the same small model judges private windows)

    def route_async(result):
        amb["deciding"] = (result["hyps"][0][0].capitalize(), time.monotonic())
        front = (screen.get("last") or "").split("\n", 1)[0]

        def go():
            try:
                d = router.decide(result["hyps"], front, running_app_names())
            except Exception as e:
                d = {"kind": "claude", "text": result["hyps"][0][0], "cmd": None, "ms": 0, "error": str(e)}
            routed.put((result, d))
        threading.Thread(target=go, daemon=True).start()
    amb = {"facing": ({"yaw": float(np.median(np.array(settings["real_poses"])[:, 0])),
                       "pitch": float(np.median(np.array(settings["real_poses"])[:, 1]))}
                      if len(settings.get("real_poses") or []) >= 8 else None),
           "hidden_at": None, "noface_since": None, "read_failed": 0.0, "duck_off_since": None, "np": queue.Queue(),
           "hf_gain": settings.get("hf_gain", 1.0)}

    def headphones():
        """Headphones in? AirPods report head motion only while they're in your ears; other headphones by name."""
        if airpods is not None and airpods.helper_ok():
            return airpods.active()
        try:
            name = speaker._route()[1].lower()
        except Exception:
            name = ""
        return any(w in name for w in ("headphone", "buds", "beats", "airpods"))

    def front_line():
        return (screen.get("last") or "").split("\n", 1)[0]

    def speak_now(spoke_aloud):
        return presence.should_speak(headphones(), spoke_aloud, ui["voice_pref"], presence.watching(front_line()))

    def now_playing_soon(delay=1.2):
        """A short card with what Spotify is playing now (after a play / skip)."""
        def go():
            time.sleep(delay)
            r = subprocess.run(["osascript", "-e", 'tell application "Spotify" to return (player state as string) & '
                                '"|" & name of current track & "|" & artist of current track & "|" & '
                                'artwork url of current track'], capture_output=True, text=True)
            parts = r.stdout.strip().split("|")
            if r.returncode == 0 and len(parts) == 4:
                amb["np"].put({"state": parts[0], "title": parts[1], "artist": parts[2], "art": parts[3]})
        threading.Thread(target=go, daemon=True).start()

    def screen_read(gaze=None):
        """The screen summary, through the privacy check (privacy.py): private windows become one line."""
        return guard.filter(control.context(gaze=gaze))

    def screen_soon(look=None):
        """Start reading the screen now (in the background), for the message that's about to be sent. look: where
        their eyes were while they spoke (x, y), if gaze is calibrated."""
        if ui["screen"]:
            g = (look[0], look[1], max(60.0, gazer.radius)) if look and ui["gaze"] and gazer.ready else None
            screen["future"], screen["at"] = ctx_pool.submit(screen_read, g), time.monotonic()

    def screen_now():
        """(block for this message, full block): the screen summary, or one line if it hasn't changed."""
        if not ui["screen"]:
            return "", ""
        fut = screen["future"] if screen["future"] is not None and time.monotonic() - screen["at"] < 10 else None
        screen["future"] = None
        if fut is None:  # (typed, or nothing started early): where they looked in the last couple of seconds
            pt = gazer.point(time.monotonic() - 2.0, time.monotonic()) if ui["gaze"] and gazer.ready else None
            fut = ctx_pool.submit(screen_read, (pt[0], pt[1], max(60.0, gazer.radius)) if pt else None)
        try:
            ctx = fut.result(timeout=0.8)
        except Exception:
            ctx = ""
        if not ctx:
            return "", ""
        for line in ctx.splitlines():
            if line.startswith("Looking at"):
                print("[gaze] sent: " + line, flush=True)
        full = "Screen when they said it:\n" + ctx
        if ctx == screen["last"]:
            return "(Screen: unchanged since their previous message.)", full
        screen["last"] = ctx
        return full, full

    def log_usage(t, ev):
        """One line per Claude reply: tokens, model calls and tool calls (to measure what the screen summary costs
        and saves). /usage summarises it."""
        info = ev[3] if len(ev) > 3 else {}
        parts = [info] + ([info["pre"]] if info.get("pre") else [])
        tot = {k: sum((i.get("usage") or {}).get(k) or 0 for i in parts) for k in
               ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
        row = {"at": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "model": info.get("model", "?"),
               "escalated": bool(info.get("pre")), "screen": t.get("screen", 0),
               "fresh": tot["input_tokens"], "cache_write": tot["cache_creation_input_tokens"],
               "cache_read": tot["cache_read_input_tokens"], "out": tot["output_tokens"],
               "api_calls": sum(i.get("num_turns") or 0 for i in parts), "tools": t.get("tools", 0),
               "ms": round((time.monotonic() - t["sent"]) * 1000)}
        try:
            with open(usage_log, "a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass

    def on_notification(b, now):
        """A notification just appeared."""
        if b["app"] in ("Silent Chat", HOST) or not ui["notify"]:
            return
        line = notify.describe(b)
        ui["notifs"] = [n for n in ui["notifs"] if now - n["at"] < 180][-2:] + [{**b, "at": now, "line": line}]
        print(f"[notification] {line}", flush=True)
        if not ui["open"] and not ui["ambient"]:  # (the island shows them with the camera off too)
            return
        at = len(conversation) - 2 if turn is not None and turn.get("shown") else len(conversation)
        conversation.insert(max(0, at), ("Notification", line))  # (the reply streaming in stays last)
        pres.poke(now, 6)
        if settings.get("nudges", False) and not b["private"] and b.get("body"):  # (off by default)
            conversation.insert(max(0, at + 1), ("Notification", "Reply? Mouth or say “reply …”"))
            pres.poke(now, 10)
        if speaker.enabled and headphones() and not b["private"] and turn is None and cap is None \
                and not speaker.active:
            speaker.begin_reply()
            speaker.say(line.replace(" / ", ", ", 1))

    def notifications_note(now):
        """Recent notifications for Claude's next message (codes and passwords left out), once."""
        fresh = [n for n in ui["notifs"] if now - n["at"] < 180 and not n["private"]]
        ui["notifs"] = []
        if not fresh:
            return []
        lines = "; ".join(f"{n['line']} ({now - n['at']:.0f} s ago)" for n in fresh)
        return [f"(Notifications that just arrived, newest last: {lines})"]

    def setup_macctl(workdir):
        """Give the chat session ./macctl (its only command) and a screens/ folder it may read."""
        wrapper = Path(workdir) / "macctl"
        wrapper.write_text(f'#!/bin/zsh\nexec "{ROOT}/.venv/bin/python" "{ROOT}/mac.py" "$@"\n')
        wrapper.chmod(0o755)
        (Path(workdir) / "screens").mkdir(exist_ok=True)

    from claude_session import MODELS, Router
    if args.no_mac:
        session_args = {"system": LIVE_SYSTEM, "effort": args.effort}
    else:  # Claude may run only ./macctl and read only its own screenshots: the CLI refuses anything else
        found = {}
        if not args.no_connectors:  # Todoist / Spotify connected on your claude.ai account (connectors.py)
            disc.join(timeout=25)
            found = found_box or connectors.cached()
        mcp, extra_allowed, disallowed, extra_prompt = connectors.session_options(found)
        print(f"Connectors: {', '.join(found) or 'none'}.", flush=True)
        session_args = {"system": LIVE_SYSTEM + "\n" + MAC_CONTROL + ("\n" + extra_prompt if extra_prompt else ""),
                        "effort": args.effort, "tools": "Bash,Read,WebSearch",
                        "allowed": ["Bash(./macctl:*)", "Read(./screens/**)", "WebSearch", *extra_allowed],
                        "setup": setup_macctl, "mcp": mcp, "disallowed": disallowed}

    def make_session(model):
        """auto: Haiku answers, and hands hard requests to Opus. Or one model for everything."""
        print(f"Claude model: {'auto (Haiku first, Opus when needed)' if model == 'auto' else model}.", flush=True)
        if model == "auto":
            return Router(**session_args)
        return ClaudeSession(model=MODELS[model], **session_args)
    session = make_session(args.model or settings.get("model", "auto"))
    typed = queue.Queue()
    if args.commands:
        threading.Thread(target=fifo_reader, args=(typed, args.commands), daemon=True).start()
    else:
        threading.Thread(target=stdin_reader, args=(typed,), daemon=True).start()
    lip_pool, io_pool = cf.ThreadPoolExecutor(1), cf.ThreadPoolExecutor(1)

    conversation, pending, lip_jobs, notes = [], [], [], []
    hold = None  # Option held while hidden: dictation into the app in front
    hf = {"since": None, "shadow": None}  # hands-free: when mouthing began; a would-be capture (measuring)
    voice = {"quiet_since": 0.0, "listen_until": 0.0, "turn": False, "peek": False, "peek_until": None,
             "restore_voice": False}  # "Hey Claude" conversations
    ui["voice_pref"] = speaker.enabled  # VOICE in the menu: speak replies to mouthed / typed requests
    turn, last_turn, memory_sent = None, None, None
    recent_file = DATA / "recent.json"  # the last few exchanges, so a restart doesn't wipe the conversation
    resume = {"pending": True}

    def remember_exchange(said, reply):
        if not settings.get("remember", True):
            return
        try:
            items = json.loads(recent_file.read_text()) if recent_file.exists() else []
        except (OSError, ValueError):
            items = []
        items = (items + [{"at": time.time(), "you": (said or "")[:200], "claude": " ".join((reply or "").split())[:300]}])[-8:]
        try:
            recent_file.write_text(json.dumps(items))
        except OSError:
            pass

    def resume_note():
        """For the first message to a new Claude session: what you two were just talking about (last 2 hours)."""
        if not settings.get("remember", True):
            return None
        try:
            items = [i for i in json.loads(recent_file.read_text()) if time.time() - i["at"] < 7200]
        except (OSError, ValueError, KeyError):
            return None
        if not items:
            return None
        lines = "; ".join(f'you: "{i["you"]}" -> you replied: "{i["claude"][:160]}"' for i in items[-5:])
        return ("(Silent chat restarted, so this is a new session. Your recent exchanges with the user, oldest first: "
                + lines + ". Carry on from there if it's relevant.)")
    preroll, cap = collections.deque(), None
    smile_since, smile_deleted, smiling, gesture = None, 0, False, False
    eyes_open_at, closed_since, blink_fired, last_smile = 0.0, None, False, -10.0
    face_trace = collections.deque()  # (t, eye L, eye R, smile, jaw, cheek squint, brows) for the last 3 s
    face_log = DATA / "gestures.log"

    def log_face(event, now):
        """Print and save what the face was doing just before a gesture fired, so thresholds can be tuned."""
        recent = [r for r in face_trace if now - r[0] <= 1.5]
        if not recent:
            return
        cols = np.array([r[1:] for r in recent])
        last, peak = cols[-1], cols.max(axis=0)
        names = ("eyeL", "eyeR", "smile", "jaw", "cheek", "brows")
        summary = "  ".join(f"{n} {v:.2f}/{m:.2f}" for n, v, m in zip(names, last, peak))
        print(f"[face] {event}: now/peak(1.5 s)  {summary}", flush=True)
        with open(face_log, "a") as f:
            f.write(f"\n# {datetime.now():%Y-%m-%d %H:%M:%S} {event} (TYPE={ui['type']})\n")
            f.write("t      " + "  ".join(f"{n:>5s}" for n in names) + "\n")
            for r in recent:
                f.write(f"{r[0] - now:+.2f}  " + "  ".join(f"{v:5.2f}" for v in r[1:]) + "\n")
    send_since, send_fired, last_capture_end = None, False, 0.0
    send_name = "eyebrow raise" if args.send == "brows" else "wink"
    notice = f"Hold {hotkey.label} while you mouth" + (", or blink long to start." if ui["face"] else ".")
    print(f"Ready ({which}, effort {args.effort}). Hold {hotkey.label} while you silently mouth a message, then let "
          "go.\nType here to send text, or /fix <text>, /learn <words>, /voice, /face, /type, /mic, /hide, /show, /quit.",
          flush=True)
    if args.type:
        on_button("type")

    stop_file = DATA / "agent_stop"  # while it exists, every macctl action is refused (control.py)

    def stop_agent(why):
        """Stop Claude acting on the Mac, right now: macctl refuses further actions and the turn is interrupted."""
        nonlocal notice
        stop_file.write_text(why)
        session.interrupt()
        speaker.stop()
        notice = "Stopped."
        print(f"[agent] stopped ({why})", flush=True)

    def set_open(flag):
        nonlocal cap, notice, hold
        if not flag and voice["peek"]:  # hide the "Hey Claude" pop-up
            voice["peek"], voice["peek_until"] = False, None
            conversation.clear()
            if not ui['ambient']:  # (auto-hide: the island shows everything; the old panel stays away)
                panel.hide()
        if flag == ui["open"]:
            return
        if flag:
            voice["peek"], hold = False, None
        conversation.clear()  # fresh display each time it opens or closes (Claude still remembers the conversation)
        notice = ""
        if turn is not None:
            turn["shown"] = False  # a reply still streaming in starts a new line in the cleared display
        if flag:
            try:
                cam.resume()
            except RuntimeError:  # not allowed yet: the loop keeps trying while it's shown
                print(f"No camera yet: allow {HOST} under Privacy & Security > Camera.", flush=True)
            if ui["mic"]:
                mic_on(True)
            if not ui['ambient']:  # (auto-hide: the island shows everything; the old panel stays away)
                panel.unhide()
            if reader.reload_if_newer():  # a background retrain kept a better model while you were away
                notice = "Now using your newly retrained lip-reader."
                print(notice, flush=True)
            preroll.clear()
            notice = "" if cam.cap is not None else (f"Waiting for the camera: allow {HOST} under System Settings > "
                                                     "Privacy & Security > Camera.")
        else:
            speaker.stop()
            cap = None
            preroll.clear()
            mic_on(False)  # mic closed while hidden
            if not ui['ambient']:  # (auto-hide: the island shows everything; the old panel stays away)
                panel.hide()
            cam.pause()  # camera light off
            if not args.no_train:  # you're not using it: a good moment to retrain on your clips, if there are enough
                import train_personal
                ok, _ = train_personal.should_train()
                if ok:
                    print(train_personal.train_in_background(DATA / "training.log"), flush=True)
        ui["open"] = flag
        if menu is not None:
            menu.set_shown(flag)
        print("Shown." if flag else f"Hidden (camera off). Press {key_label} to show it again; hold "
              f"{hotkey.label} to dictate into any app" + (', or say "Hey Claude".' if wake.enabled else "."),
              flush=True)

    key_label = f"'{args.toggle_key}'"
    toggle, menu, retry = None, None, {"key": 0.0, "camera": 0.0}
    if args.toggle_key != "none":
        typer.trusted(prompt=True)
        typer.request_listen_access()
        toggle = typer.ToggleKey(args.toggle_key)
        if toggle.ok:
            print(f"Press {key_label} anywhere to show/hide silent chat (double-tap it to type {key_label}).",
                  flush=True)
        else:
            print(f"Can't use the {key_label} key yet: allow {HOST} under System Settings > Privacy & "
                  "Security > Accessibility AND Input Monitoring (it starts working by itself). Until then use "
                  "the menu bar icon.", flush=True)
    from menubar import StatusMenu
    def settings_state():
        return {"open": ui["open"], "type": ui["type"], "voice": ui["voice_pref"], "face": ui["face"],
                "mic": ui["mic"], "wake": wake.enabled, "model": settings.get("model", args.model or "auto"),
                "handsfree": ui["handsfree"], "screen": ui["screen"], "notify": ui["notify"], "lens": ui["lens"],
                "ambient": ui["ambient"], "gaze": ui["gaze"], "warp": ui["warp"],
                "improve": improve.load().get("enabled", True), "clips": SAVE_CLIPS[0],
                "remember": settings.get("remember", True), "clicks": gazer.learn_clicks,
                "nudges": settings.get("nudges", False)}
    menu = StatusMenu(typed.put, settings_state, args.toggle_key)
    import settings_window

    def permission_checks():
        import shutil
        import objc
        import ApplicationServices
        import Quartz
        from Foundation import NSBundle
        NSBundle.bundleWithPath_("/System/Library/Frameworks/AVFoundation.framework").load()
        av = objc.lookUpClass("AVCaptureDevice")
        return [
            ("Camera", "To see your mouth (lip-reading) and where you look.",
             lambda: av.authorizationStatusForMediaType_("vide") == 3, "Privacy_Camera"),
            ("Microphone", "For “Hey Claude” and speaking out loud.",
             lambda: av.authorizationStatusForMediaType_("soun") == 3, "Privacy_Microphone"),
            ("Accessibility", "To read the window in front and act in apps for you.",
             lambda: ApplicationServices.AXIsProcessTrusted(), "Privacy_Accessibility"),
            ("Input Monitoring", f"For the {key_label} key and holding Option.",
             lambda: bool(toggle and toggle.ok), "Privacy_ListenEvent"),
            ("Screen Recording", "Optional: for the glass lens and screenshots Claude asks for.",
             lambda: Quartz.CGPreflightScreenCaptureAccess(), "Privacy_ScreenCapture"),
            ("Claude Code", "Signed in with your claude.ai plan (claude in Terminal).",
             lambda: shutil.which("claude") is not None, "https://claude.com/claude-code"),
        ]
    win = {"settings": None, "setup": None}

    def open_window(which):
        if which == "settings":
            win["settings"] = win["settings"] or settings_window.SettingsWindow(typed.put, settings_state)
            win["settings"].show()
        else:
            win["setup"] = win["setup"] or settings_window.SetupWindow(
                permission_checks(), typed.put, lambda: save_setting("setup_done", True))
            win["setup"].show()
    if not settings.get("setup_done") and not settings.get("model") and not settings.get("handsfree"):
        open_window("setup")  # (first run)
    if args.mic:
        on_button("mic")
    if settings.get("wake", True) and not args.no_wake:
        set_wake(True)
    if args.hidden or settings.get("restart_hidden"):  # (a restart keeps it hidden if it was)
        set_open(False)
    if "restart_hidden" in settings:
        save_setting("restart_hidden", False)

    def start_hold(now):
        nonlocal hold
        hold = {"start": now, "items": [], "release_at": None, "mouthing": 0,
                "keys": toggle.last_key if toggle else 0.0}
        try:
            mic_on(True)
        except Exception as e:
            print(f"(no microphone: {e})", flush=True)
        cam.open_async()  # the camera only runs while you hold Option (in case you mouth instead of speaking)
        speaker.stop()
        print(f"[hold] {hotkey.label} held: dictating into {typer.frontmost()[2]}", flush=True)

    def finish_hold(now):
        nonlocal hold, notice
        h, hold = hold, None
        end = h["release_at"] or now
        if not ui["open"]:
            cam.close_async()
        audio = mic.segment(h["start"] - 0.3, end + 0.15) if mic.on else None
        if not wake.enabled:
            mic_on(False)
        if end - h["start"] < 0.5 or (toggle and toggle.last_key > h["keys"]):
            return  # a tap, or Option as part of a shortcut (Option+arrow, Option+letter ...): not dictation
        if (audio is None or speech.voiced(audio) < 0.15) and h["mouthing"] < 6:
            notice = "Didn't hear or see anything to type."
            print(f"[hold] {notice}", flush=True)
            return
        clip = DATA / "clips" / f"{datetime.now():%Y%m%d_%H%M%S_%f}.mp4"
        job = lip_pool.submit(read_hold, reader, transcriber["model"], [f for _, f in h["items"]],
                              [t for t, _ in h["items"]], clip, io_pool, audio, top_vocab())
        lip_jobs.append((job, "hold"))

    def type_now(text):
        """Type what you said out loud straight into the app in front (no Claude round trip needed). Claude is told
        about it with your next message, so "why didn't you do that?" makes sense to it."""
        nonlocal notice
        bundle, pid, app = typer.frontmost()
        if not typer.trusted():
            notice = f"Can't type yet: allow {HOST} under Privacy & Security > Accessibility."
            return
        if pid != dictation["pid"]:
            dictation.update(pid=pid, app=app, field="", last="", words=0, last_clips=[])
        last = dictation["last"]
        prefix = " " if last and not last[-1].isspace() and text[:1] not in " \n.,;:!?)'\"" else ""
        typer.type_text(prefix + text)
        dictation.update(field=dictation["field"] + prefix + text, last=prefix + text, deleted=False,
                         words=dictation["words"] + len(text.split()), last_clips=[])
        conversation.append(("Typed", text))
        amb.setdefault("dictated", []).append((app, text[:300]))
        print(f"Typed (said out loud) into {app}: {text}", flush=True)

    local_done = queue.Queue()
    commands_log = DATA / "commands.log"

    def closed_set_candidates():
        """What can be done instantly right now: fixed commands, running apps, and the front window's controls."""
        import control
        from AppKit import NSRunningApplication, NSWorkspace
        bundle, pid, _ = typer.frontmost()
        nodes = []
        try:
            a = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
            _, raw = control.walk(a, budget_s=0.4)
            nodes = [{"role": n["role"], "label": control.label(n)} for n in raw if n["role"] in control.ROLES]
        except Exception:
            pass
        apps = [a.localizedName() for a in NSWorkspace.sharedWorkspace().runningApplications()
                if a.activationPolicy() == 0 and a.localizedName()][:15]
        names = [local_commands._label(n["label"]) for n in nodes if n["role"] in local_commands.OPEN_ROLES]
        return local_commands.candidates(bundle, nodes, apps, shortcut_candidates([n for n in names if n][:10]))

    def shortcut_candidates(names=()):
        return shortcuts.candidates(names)

    def closed_set_accept(c):
        """Instant only when the best candidate fits about as well as the free reading, and clearly beats the
        runner-up; otherwise the normal path decides (and asks Claude)."""
        return c["score"] >= c["free"] - CLOSED_MARGIN and c["score"] - c["second"] >= CLOSED_GAP

    def try_closed_set(result):
        """If a captured clip clearly matches an instant command, run it. True if it did."""
        c = result.get("closed")
        if not c or ui.get("confirm"):  # (a waiting "go ahead?" is answered by the yes/no check instead)
            return False
        ok = closed_set_accept(c)
        with open(commands_log, "a") as fh:
            fh.write(json.dumps({"time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "clip": str(result["clip"]),
                                 "free_text": result["hyps"][0][0], **c, "accepted": ok}) + "\n")
        if not ok:
            return False
        kind, what = c["action"]
        part = {"typed": c["phrase"], "clip": result["clip"], "hyps": result["hyps"]}
        if kind == "shortcut":
            phrase, values = what
            found = next((s for s in shortcuts.load() if s["phrase"] == phrase), None)
            if not found:
                return False
            replay(found, values, part)
        elif kind == "press":
            run_local(local_commands.Command("macctl", ["press", what], f"Pressed {what}."), part)
        else:
            cmd = local_commands.parse(what, typer.frontmost()[0])
            if not cmd:
                return False
            run_local(cmd, part)
        print(f"[closed-set] {c['phrase']!r} (score {c['score']:.1f}, runner-up {c['runner_up']!r} "
              f"{c['second']:.1f}, free reading {c['free']:.1f}; {c['n']} candidates in {c['ms']} ms)", flush=True)
        return True

    def run_local(cmd, part):
        """Carry out an everyday command without Claude (local_commands.py); the result shows up via local_done."""
        heard = part.get("typed") or part.get("spoken") or part["hyps"][0][0]
        if part.get("clip") and part.get("hyps"):
            labels.add(part["clip"], heard, "heard", "local", part["hyps"])  # still training data
            amb.setdefault("local_clips", []).append((part["clip"], time.monotonic()))
        conversation.extend([("You", heard), ("Claude", "...")])
        stop_file.unlink(missing_ok=True)
        speak = bool(cmd.say) and (speaker.enabled or bool(part.get("voice")))
        if cmd.kind == "action":
            done = apply_action(cmd.args, from_chat=True)
            local_done.put((heard, cmd.say or f"Done: {done or cmd.args}.", cmd.say if speak else ""))
            return

        def go():
            work = DATA / "macctl_local"
            work.mkdir(exist_ok=True)
            r = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "mac.py"), *cmd.args],
                               cwd=work, capture_output=True, text=True, timeout=30)
            out = (r.stdout.strip().splitlines() or [""])[-1]
            ok = r.returncode == 0 and not out.startswith(("FAILED", "Couldn't", "No "))
            reply = (cmd.say or out or "Done.") if ok else (out or "That didn't work.")
            local_done.put((heard, reply, (cmd.say if ok else reply) if speak else ""))
        threading.Thread(target=go, daemon=True).start()

    def replay(shortcut, values, part):
        """Run a saved shortcut on the Mac. Steps marked ask: wait for a yes; if a step fails, Claude takes over."""
        heard = part.get("typed") or part.get("spoken") or part["hyps"][0][0]
        steps = shortcuts.fill(shortcut["steps"], values)
        ask = next((i for i, st in enumerate(steps) if st.startswith("ask:")), None)
        now_steps, later = (steps, []) if ask is None else (steps[:ask], [steps[ask][4:]] + steps[ask + 1:])
        if part.get("clip") and part.get("hyps"):
            labels.add(part["clip"], heard, "heard", "shortcut", part["hyps"])
            amb.setdefault("local_clips", []).append((part["clip"], time.monotonic()))
        conversation.extend([("You", heard), ("Claude", "...")])
        stop_file.unlink(missing_ok=True)
        speak = speaker.enabled or bool(part.get("voice"))

        def go():
            work = DATA / "macctl_local"
            work.mkdir(exist_ok=True)
            r = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "mac.py"), "do", *now_steps],
                               cwd=work, capture_output=True, text=True, timeout=60) if now_steps else None
            if r is not None and r.returncode != 0:
                lines = [l for l in r.stdout.splitlines() if l.startswith("FAILED")]
                local_done.put((heard, "My saved shortcut didn't work this time, so Claude is taking over.", "",
                                {"handover": heard, "why": (lines or ["a step failed"])[0]}))
                shortcuts.failed(shortcut["phrase"])
                improve.log("shortcut", said=heard, phrase=shortcut["phrase"], ok="no", why=(lines or [""])[0][:160])
                return
            shortcuts.used(shortcut["phrase"], values)
            improve.log("shortcut", said=heard, phrase=shortcut["phrase"], ok="yes")
            if later:
                question = f"Ready: {later[0]}. Go ahead? (yes / no)"
                local_done.put((heard, question, "Go ahead?" if speak else "", {"confirm": later}))
            else:
                local_done.put((heard, "Done.", "Done." if speak else "", {}))
        threading.Thread(target=go, daemon=True).start()
        print(f"[shortcut] {shortcut['phrase']!r} {values}", flush=True)

    def confirm_answer(text):
        """yes / no to a waiting 'ask first' step of a shortcut. Returns True if it was an answer."""
        t = local_commands.clean(text)
        yes = re.match(r"(yes|yeah|yep|sure|do it|send it|go ahead|ok|okay|correct|confirm)\b", t)
        no = re.match(r"(no|nope|cancel|stop|don't|never ?mind|wait)\b", t)
        if not (yes or no) or not ui.get("confirm"):
            return False
        steps, ui["confirm"] = ui["confirm"]["steps"], None
        if no:
            local_done.put((text, "Cancelled.", "Cancelled." if speaker.enabled else "", {}))
            return True
        replay({"phrase": "(confirmed)", "steps": steps}, {}, {"typed": text})
        return True

    def apply_fix(text, tell_claude):
        nonlocal notice
        if not last_turn:
            notice = "Nothing to correct yet."
            return
        labels.update(last_turn["clips"], text=text, status="fixed")
        new = memory.learn_correction(last_turn["top1"], last_turn["heard"], text)
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i][0] == "You":
                conversation[i] = ("You", f"{text} (fixed)")
                break
        if tell_claude:
            notes.append(f"(Correction: my previous message was actually: \"{text}\")")
        last_turn.update(heard=text, fixed=True)
        notice = f"Learned: \"{text}\"" + (f" (new words: {', '.join(new)})" if new else "")
        print(notice, flush=True)

    def start_capture(mode, now):
        nonlocal cap, preroll
        cap = {"mode": mode, "items": list(preroll), "start": now, "release_at": None, "trigger_end": None,
               "speech_start": None, "last_active": None, "above": 0}
        preroll = collections.deque()
        interrupted = speaker.stop()
        if interrupted is not None:  # barge-in
            notes.append(f"(I interrupted your last reply after you'd said about {interrupted[0]} of "
                         f"{interrupted[1]} words.)")
        if turn is not None:
            turn["muted"] = True  # don't speak the rest of a reply that's still streaming in
        if mode == "blink" and speaker.enabled:  # your eyes were shut, so a soft sound says it's listening
            subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])

    def finish_capture(now, lo=None, hi=None):
        nonlocal cap, notice, last_capture_end
        items = [(t, f) for t, f in cap["items"] if (lo is None or t >= lo) and (hi is None or t <= hi)]
        mouthed = (cap["last_active"] or now) - (cap["speech_start"] or now)
        mode, cap, last_capture_end = cap["mode"], None, now
        if mode == "auto" and mouthed < HF_MIN:
            hf_note("auto-dropped", f"mouthed {mouthed:.2f}s")  # too short to be a message: quietly ignored
            return
        if len(items) < 12:
            if mode != "auto":
                notice = f"Too short: hold {hotkey.label} for the whole message."
            return
        clip = DATA / "clips" / f"{datetime.now():%Y%m%d_%H%M%S_%f}.mp4"
        audio = mic.segment(items[0][0] - 0.15, items[-1][0] + 0.15) if mic.on and ui["mic"] else None
        vocab = sorted(memory.data["vocabulary"], key=lambda w: -memory.data["vocabulary"][w])[:40]
        cands = None if ui["type"] or args.no_local else closed_set_candidates
        sw = set(re.findall(r"[A-Z][a-z]{2,}|[a-z]{4,}", (screen.get("last") or "")))  # names, labels in front
        sw |= {w for w in vocab if len(w) >= 3}
        job = lip_pool.submit(lipread, reader, [f for _, f in items], [t for t, _ in items], clip, io_pool,
                              audio, transcriber["model"], vocab, cands, sorted(sw)[:120])
        lip_jobs.append((job, ("auto:" if mode == "auto" else "") + ("type" if ui["type"] else "chat")))
        if not ui["type"]:
            screen_soon(gazer.point(items[0][0] - 1.0, items[-1][0] + 0.2))  # (from just before they started)

    def can_edit_field():
        bundle, pid, app = typer.frontmost()
        return pid == dictation["pid"] and bundle not in typer.TERMINALS and typer.trusted()

    def delete_dictated_word():
        """Option+Delete the last word you dictated into the app in front. False if there's nothing of yours."""
        nonlocal notice
        if dictation["words"] <= 0 or not can_edit_field():
            notice = "Nothing you dictated here to delete."
            return False
        typer.delete_word()
        log_face("smile deleted a word", time.monotonic())
        gone = dictation["field"].rstrip().rsplit(" ", 1)
        dictation.update(field=(gone[0] + " ") if len(gone) > 1 else "", last=" ",
                         words=dictation["words"] - 1, deleted=True)
        notice = "Deleted a word." + (" Keep smiling to delete more." if dictation["words"] else "")
        return True

    def send_dictated():
        """Enter, if you've dictated something into the app in front that hasn't been sent yet."""
        nonlocal notice
        if not dictation["field"].strip() or not can_edit_field():
            notice = f"Nothing dictated here to send ({send_name} = Enter)."
            return
        typer.press_return()
        log_face(f"{send_name} pressed Enter", time.monotonic())
        if dictation["last_clips"] and not dictation["deleted"]:
            labels.update(dictation["last_clips"], only_status="heard", status="ok")  # you sent it as it was
        dictation.update(field="", last="", words=0, last_clips=[], deleted=False)
        conversation.append(("Typed", "(sent)"))
        notice = "Sent."
        print("Sent (Enter).", flush=True)

    def app_state():
        return (f"(App state: {'TYPE' if ui['type'] else 'CHAT'} mode, voice {'on' if speaker.enabled else 'off'}, "
                f"face gestures {'on' if ui['face'] else 'off'}, mic {'on' if ui['mic'] else 'off'}, "
                f"Hey Claude {'on' if wake.enabled else 'off'}, hands-free {'on' if ui['handsfree'] == 'on' else 'off'}.)")

    def apply_action(text, from_chat=False):
        """Carry out one 'ACTION: ...' line from Claude. Returns what changed, or ''."""
        m = re.search(r"\b(type|typing|voice|face|mic|wake|hands-?\s?free|hide)\b\W*(on|off)?", text.lower())
        if not m:
            return ""
        name, value = m.group(1), m.group(2)
        name = "handsfree" if name.startswith("hands") else {"typing": "type"}.get(name, name)
        if name == "hide":
            ui["toggle"] = False
            return "hidden"
        current = {"type": ui["type"], "voice": speaker.enabled, "face": ui["face"], "mic": ui["mic"],
                   "wake": wake.enabled, "handsfree": ui["handsfree"] == "on"}[name]
        want = (not current) if value is None else value == "on"
        if want != current:
            ui["from_chat"] = from_chat
            on_button(name)
            ui["from_chat"] = False
        label = {"type": "TYPE mode", "voice": "voice", "face": "face gestures", "mic": "MIC",
                 "wake": "Hey Claude", "handsfree": "hands-free"}[name]
        print(f"[action] {label} {'on' if want else 'off'}", flush=True)
        return f"{label} {'on' if want else 'off'}"

    def input_states(now, level, smile_squint):
        """What each face input is doing right now, for the overlay's EYES / MOUTH / BROWS tiles."""
        b = activity.blend
        if activity.landmarks is None or not b:
            return None
        eyes = (b.get("eyeBlinkLeft", 0) + b.get("eyeBlinkRight", 0)) / 2
        if cap is not None and cap["mode"] == "blink":
            eye_progress = 1.0
        elif closed_since and not smile_squint and ui["face"]:
            eye_progress = (now - closed_since) / BLINK_HOLD
        else:
            eye_progress = 0.0
        eye_state = "CLOSED" if eyes > BLINK_CLOSED else "HALF" if eyes > BLINK_OPEN else "OPEN"
        smile = (b.get("mouthSmileLeft", 0) + b.get("mouthSmileRight", 0)) / 2
        jaw = b.get("jawOpen", 0)
        talking = level is not None and level > cal["on"] and smile < SMILE_ON
        if smile > SMILE_ON and jaw < JAW_MAX:
            mouth = "DELETING" if ui["type"] and smile_deleted > 0 else "SMILE"
        else:
            mouth = "TALKING" if talking else "OPEN" if jaw > 0.25 else "CLOSED"
        inner = b.get("browInnerUp", 0)
        brows_up = inner > BROWS_UP and (b.get("browOuterUpLeft", 0) + b.get("browOuterUpRight", 0)) / 2 > 0.8 * BROWS_UP
        brow_progress = (now - send_since) / BROWS_HOLD if send_since and args.send == "brows" else 0.0
        return {
            "eyes": {"state": eye_state, "closure": min(1.0, max(0.0, (eyes - 0.25) / (BLINK_CLOSED - 0.2))),
                     "progress": eye_progress, "active": eye_state != "OPEN" or eye_progress > 0},
            "mouth": {"state": mouth, "openness": min(1.0, max(jaw / 0.4, 0.35 if talking else 0.0)),
                      "active": mouth != "CLOSED"},
            "brows": {"state": "UP" if brows_up else "DOWN", "lift": min(1.0, inner / 0.6),
                      "progress": brow_progress, "active": brows_up},
        }

    def dictate_flush(t, final):
        """Type the dictated text as it streams in: whole words only, the last word once the reply is complete."""
        quotes = '"\u201c\u201d'
        if not t["started"]:
            head = t["buf"].lstrip().lstrip(quotes)
            upper = head.upper()
            if upper.startswith("ACTION") or (upper and "ACTION:".startswith(upper[:7])):
                if final:  # a command to the app, not text to type
                    t["actions"] = [l for l in head.splitlines() if l.strip().upper().startswith("ACTION")]
                    t["buf"] = ""
                return
            if head.strip().strip(quotes) in ("", "?"):
                if final:
                    t["buf"] = ""
                return
            if not head.startswith(TYPE_MARK):
                if not final and TYPE_MARK.startswith(head):
                    return  # the marker is still arriving
                # anything else is Claude talking, not a transcription: never type it
                t["started"], t["blocked"] = True, "Claude replied instead of transcribing, so nothing was typed."
                print(f"[dictate] not typed (no {TYPE_MARK.strip()!r}): {head[:80]!r}", flush=True)
                return
            head = head[len(TYPE_MARK):].lstrip().lstrip(quotes)
            if not final and " " not in head:
                return  # wait for a whole first word
            if final and not head.strip():
                t["buf"] = ""
                return
            bundle, pid, app = typer.frontmost()
            if bundle in typer.TERMINALS and HOST == "Terminal":  # (that Terminal window is silent chat's console)
                t["blocked"] = "Click into the app you want to type into (not Terminal), then mouth it again."
            elif not typer.trusted():
                t["blocked"] = f"Can't type yet: allow {HOST} under Privacy & Security > Accessibility."
            t["started"], t["buf"] = True, head
            t.setdefault("first", time.monotonic() - t["sent"])
            if t["blocked"]:
                return
            if pid != dictation["pid"]:
                dictation.update(pid=pid, app=app, field="", last="", words=0, last_clips=[])
            last = dictation["last"]
            if last and not last[-1].isspace() and head[:1] not in " \n.,;:!?)'\"":
                t["prefix"] = " "
        if t["blocked"]:
            return
        buf = t["buf"]
        cut = len(buf) if final else buf.rfind(" ") + 1
        chunk = buf[:cut].rstrip().rstrip(quotes) if final else buf[:cut]
        if chunk:
            typer.type_text(t["prefix"] + chunk)
            t["typed"] += t["prefix"] + chunk
            t["prefix"] = ""
        t["buf"] = buf[cut:]

    def emit_reply(t, text):
        if not t["shown"]:
            conversation.extend([("You", "..."), ("Claude", "")])
            t["shown"] = True
        t["reply"] += text
        conversation[-1] = ("Claude", t["reply"].strip())
        t["speech"] += text
        speak_ready(t)

    def speak_ready(t, final=False):
        buf, cut = t["speech"], None
        for m in SENTENCE_END.finditer(buf):
            cut = m.end()
        if cut is None and len(buf) > 160 and buf.rfind(", ") > 40:  # long run-on sentence: break at a comma
            cut = buf.rfind(", ") + 2
        if final:
            cut = len(buf)
        if cut:
            chunk, t["speech"] = buf[:cut], buf[cut:]
            if not t["muted"]:
                speaker.say(chunk)

    def feed(t, delta):
        """Parse the streamed reply: HEARD / FIX lines first, then the reply, spoken as sentences complete."""
        t["buf"] += delta
        while True:
            if t["in_reply"]:
                if "\n" in t["buf"]:
                    seg, t["buf"] = t["buf"].split("\n", 1)
                    emit_reply(t, seg + " ")
                    t["in_reply"] = False
                    continue
                if t["buf"]:
                    emit_reply(t, t["buf"])
                    t["buf"] = ""
                return
            if "\n" in t["buf"]:
                line, t["buf"] = t["buf"].split("\n", 1)
                head = line.strip()
                if head.upper().startswith("HEARD:") and t["heard"] is None:
                    t["heard"] = head.split(":", 1)[1].strip()
                    shown = t["heard"] if t["heard"] and t["heard"] != "?" else "(couldn't tell)"
                    conversation.extend([("You", shown), ("Claude", "")])
                    t["shown"] = True
                elif head.upper().startswith("FIX:") and t["fix"] is None:
                    t["fix"] = head.split(":", 1)[1].strip()
                elif head.upper().startswith(("HEARD:", "FIX:")):
                    pass  # repeated after a tool call: not part of the reply
                elif head.upper().startswith("ACTION:"):
                    apply_action(head.split(":", 1)[1], from_chat=True)
                elif head:
                    emit_reply(t, head + " ")
                continue
            head = t["buf"].lstrip().upper()
            if head and not head.startswith(("HEARD:", "FIX:", "ACTION:")) and not "HEARD:".startswith(head[:6]) \
                    and not "FIX:".startswith(head[:4]) and not "ACTION:".startswith(head[:7]):
                t["in_reply"] = True
                continue
            return

    def finish_turn(t, result, is_error):
        nonlocal last_turn, notice
        if is_error:
            notice = f"Claude error: {result[:80]}"
            return
        if not t["shown"] and not t["reply"]:
            feed(t, result)  # nothing streamed: parse the final text instead
        feed(t, "\n")
        speak_ready(t, final=True)
        if t["reply"].strip().endswith("?"):
            ui["question_at"] = time.monotonic()  # a nod / head shake can answer it for the next 25 s
        heard, fix = t["heard"] or "", t["fix"]
        remember_exchange(heard or " / ".join((p.get("typed") or p.get("spoken") or "") for p in t["parts"]),
                          t["reply"])
        if fix:
            apply_fix(fix, tell_claude=False)
        elif last_turn and not last_turn.get("fixed"):
            labels.update(last_turn["clips"], only_status="heard", status="ok")  # you didn't object
        ok = bool(heard) and heard != "?"
        clip_parts = [p for p in t["parts"] if p["clip"]]
        for p in clip_parts:
            if p.get("spoken"):
                labels.add(p["clip"], p["spoken"], "audio", "mic", p["hyps"])
                learned = memory.learn_from_speech(p["spoken"])
                if learned:
                    notice = f"Learned: {', '.join(learned)}"
                continue
            single = ok and len(t["parts"]) == 1
            labels.add(p["clip"], heard if single else "", "heard" if single else "unlabelled",
                       "live" if single else "live_multipart", p["hyps"])
        last_turn = ({"clips": [p["clip"] for p in clip_parts], "heard": heard, "top1": clip_parts[0]["hyps"][0][0]}
                     if ok and len(t["parts"]) == 1 and clip_parts else None)
        top1s = " | ".join(("(said) " + p["spoken"]) if p.get("spoken") else p["hyps"][0][0] for p in clip_parts)
        print(f"{'lip-reader: ' + top1s + chr(10) if top1s else ''}Heard:  {heard or '(none)'}\n"
              f"Claude: {t['reply'].strip()}  [first words {t.get('first', 0):.1f} s, done {time.monotonic() - t['sent']:.1f} s]\n",
              flush=True)

    try:
        while not ui["quit"]:
            panel.pump()
            for action in (toggle.poll() if toggle else []):
                if action == "toggle" and turn is not None and turn.get("acting"):
                    stop_agent("\\ pressed")  # while Claude is working on your Mac, \ means stop (not hide)
                elif action == "toggle":  # \ turns the camera on / off (hover the island to type)
                    set_open(not ui["open"])
                else:  # double tap: type the key itself
                    typer.type_text(args.toggle_key)
            if ui["toggle"] is not None:
                set_open(ui["toggle"])
                ui["toggle"] = None
            now = time.monotonic()
            if args.toggle_key != "none" and not (toggle and toggle.ok) and now - retry["key"] > 3:
                retry["key"] = now  # permissions granted after start-up: the key starts working without a restart
                toggle = typer.ToggleKey(args.toggle_key)
                if toggle.ok:
                    notice = f"{key_label} now shows/hides silent chat anywhere."
                    print(notice, flush=True)
            # ---- camera off while you're away: no face for 2 min and no typing / mouse for 1 min, or on battery
            # with nothing touched for 3 min. Back on the moment you touch the keyboard or mouse.
            if now - amb.get("power_at", -99) > 60:
                amb["power_at"] = now
                try:
                    amb["battery"] = "Battery Power" in subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                                                                       text=True, timeout=2).stdout
                except Exception:
                    amb["battery"] = False
            if now - amb.get("away_check", 0) > 1.0:
                amb["away_check"] = now
                import Quartz
                idle_input = Quartz.CGEventSourceSecondsSinceLastEventType(
                    Quartz.kCGEventSourceStateCombinedSessionState, Quartz.kCGAnyInputEventType)
                if activity.landmarks is not None or (cam.cap is None and not amb.get("away")):
                    amb["face_at"] = now  # (a face seen, or the camera is off for another reason: not "away")
                busy_now = turn is not None or cap is not None or hold is not None or bool(pending)
                if ui["open"] and not amb.get("away") and cam.cap is not None and not busy_now:
                    no_face = now - amb.get("face_at", now)
                    if (no_face > 120 and idle_input > 60) or (amb.get("battery") and idle_input > 180 and
                                                               now - imp["last_msg"] > 180):
                        amb["away"] = True
                        cam.pause()
                        print(f"[camera] paused: {'no one here' if no_face > 120 else 'idle on battery'} "
                              "(back on when you touch the keyboard or mouse)", flush=True)
                elif amb.get("away") and (idle_input < 1.5 or not ui["open"]):
                    amb["away"], amb["face_at"] = False, now
                    if ui["open"]:
                        print("[camera] back on", flush=True)
            if ui["open"] and not amb.get("away") and cam.cap is None and now - retry["camera"] > 3:
                retry["camera"] = now
                try:
                    cam.resume(wait_seconds=0)
                    notice = ""
                except RuntimeError:
                    pass
            if (ui["open"] or hold is not None) and cam.cap is not None:
                frame = cam._read()
                now = time.monotonic()
                level = activity(np.ascontiguousarray(frame[..., ::-1]), now)
                while not clicks_q.empty():
                    note = gazer.add_click(*clicks_q.get())
                    if note:
                        print("[gaze] " + note, flush=True)
                gazer.update(now, activity.landmarks, activity.matrix, activity.blend,
                             airpods.latest() if airpods is not None else None)
                if gazer.calib is not None:
                    r = gazer.step_calibration(now)
                    if r is not None:
                        notice = "Gaze: " + gaze_mod.describe(r) + ("" if "error" in r else ". Saved.")
                        print("[gaze] calibrated: " + gaze_mod.describe(r), flush=True)
                        if "error" not in r:
                            gazer.show(8)
                if ui.get("gaze_where") and now >= ui["gaze_where"][0]:  # /gaze where: a dry run, presses nothing
                    ui["gaze_where"].pop(0)
                    pt = gazer.point(now - 0.8, now)
                    subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])
                    if pt is None:
                        print("[gaze] where: no gaze (face not seen, or eyes closed)", flush=True)
                    else:
                        gazer.show(1.5)
                        r = max(60.0, gazer.radius)
                        ctx_pool.submit(lambda x=pt[0], y=pt[1], r=r: print(
                            f"[gaze] where ({x:.0f}, {y:.0f}): " + (control.looking_at(x, y, r) or "nothing readable"),
                            flush=True))
                # ignore the talk key just after we pressed keys ourselves (e.g. Option+Delete when smiling to delete)
                down = hotkey.held() and now - typer.LAST_SYNTHETIC[0] > 0.5
            else:  # hidden (camera off): only Option dictation and "Hey Claude"; replies in flight still finish
                time.sleep(0.03)
                frame, level = None, None
                gazer.hide_lens()
                down = (not ui["open"] and not args.no_hold_dictation and hotkey.held()
                        and now - typer.LAST_SYNTHETIC[0] > 0.5)
                activity.blend, activity.landmarks = {}, None
                if gazer.calib is not None:  # (hidden mid-calibration: cancelled)
                    gazer.cancel_calibration()
                    print("[gaze] calibration cancelled", flush=True)
            b = activity.blend
            smile = (b.get("mouthSmileLeft", 0) + b.get("mouthSmileRight", 0)) / 2
            smiling = smile > SMILE_ON and b.get("jawOpen", 1) < JAW_MAX
            eye_l, eye_r = b.get("eyeBlinkLeft", 0), b.get("eyeBlinkRight", 0)
            eyes = (eye_l + eye_r) / 2
            # long blink: both eyes close quickly (not slowly drooping lids from looking down) and stay shut ~0.5 s.
            # A real smile squints the eyes (cheeks push the lids up), so eye closure doesn't count while you're
            # smiling or just after: otherwise smiling to delete words would start listening instead.
            if smile > SMILE_OFF or b.get("cheekSquintLeft", 0) + b.get("cheekSquintRight", 0) > 0.8:
                last_smile = now
            smile_squint = now - last_smile < 0.5
            both_shut = eyes > BLINK_CLOSED and min(eye_l, eye_r) > BLINK_EACH
            long_blink = False
            if eyes < BLINK_OPEN or smile_squint:
                eyes_open_at, closed_since = now, None
                if eyes < BLINK_OPEN:
                    blink_fired = False
            elif both_shut and closed_since is None and now - eyes_open_at < 0.25:
                closed_since = now
            if closed_since and not blink_fired and both_shut and now - closed_since >= BLINK_HOLD:
                blink_fired, long_blink = True, True
            gestures_ok = ui["open"] and ui["face"] and cap is None and now - last_capture_end > GESTURE_COOLDOWN
            if b:
                face_trace.append((now, eye_l, eye_r, smile, b.get("jawOpen", 0),
                                   (b.get("cheekSquintLeft", 0) + b.get("cheekSquintRight", 0)) / 2,
                                   b.get("browInnerUp", 0)))
                while face_trace and now - face_trace[0][0] > 3:
                    face_trace.popleft()

            # ---- hands-free start: lips moving while you face the screen (measuring mode only logs it)
            hf_trigger = False
            pose = activity.pose
            # "facing the screen": learned from your own deliberate starts (Option / long blink) once there are
            # 8+, as their typical head pose +-10 deg (+-12 yaw); the wide default until then
            fp = amb.get("facing") or {}
            yaw_c, pitch_c = fp.get("yaw", 0.0), fp.get("pitch", (HF_PITCH[0] + HF_PITCH[1]) / 2)
            yaw_w, pitch_w = (12.0, 10.0) if fp else (HF_YAW, (HF_PITCH[1] - HF_PITCH[0]) / 2)
            facing = pose is not None and abs(pose[0] - yaw_c) < yaw_w and abs(pose[1] - pitch_c) < pitch_w
            if (ui["handsfree"] != "off" and ui["open"] and not ui["type"] and frame is not None and cap is None
                    and hold is None and level is not None):
                ready = (facing and not smiling and not smile_squint and eyes < BLINK_OPEN and not speaker.active
                         and now - last_capture_end > HF_COOLDOWN)
                if hf["shadow"] is not None:  # a would-be capture in progress: it ends like a real one would
                    if level > cal["off"]:
                        hf["shadow"]["last"] = now
                    if now - hf["shadow"]["last"] > 1.1:
                        hf_note("would-end", f"mouthed {hf['shadow']['last'] - hf['shadow']['start']:.2f}s")
                        hf["shadow"] = None
                elif ready and level > cal["on"] * amb["hf_gain"]:
                    hf["since"] = hf["since"] or now
                elif not ready or level < cal["off"]:
                    hf["since"] = None
                if hf["since"] and now - hf["since"] >= HF_HOLD and hf["shadow"] is None:
                    hf_trigger = True
            else:
                hf["since"] = None

            # ---- nod = yes, head shake = no, when a question is waiting (AirPods motion if they're in)
            if pose is not None and ui["open"]:
                nod_camera.add(now, pose[1], pose[0])
            awaiting = bool(ui.get("confirm")) or (now - ui["question_at"] < 25 and turn is None and not pending)
            if awaiting and now > ui["nod_quiet_until"] and cap is None:
                source = "AirPods" if airpods is not None and airpods.active() else "camera"
                gesture_seen = (nod_airpods if source == "AirPods" else nod_camera).detect()
                if gesture_seen:
                    answer = "yes" if gesture_seen == "nod" else "no"
                    ui["nod_quiet_until"], ui["question_at"] = now + 2.0, -99.0
                    notice = f"{'Nodded' if gesture_seen == 'nod' else 'Shook your head'}: {answer}."
                    log_face(f"{gesture_seen.upper()} ({source}) -> {answer}", now)
                    if not (ui.get("confirm") and confirm_answer(answer)):
                        pending.append({"clip": None, "hyps": None, "typed": answer, "spoken": "",
                                        "gesture": gesture_seen})
            elif not awaiting:  # (head movements only count while a question is waiting)
                nod_airpods.clear()
                nod_camera.clear()

            # ---- capture: hold Option, or a long blink, or hands-free
            if not ui["open"]:  # hidden: hold Option to dictate into the app in front (out loud or mouthed)
                if hold is None and down:
                    start_hold(now)
                elif hold is not None:
                    if frame is not None:
                        hold["items"].append((now, frame))
                        hold["mouthing"] += bool(level and level > cal["on"])
                    hold["release_at"] = None if down else (hold["release_at"] or now)
                    if (hold["release_at"] and now - hold["release_at"] >= 0.25) or now - hold["start"] > 60:
                        finish_hold(now)
            elif frame is None:
                pass
            elif cap is None:
                preroll.append((now, frame))
                while preroll and now - preroll[0][0] > 0.8:
                    preroll.popleft()
                if down:
                    start_capture("key", now)
                    log_face("listening started by Option key", now)
                elif long_blink and gestures_ok:
                    start_capture("blink", now)
                    log_face("listening started by LONG BLINK", now)
                elif hf_trigger and ui["handsfree"] == "on":
                    started = hf["since"]
                    start_capture("auto", now)
                    cap["speech_start"], cap["last_active"], cap["trigger_end"] = started - 0.15, now, now
                    hf["since"] = None
                    hf_note("auto-start", f"pose {pose[0]:.0f},{pose[1]:.0f} level {level:.3f}")
                    log_face("listening started HANDS-FREE", now)
                elif hf_trigger:  # measuring: note when it would have started listening
                    hf["shadow"], hf["since"] = {"start": hf["since"], "last": now}, None
                    hf_note("would-start", f"pose {pose[0]:.0f},{pose[1]:.0f} level {level:.3f}")
                if cap is not None and cap["mode"] in ("key", "blink"):
                    hf_note("real-start", cap["mode"] + (f" pose {pose[0]:.0f},{pose[1]:.0f}" if pose is not None else ""))
                    if pose is not None:
                        poses = (settings.get("real_poses") or [])[-29:] + [[round(pose[0], 1), round(pose[1], 1)]]
                        save_setting("real_poses", poses)
                        if len(poses) >= 8:
                            a = np.array(poses)
                            amb["facing"] = {"yaw": float(np.median(a[:, 0])), "pitch": float(np.median(a[:, 1]))}
            else:
                cap["items"].append((now, frame))
                if cap["mode"] == "key":
                    cap["release_at"] = None if down else (cap["release_at"] or now)
                    if (cap["release_at"] and now - cap["release_at"] >= 0.25) or now - cap["start"] > 30:
                        finish_capture(now)
                elif cap["speech_start"] is None and smiling:  # smiled while it waited: cancel listening
                    cap, last_capture_end = None, now - GESTURE_COOLDOWN  # (so the smile can delete right away)
                    notice = "Cancelled listening (you smiled)."
                    log_face("smile cancelled listening", now)
                elif cap["speech_start"] is None:  # long blink: wait for your eyes to open and mouthing to begin
                    if eyes < BLINK_OPEN:
                        cap["trigger_end"] = cap["trigger_end"] or now
                    # a smile, or relaxing one, moves the mouth too: that's not mouthing
                    mouthing = cap["trigger_end"] and level and level > cal["on"] and not smile_squint
                    cap["above"] = cap["above"] + 1 if mouthing else 0  # a smile moves the mouth too: not mouthing
                    if cap["above"] >= 3:
                        cap["speech_start"], cap["last_active"] = now - 0.1, now
                        log_face("mouthing detected after long blink -> Listening", now)
                    elif (cap["trigger_end"] and now - cap["trigger_end"] > 5) or now - cap["start"] > 12:
                        cap, last_capture_end = None, now
                        notice = "Saw your long blink, but no mouthing after it."
                else:  # listening until you stop mouthing for ~1 s
                    if level is not None and level > cal["off"]:
                        cap["last_active"] = now
                    if now - cap["last_active"] > 1.1 or now - cap["start"] > 30:
                        finish_capture(now, lo=cap["speech_start"] - 0.4, hi=cap["last_active"] + 0.3)

            # ---- Terminal input
            while not typed.empty():
                line = typed.get()
                cmd, _, rest = line.partition(" ")
                if cmd == "/quit":
                    ui["quit"] = True
                elif cmd == "/macctl":  # (for testing) run macctl with this app's permissions; output goes to the log
                    def run_macctl(line=rest):
                        dbg = DATA / "macctl_debug"
                        dbg.mkdir(exist_ok=True)
                        import shlex
                        r = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "mac.py"),
                                            *shlex.split(line)], cwd=dbg, capture_output=True, text=True, timeout=90)
                        print(f"[macctl] {line}\n{r.stdout}{r.stderr}[macctl end]", flush=True)
                    threading.Thread(target=run_macctl, daemon=True).start()
                elif cmd == "/handsfree" and rest.strip() in ("on", "off", "shadow"):
                    set_handsfree(rest.strip())
                elif cmd == "/handsfree":  # stats: how often it would have started, vs your real starts
                    print("[handsfree] " + handsfree_stats(hf_log), flush=True)
                elif cmd in ("/screen", "/notify"):  # on | off | (no argument: switch)
                    key, setting = ("screen", "screen_context") if cmd == "/screen" else ("notify", "notifications")
                    ui[key] = rest.strip() == "on" if rest.strip() in ("on", "off") else not ui[key]
                    save_setting(setting, ui[key])
                    if key == "notify":
                        notif_watch.enabled = ui["notify"]
                    screen["last"] = None
                    on_text = {"screen": "Screen summary on: Claude sees what's in front with each message.",
                               "notify": "Notifications on: shown here, and read out when voice is on."}
                    notice = on_text[key] if ui[key] else f"{'Screen summary' if key == 'screen' else 'Notifications'} off."
                    print(notice, flush=True)
                elif cmd == "/eval" and rest.split()[:1] == ["context"]:  # (measuring) needs this app's permissions
                    name = "eval_context_with.txt" if "--with-only" in rest else "eval_context.txt"
                    with open(ROOT / "data" / "eval_context.log", "w") as log:
                        subprocess.Popen([str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "eval_context.py"),
                                          str(ROOT / "data" / name), *rest.split()[1:]], cwd=ROOT, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                    print("[eval] context benchmark started: data/eval_context.txt", flush=True)
                elif cmd == "/gaze" and rest.strip() == "calibrate":
                    if not ui["open"]:
                        notice = "Show silent chat first (tap \\): calibration needs the camera."
                    else:
                        gazer.start_calibration()
                        notice = "Gaze calibration: look at each dot (about a minute)."
                        print("[gaze] calibrating on the screen under the mouse", flush=True)
                elif cmd == "/gaze" and rest.strip() == "cancel":
                    gazer.cancel_calibration()
                elif cmd == "/gaze" and rest.strip().startswith("show"):
                    secs = float(rest.split()[1]) if len(rest.split()) > 1 else 20
                    gazer.show(secs)
                    notice = ("A ring follows your gaze for a while." if gazer.ready else
                              "Calibrate first: /gaze calibrate (or the menu bar).")
                elif cmd == "/gaze" and rest.strip().startswith("where"):
                    w = rest.split()
                    n, gap = (int(w[1]) if len(w) > 1 else 5), (float(w[2]) if len(w) > 2 else 4.0)
                    if not (gazer.ready and ui["open"]):
                        notice = "Show silent chat (camera on) and calibrate first."
                    else:
                        ui["gaze_where"] = [time.monotonic() + gap * (i + 1) for i in range(n)]
                        notice = f"Look at a button; every {gap:.0f} s a tick reads what you look at (presses nothing)."
                        print(f"[gaze] where: {n} reads, {gap:.0f} s apart", flush=True)
                elif cmd == "/improve":
                    arg = rest.strip()
                    if arg == "list":
                        print("[improve] " + improve.listing(), flush=True)
                    elif arg in ("on", "off"):
                        d = improve.load()
                        d["enabled"] = arg == "on"
                        improve.save(d)
                        notice = f"Self-improvement {arg}."
                    else:
                        improve_now(dry=arg == "dry")
                        notice = "Looking at how you've used silent chat this week ..."
                elif cmd == "/lens" and rest.strip().startswith("warp"):
                    ui["warp"] = gazer.warp_on = rest.split()[-1] == "on" if len(rest.split()) > 1 else not ui["warp"]
                    save_setting("gaze_warp", ui["warp"])
                    gazer.hide_lens()  # (rebuilt next frame, with or without the bend)
                    ok = gaze_mod.Warper.allowed(ask=ui["warp"])
                    notice = ("Glass bend on." if ok else "Glass bend: allow Silent Chat under Screen & System Audio "
                              "Recording in System Settings, then restart.") if ui["warp"] else "Glass bend off."
                elif cmd == "/mem":  # what's holding memory
                    trim_gpu()
                    rep = []
                    try:
                        import mlx.core as _mx
                        rep.append(f"MLX active {_mx.get_active_memory() / 1e6:.0f} MB, cache {_mx.get_cache_memory() / 1e6:.0f} MB")
                    except Exception as e:
                        rep.append(f"MLX ? {e}")
                    try:
                        import torch
                        rep.append(f"torch MPS allocated {torch.mps.current_allocated_memory() / 1e6:.0f} MB, "
                                   f"driver {torch.mps.driver_allocated_memory() / 1e6:.0f} MB")
                    except Exception as e:
                        rep.append(f"torch ? {e}")
                    tm = transcriber["model"]
                    rep.append(f"whisper turbo {'loaded' if tm is not None and tm.loaded is not None else 'not loaded'}, "
                               f"small {'loaded' if wake.fast is not None and wake.fast.loaded is not None else 'not loaded'}, "
                               f"local model {'loaded' if router.ready else 'not loaded'}")
                    print("[mem] " + "; ".join(rep), flush=True)
                elif cmd == "/island":  # (debugging) where the island is and what it shows
                    print(f"[island] shown {isl.shown} size {isl.w:.0f}x{isl.h:.0f} goal {isl.goal} field hidden "
                          f"{isl.field.isHidden()} frame {isl.field.frame()} state {isl.state} hover {amb.get('hover')}",
                          flush=True)
                elif cmd == "/power":  # measure its own CPU and energy impact for 20 s, as it is right now
                    def measure(state=("camera on" if ui["open"] and cam.cap is not None else "camera off")):
                        pid = os.getpid()
                        out = subprocess.run(["top", "-l", "11", "-s", "2", "-pid", str(pid), "-stats", "pid,cpu,power"],
                                             capture_output=True, text=True).stdout
                        rows = [l.split() for l in out.splitlines() if l.split()[:1] == [str(pid)]][1:]
                        if rows:
                            cpu = sum(float(r[1]) for r in rows) / len(rows)
                            pw = sum(float(r[2]) for r in rows) / len(rows)
                            amb["np_note"] = (f"Power ({state}): CPU {cpu:.0f}% of one core, energy impact {pw:.0f} "
                                              "(Activity Monitor's scale)")
                            print("[power] " + amb["np_note"], flush=True)
                    threading.Thread(target=measure, daemon=True).start()
                    notice = "Measuring power use for 20 s ..."
                elif cmd == "/nudges" and rest.strip() in ("on", "off"):
                    save_setting("nudges", rest.strip() == "on")
                    notice = f"Reply suggestions {rest.strip()}."
                elif cmd == "/privacy" and len(rest.split()) == 2 and rest.split()[1] in ("on", "off"):
                    what, flag = rest.split()[0], rest.split()[1] == "on"
                    if what in ("clips", "remember", "clicks"):
                        save_setting({"clips": "save_clips", "remember": "remember", "clicks": "gaze_clicks"}[what], flag)
                        if what == "clips":
                            SAVE_CLIPS[0] = flag
                            labels.enabled = flag
                        elif what == "remember" and not flag:
                            recent_file.unlink(missing_ok=True)  # (and forget what was kept)
                        elif what == "clicks":
                            gazer.learn_clicks = flag
                        notice = {"clips": "Saving mouth clips for training", "remember": "Remembering the conversation "
                                  "across restarts", "clicks": "Learning gaze from your clicks"}[what] + \
                            (" on." if flag else " off.")
                elif cmd == "/stats":
                    import stats
                    text = stats.summary()
                    print("[stats] " + text.replace("\n", " | "), flush=True)
                    if rest.strip() == "window":
                        from AppKit import NSAlert, NSApp
                        al = NSAlert.alloc().init()
                        al.setMessageText_("How Silent Chat is doing")
                        al.setInformativeText_(text.split("\n", 1)[1] if "\n" in text else text)
                        NSApp.activateIgnoringOtherApps_(True)
                        al.runModal()
                elif cmd == "/practice":  # ~2 min: mouth 20 short phrases it shows you; each clip is labelled
                    if not SAVE_CLIPS[0]:
                        notice = "Practice needs Save mouth clips on (Settings)."
                    elif not ui["open"]:
                        notice = f"Turn the camera on first ({key_label})."
                    else:
                        import random as _r
                        fixed = list(local_commands.FIXED)
                        names = sorted(memory.data["vocabulary"], key=lambda w: -memory.data["vocabulary"][w])[:8]
                        everyday = ["open my messages", "what's on my calendar today", "reply sounds good",
                                    "play some music", "search for the weather tomorrow", "close this tab",
                                    "remind me to call mom", "read that again", "turn it down", "go back"]
                        pool = _r.sample(fixed, min(7, len(fixed))) + _r.sample(everyday, 8) + \
                            [f"message {n.lower()}" for n in names[:5]]
                        _r.shuffle(pool)
                        ui["practice"] = {"phrases": pool[:20], "i": 0, "right": 0}
                        conversation.clear()
                        conversation.append(("Notification", "Practice: hold Option (or blink long) and mouth each "
                                             "phrase. Esc stops."))
                        conversation.append(("Claude", f"1/{len(ui['practice']['phrases'])}  Mouth: "
                                                       f"“{ui['practice']['phrases'][0]}”"))
                        pres.poke(time.monotonic(), 30)
                elif cmd in ("/settings", "/setup"):
                    open_window(cmd[1:])
                elif cmd == "/ambient":
                    arg = rest.strip()
                    ui["ambient"] = (arg == "on") if arg in ("on", "off") else not ui["ambient"]
                    save_setting("ambient", ui["ambient"])
                    ui["pinned"] = False
                    if not ui["ambient"]:
                        panel.set_parts()
                        panel.tick(True)
                    notice = ("Auto-hide on: silent chat shows itself only while you use it; \\ pins it open."
                              if ui["ambient"] else "Auto-hide off: the panel stays up while silent chat is on.")
                    pres.poke(time.monotonic(), 5)
                elif cmd == "/lens" or (cmd == "/gaze" and rest.strip().startswith("lens")):
                    arg = rest.split()[-1] if rest.split() and rest.split()[-1] in ("on", "off") else None
                    ui["lens"] = (arg == "on") if arg else not ui["lens"]
                    gazer.lens_on = ui["lens"]
                    save_setting("gaze_lens", ui["lens"])
                    notice = ("Glass circle on: it follows your gaze while silent chat is showing." +
                              ("" if gazer.ready else " Calibrate gaze first.")) if ui["lens"] else "Glass circle off."
                elif cmd == "/gaze" and rest.strip() in ("on", "off"):
                    ui["gaze"] = rest.strip() == "on"
                    save_setting("gaze", ui["gaze"])
                    notice = f"Gaze {'on: Claude knows what you look at when you say this' if ui['gaze'] else 'off'}."
                elif cmd == "/gaze":
                    print("[gaze] " + (gaze_mod.describe(gazer.info) + f" (calibrated {gazer.info.get('date')})"
                                       if gazer.ready else "not calibrated: /gaze calibrate") +
                          f"; {'on' if ui['gaze'] else 'off'}; head: "
                          f"{'AirPods + camera' if gazer.head.gain != (0.0, 0.0) else 'camera'}; glass circle "
                          + (("showing" if gazer.lens.shown else "hidden") + (
                              f", bend {gazer.lens.warper.frames} frames, {gazer.lens.warper.busy * 1000:.0f} ms each"
                              if gazer.lens.warper else ", no bend") if gazer.lens else "off"), flush=True)
                elif cmd == "/usage":
                    print("[usage] " + usage_stats(usage_log), flush=True)
                elif cmd == "/model" and rest.strip() in ("auto", "haiku", "sonnet", "opus"):
                    ui["model"] = rest.strip()  # switched below, once Claude isn't in the middle of a reply
                elif cmd == "/restart":  # the app's launcher starts it again (exit code 75)
                    ui["quit"] = ui["restart"] = True
                    save_setting("restart_hidden", not ui["open"])
                elif cmd in ("/voice", "/face", "/type", "/mic", "/wake"):
                    on_button(cmd[1:])
                elif cmd in ("/smile", "/blink"):
                    on_button("face")
                elif cmd in ("/hide", "/show"):
                    ui["toggle"] = cmd == "/show"
                elif cmd == "/status":
                    print(f"[status] {'shown' if ui['open'] else 'hidden'}, {'TYPE' if ui['type'] else 'CHAT'} mode, "
                          f"voice {'on' if speaker.enabled else 'off'}, face gestures {'on' if ui['face'] else 'off'}, "
                          f"mic {'on' if ui['mic'] else 'off'}, Hey Claude {'on' if wake.enabled else 'off'}, "
                          f"toggle key {key_label}; {wake.level_report()}; head motion: "
                          f"{airpods.status if airpods is not None else 'off'}", flush=True)
                elif cmd == "/fix" and rest.strip():
                    apply_fix(rest.strip(), tell_claude=True)
                elif cmd == "/learn" and rest.strip():
                    added = memory.add_words(rest.replace(",", " ").split())
                    notice = f"Added to your vocabulary: {', '.join(added) or '(already known)'}"
                elif line.startswith("/"):
                    notice = ("Commands: /fix <text>, /learn <words>, /voice, /face, /type, /mic, /wake, /screen, /notify, "
                              "/handsfree, /usage, /hide, /show, /status, /restart, /quit")
                else:
                    pending.append({"clip": None, "hyps": None, "typed": line})

            if ui.get("note_change"):
                notes[:] = [n for n in notes if not n.startswith("(The user changed the app's settings")]
                notes.append(f"(The user changed the app's settings themselves; now {app_state()[1:-1]}.)")
                ui["note_change"] = None

            while not notif_q.empty():  # ---- new notifications
                on_notification(notif_q.get(), now)

            # ---- self-improvement: once a day, after 10 idle minutes; its changes show as one-line notes
            if now > imp["next_check"]:
                imp["next_check"] = now + 300
                router.unload_if_idle(300)  # (memory: the local model reloads in ~2 s when next needed)
                trim_gpu()
                if (turn is None and not pending and cap is None and hold is None and now - imp["last_msg"] > 600
                        and improve.due()):
                    improve_now(quiet=True)
            while not imp["q"].empty():
                changes, stats, dry, quiet = imp["q"].get()
                real = [c for c in changes if not c.startswith("(")]
                print(f"[improve] {'(dry run) ' if dry else ''}{changes or ['no changes']} {stats}", flush=True)
                for c in (real if quiet else changes or ["no changes"])[:4]:
                    conversation.append(("Notification", ("Would have " if dry else "Learned: ") + c))

            # ---- "Hey Claude": talk out loud, even while hidden (it pops up without the camera)
            for ev in wake.poll():
                if ev[0] == "wake":
                    voice["listen_until"], voice["peek_until"], voice["wake_at"] = now + 8, None, now
                    if not ui["open"] and not voice["peek"]:
                        conversation.clear()
                        voice["peek"] = True
                        if not ui['ambient']:  # (auto-hide: the island shows everything; the old panel stays away)
                            panel.unhide()
                    notice = 'Listening ("Hey Claude") ...'
                    subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])
                    print("[voice] Hey Claude", flush=True)
                elif ev[0] == "said" and turn is not None and turn.get("acting") and \
                        re.match(r"\W*(stop|cancel|never ?mind|wait)\b", ev[1], re.I):
                    stop_agent(f"said {ev[1]!r}")
                elif ev[0] == "said":
                    voice["listen_until"], voice["turn"], notice = 0.0, True, ""
                    # where their eyes were while they spoke (camera on), so "press this" works out loud too
                    screen_soon(gazer.point(voice.get("wake_at", now - 4), now) if ui["open"] else None)
                    pending.append({"clip": None, "hyps": None, "typed": None, "spoken": ev[1], "voice": True})
                    print(f"[voice] said: {ev[1]}", flush=True)
                elif ev[0] == "idle":
                    voice["listen_until"], notice = 0.0, ""

            # ---- lip-reading results
            while lip_jobs and lip_jobs[0][0].done():
                job, mode = lip_jobs.pop(0)
                if not lip_jobs:
                    trim_gpu()
                if ui.get("practice") and not mode.startswith("auto:"):  # practice: label it, show the next one
                    try:
                        result = job.result()
                    except Exception:
                        result = None
                    pr = ui["practice"]
                    phrase = pr["phrases"][pr["i"]]
                    if result and result.get("clip"):
                        labels.add(result["clip"], phrase, "confirmed", "practice", result["hyps"])
                        got = result["hyps"][0][0] if result["hyps"] else "?"
                        right = got.lower().strip(" .") == phrase.lower().strip(" .")
                        pr["right"] += right
                        conversation.append(("Notification", f"{'✓' if right else '•'} it read: {got.lower()}"))
                    pr["i"] += 1
                    if pr["i"] >= len(pr["phrases"]):
                        conversation.append(("Notification", f"Practice done: {pr['right']} of {len(pr['phrases'])} "
                                             "read exactly right. Those clips now train your lip-reader."))
                        ui["practice"] = None
                    else:
                        conversation.append(("Claude", f"{pr['i'] + 1}/{len(pr['phrases'])}  Mouth: "
                                                       f"“{pr['phrases'][pr['i']]}”"))
                    pres.poke(now, 30)
                    continue
                try:
                    result = job.result()
                    notice = ""
                    if mode.startswith("auto:"):
                        mode = mode[5:]
                        top = [h[0].lower().split()[:1] for h in (result.get("hyps") or [])[:3]]
                        if settings.get("handsfree_cue") and not any(w and w[0] in CUE_WORDS for w in top):
                            hf_note("auto-no-cue", repr(result["hyps"][0][0]))
                            continue
                        if now - ui["question_at"] < 25 and turn is None and not ui.get("confirm"):
                            # Claude just asked you something: this is most likely your answer, so it goes back
                            # to Claude as is (it has the context to tell an answer from chatter)
                            hf_note("auto-answer", repr(result["hyps"][0][0]))
                            result["handsfree"] = True
                            notes.append("(Hands-free, right after your question: probably their answer to it.)")
                            pending.append(result)
                        elif try_closed_set(result):  # hands-free only acts on a clear instant command or shortcut
                            hf_note("auto-used", repr(result["hyps"][0][0]))
                            amb["hf_gain"] = max(1.0, amb["hf_gain"] * 0.98)
                        elif (cmd_ := local_commands.parse_part(result, typer.frontmost()[0])) and \
                                cmd_.kind == "macctl":  # its own reading is an everyday command ("please pause")
                            hf_note("auto-used", repr(result["hyps"][0][0]))
                            run_local(cmd_, result)
                        elif presence.plausible_request(result.get("hyps"), running_app_names()) and router.usable:
                            hf_note("auto-to-local", repr(result["hyps"][0][0]))
                            route_async(result)  # (decided in the background: see "routed" below)
                        elif presence.plausible_request(result.get("hyps"), running_app_names()):
                            # looks like it could be a real request (a free local check): Claude decides, and says
                            # IGNORE if it isn't one (then it's struck through on the island, nothing else happens)
                            hf_note("auto-to-claude", repr(result["hyps"][0][0]))
                            result["handsfree"] = True
                            pending.append(result)
                        else:  # a false start (talking to someone, chewing ...): struck through, then gone
                            hf_note("auto-rejected", repr(result["hyps"][0][0]))
                            amb["struck"] = (result["hyps"][0][0].capitalize(), time.monotonic())
                            amb["hf_gain"] = min(1.6, amb["hf_gain"] * 1.05)  # (a bit harder to set off next time)
                        save_setting("hf_gain", round(amb["hf_gain"], 3))
                        continue
                    quick = local_commands.parse_part(result) if mode == "type" else None
                    if quick and quick.kind == "action":  # e.g. "turn off typing mode": no model needed
                        run_local(quick, result)
                    elif mode == "chat" and try_closed_set(result):  # an instant command, read closed-set
                        pass
                    elif mode != "hold":
                        (dictation["pending"] if mode == "type" else pending).append(result)
                    elif result.get("spoken") and local_commands.is_request(result["spoken"]):
                        # Option held, but it's a request for Claude ("message Marianne on Discord that ..."), not
                        # words to type: Claude does it instead
                        print(f"[hold] that's a request, not dictation: {result['spoken']!r}", flush=True)
                        pending.append({"clip": None, "hyps": None, "typed": None, "spoken": result["spoken"],
                                        "voice": True})
                    elif result.get("spoken"):
                        type_now(result["spoken"])
                    else:  # mouthed: Claude picks the right words from the lip-reader's guesses, then it's typed
                        if dictation["session"] is None:
                            dictation["session"] = ClaudeSession(DICTATE_SYSTEM, model=MODELS["haiku"], effort=args.effort)
                        dictation["pending"].append(result)
                except Exception as e:  # e.g. mouth not visible
                    amb["read_failed"] = time.monotonic()  # (the text box shows for a while: typing may be easier)
                    if mode.startswith("auto:"):
                        hf_note("auto-error", type(e).__name__)
                        continue
                    notice = f"Couldn't read that ({type(e).__name__}). Keep your mouth in view and try again."
                    print(notice, e, flush=True)
                    if mode == "hold":
                        subprocess.Popen(["afplay", "-v", "0.4", "/System/Library/Sounds/Funk.aiff"])

            if amb.get("np_note"):  # (a background measurement finished)
                conversation.append(("Notification", amb.pop("np_note")))
                pres.poke(now, 8)
            # ---- the local model's decisions on hands-free reads
            while not routed.empty():
                result, d = routed.get()
                amb["deciding"] = None
                hf_note(f"local-{d['kind']}", f"{d['ms']}ms {result['hyps'][0][0]!r} -> {d['text']!r} {d.get('cmd') or ''}")
                if d["kind"] == "ignore":
                    amb["struck"] = (result["hyps"][0][0].capitalize(), time.monotonic())
                    amb["hf_gain"] = min(1.6, amb["hf_gain"] * 1.05)
                elif d["kind"] == "act" and d.get("cmd"):
                    say = {"pause": "Paused.", "next": "Skipped.", "previous": "Back one."}.get(
                        d["cmd"][1] if d["cmd"][0] == "spotify" else "", "Done.")
                    run_local(local_commands.Command("macctl", d["cmd"], say),
                              {"typed": d["text"], "clip": None, "hyps": result.get("hyps")})  # (an unchecked guess:
                    # not saved as lip-reading training data)
                else:  # a real request that needs Claude: with the words the local model fixed
                    result["handsfree"] = True
                    notes.append(f"(A small local model thinks the user meant: \"{d['text']}\")")
                    pending.append(result)

            # ---- Claude: send when you've finished; stream the reply
            if ui.get("model") and turn is None and not session.busy:
                session.close()
                session = make_session(ui["model"])  # (a fresh conversation)
                screen["last"] = None
                save_setting("model", ui["model"])
                notice, ui["model"] = f"Claude model: {ui['model']}.", None
            if (pending and len(pending) == 1 and turn is None and not lip_jobs and cap is None
                    and not args.no_local):
                part = pending[0]
                text = part.get("typed") or part.get("spoken") or (part.get("hyps") or [("", 0)])[0][0]
                hit = shortcuts.match(text) if text and not text.startswith("/") else None
                if ui.get("confirm") and confirm_answer(text):
                    pending.pop()
                elif hit:  # a saved shortcut: replayed on the Mac, no model needed
                    replay(hit[0], hit[1], pending.pop())
                elif cmd := local_commands.parse_part(part, typer.frontmost()[0]):
                    run_local(cmd, pending.pop())  # an everyday command: done on the Mac, no model needed
            if ui.get("confirm") and time.monotonic() - ui["confirm"]["at"] > 45:
                ui["confirm"] = None  # an unanswered "go ahead?" expires
            while not local_done.empty():
                heard, reply, spoken, *extra = local_done.get()
                extra = extra[0] if extra else {}
                if extra.get("confirm"):
                    ui["confirm"] = {"steps": extra["confirm"], "at": time.monotonic()}
                if extra.get("handover"):  # a shortcut failed: Claude does it properly
                    notes.append(f"(A saved shortcut for this failed: {extra['why']}. Do it yourself, and "
                                 "fix or re-save the shortcut if it's outdated.)")
                    pending.append({"clip": None, "hyps": None, "typed": extra["handover"], "spoken": ""})
                for i in range(len(conversation) - 1, -1, -1):
                    if conversation[i] == ("Claude", "..."):
                        conversation[i] = ("Claude", reply)
                        break
                notes.append(f"(Done on the Mac without you, as an instant command: \"{heard}\" -> {reply})")
                if spoken and headphones() and not presence.watching(front_line()):
                    was = speaker.enabled
                    speaker.enabled = True
                    speaker.begin_reply()
                    speaker.say(spoken)
                    speaker.enabled = was
                if re.search(r"\b(play|resume|next|skip|previous|last song)\b", heard, re.I) and \
                        re.search(r"spotify|song|track|music|play", heard + " " + reply, re.I):
                    now_playing_soon()
                print(f"Local:  {heard} -> {reply}", flush=True)
                if not extra.get("handover"):
                    improve.log("instant", said=heard, reply=reply[:120])
            if pending and turn is None and not session.busy and not lip_jobs and cap is None:
                mem = memory.prompt_block()
                screen_block, screen_full = screen_now()
                notes += notifications_note(time.monotonic())
                if amb.get("dictated"):  # what you dictated (Option held) since your last message, for context
                    notes.append("(Since their last message the user dictated, by holding Option, into the app in "
                                 "front: " + "; ".join(f'into {a}: "{t}"' for a, t in amb["dictated"][-3:]) +
                                 ". It was typed there as text, not sent to you.)")
                    amb["dictated"] = []
                if resume["pending"]:
                    resume["pending"] = False
                    rn = resume_note()
                    if rn:
                        notes.append(rn)
                hands_free_turn = any(p.get("handsfree") for p in pending)
                if hands_free_turn:
                    notes.append("(This came from HANDS-FREE lip-reading, which also catches the user talking to "
                                 "someone else, chewing or thinking aloud. If it isn't clearly a request meant for "
                                 "you, reply with your HEARD line and then exactly IGNORE, and do nothing else.)")
                front = re.match(r"Front: (.+?)(?: - ['\"]|$| \()", screen_full.split("\n", 1)[1] if "\n" in screen_full else "")
                looking = next((l for l in screen_full.splitlines() if l.startswith("Looking at")), "")
                try:
                    notes += improve.tips_for(front.group(1) if front else "")
                except Exception:
                    pass
                message = build_message(pending, notes, mem if mem != memory_sent else "", app_state(), screen_block)
                full = (build_message(pending, notes, mem if mem != memory_sent else "", app_state(), screen_full)
                        if screen_full != screen_block else None)
                memory_sent = mem
                # instant commands / shortcuts read from your lips: if this next message isn't an objection,
                # they were right, and those clips become training data
                if amb.get("local_clips"):
                    said = " ".join((p.get("typed") or p.get("spoken") or "") for p in pending).lower()
                    objected = re.match(r"\W*(no|nope|wrong|undo|stop|cancel|that's not|that wasn't|not that)\b", said)
                    if not objected:
                        labels.update([c for c, _ in amb["local_clips"]], only_status="heard", status="ok")
                    amb["local_clips"] = []
                turn = {"parts": pending, "buf": "", "in_reply": False, "heard": None, "fix": None, "reply": "",
                        "speech": "", "muted": False, "shown": False, "sent": time.monotonic(),
                        "screen": len(screen_block), "tools": 0,
                        "app": front.group(1) if front else "", "looking": looking[:200],
                        "handsfree": hands_free_turn}
                # speak this reply? Not without headphones (shown instead), not over a video; yes if you spoke
                speaker.enabled = speak_now(any(p.get("voice") for p in pending)) and not hands_free_turn
                voice["restore_voice"] = True
                pending, notes = [], []
                ui["question_at"] = -99.0
                speaker.begin_reply()
                stop_file.unlink(missing_ok=True)  # a new request: Claude may act again
                session.send(message, full=full)
                imp["last_msg"] = time.monotonic()
            for ev in session.poll():
                if ev[0] == "escalated" and turn is not None:  # Haiku handed it to Opus
                    turn["model"] = "Opus"
                    notice = "Thinking harder (Opus) ..."
                    print("[model] handed to Opus", flush=True)
                if ev[0] == "tool" and turn is not None:
                    turn["tools"] = turn.get("tools", 0) + 1
                if ev[0] == "text" and turn is not None:
                    turn.setdefault("first", time.monotonic() - turn["sent"])
                    feed(turn, ev[1])
                elif ev[0] == "tool" and turn is not None and ev[1].startswith("mcp__"):  # Todoist / Spotify
                    where, what = connectors.label(ev[1])
                    notice = f"{where}: {what}"
                    print(f"[{where.lower()}] {what}: {json.dumps(ev[2])[:160]}", flush=True)
                elif ev[0] == "tool" and turn is not None:  # Claude is acting on your Mac
                    cmd = ev[2].get("command") or ev[2].get("file_path") or ev[2].get("query") or ""
                    if ev[1] == "Bash":
                        turn["acting"] = True
                    shown = (steps_of(cmd) if "macctl" in cmd else
                             "looking at the screenshot" if ev[1] == "Read" else cmd)
                    notice = (f"On your Mac (\\ stops): {shown[:70]}" if ev[1] != "WebSearch"
                              else f"Searching: {cmd[:50]}")
                    print(f"[mac] {ev[1]}: {cmd[:120]}", flush=True)
                    if "macctl" in cmd:
                        turn.setdefault("cmds", []).append(steps_of(cmd)[:200])
                elif ev[0] == "done" and turn is not None and turn.get("handsfree") and \
                        re.sub(r"(?is)^.*?heard:[^\n]*\n", "", str(ev[1])).strip().upper().startswith("IGNORE"):
                    log_usage(turn, ev)  # Claude says it wasn't meant for it: strike it through, do nothing
                    hf_note("claude-ignored", repr(turn.get("heard")))
                    amb["struck"] = ((turn.get("heard") or "?").capitalize(), time.monotonic())
                    while conversation and conversation[-1][0] in ("Claude", "You"):
                        conversation.pop()
                        if conversation and conversation[-1][0] != "Claude":
                            break
                    amb["hf_gain"] = min(1.6, amb["hf_gain"] * 1.05)
                    turn = None
                elif ev[0] == "done" and turn is not None:
                    log_usage(turn, ev)
                    try:  # (stats.py: speed)
                        with open(DATA / "metrics.jsonl", "a") as fh:
                            fh.write(json.dumps({"at": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "first_ms":
                                                 round(turn.get("first", 0) * 1000), "voice": any(
                                                     p.get("voice") for p in turn["parts"]), "mouthed": any(
                                                     p.get("hyps") for p in turn["parts"])}) + "\n")
                    except OSError:
                        pass
                    improve.log("message", said=turn.get("heard") or " / ".join(
                        (p.get("typed") or p.get("spoken") or "")[:200] for p in turn["parts"]),
                        app=turn.get("app"), looking=turn.get("looking"), did=(turn.get("cmds") or [])[:12],
                        reply=" ".join(str(ev[1]).split())[:200], error=bool(ev[2]))
                    finish_turn(turn, ev[1], ev[2])
                    if any(c.startswith("spotify") and re.search(r"play|next|previous", c)
                           for c in turn.get("cmds") or []):
                        now_playing_soon()
                    turn = None
                elif ev[0] == "exit":
                    notice = "The Claude session ended; starting a new one (earlier context is lost)."
                    if turn is not None:
                        pending, turn = turn["parts"] + pending, None
                    memory_sent = None
                    resume["pending"] = True  # (a new Claude session: give it the recent exchanges)
            speaker.update()
            if speaker.active:
                voice["quiet_since"] = now
            if turn is None and not pending and not speaker.active and not lip_jobs:
                if voice["turn"] and wake.enabled:  # reply finished: answer back without "Hey Claude" for 5 s
                    wake.follow_up(5)
                    voice["listen_until"] = now + 5
                    notice = "Listening for a follow-up ..."
                voice["turn"] = False
                if voice["restore_voice"]:
                    speaker.enabled, voice["restore_voice"] = ui["voice_pref"], False
            if voice["peek"]:  # the "Hey Claude" pop-up hides itself once the conversation goes quiet
                if (turn is None and not pending and not speaker.active and now > voice["listen_until"]
                        and not wake.following):
                    voice["peek_until"] = voice["peek_until"] or now + 6
                    if now > voice["peek_until"]:
                        set_open(False)
                else:
                    voice["peek_until"] = None
            busy = ("listening" if hold is not None else
                    "thinking" if not ui["open"] and (any(m == "hold" for _, m in lip_jobs) or dictation["turn"]
                                                      or dictation["pending"]) else None)
            menu.set_activity(busy, notice if not ui["open"] else "")

            # ---- TYPE mode: Claude turns the lip-reading into text, which is typed into the app in front
            ds = dictation["session"]
            if ds is not None:
                if (dictation["pending"] and dictation["turn"] is None and not ds.busy
                        and not any(m == "type" for _, m in lip_jobs) and cap is None):
                    bundle, pid, app = typer.frontmost()
                    if pid != dictation["pid"]:
                        dictation.update(pid=pid, app=app, field="", last="", words=0, last_clips=[])
                    mem = memory.prompt_block()
                    parts = dictation["pending"]
                    body = (describe(parts[0]) if len(parts) == 1 else
                            "\n".join(f"Part {i}:\n{describe(p)}" for i, p in enumerate(parts, 1)))
                    msg = "\n\n".join([app_state()] + ([mem] if mem != dictation["memory_sent"] else []) + [
                        f"Typing into: {app or bundle}",
                        f"Typed there so far: {dictation['field'][-300:]!r}" if dictation["field"] else
                        "Typed there so far: (nothing yet)",
                        "Lip-reader hypotheses for what they just mouthed, best first:\n" + body])
                    dictation["memory_sent"] = mem
                    dictation["turn"] = {"parts": parts, "sent": now, "buf": "", "typed": "", "started": False,
                                         "blocked": "", "prefix": ""}
                    dictation["pending"] = []
                    ds.send(msg)
                for ev in ds.poll():
                    t = dictation["turn"]
                    if t is None:
                        continue
                    if ev[0] == "text":
                        t["buf"] += ev[1]
                        dictate_flush(t, final=False)  # type whole words as they stream in
                    elif ev[0] in ("done", "exit"):
                        dictation["turn"] = None
                        if ev[0] == "done" and ev[2]:
                            notice = f"Claude error: {ev[1][:60]}"
                            continue
                        if not t["started"] and not t["buf"].strip() and ev[0] == "done":
                            t["buf"] = ev[1]  # nothing streamed: use the final text
                        dictate_flush(t, final=True)
                        if t.get("actions"):
                            done = [d for d in (apply_action(a) for a in t["actions"]) if d]
                            notice = ("Done: " + ", ".join(done) + ".") if done else "Didn't understand that command."
                            conversation.append(("Typed", f"(command: {', '.join(done) or '?'})"))
                            continue
                        text = t["typed"].strip()
                        if t["blocked"]:
                            notice = t["blocked"]
                            conversation.append(("Typed", f"(not typed) {t['buf'].strip()[:120]}"))
                            if not ui["open"]:  # (hold-to-dictate while hidden: you can't see the notice)
                                subprocess.Popen(["afplay", "-v", "0.4", "/System/Library/Sounds/Funk.aiff"])
                            continue
                        if not text:
                            notice = "Couldn't make that out; try again."
                            continue
                        if dictation["last_clips"] and not dictation["deleted"]:  # you kept the previous chunk
                            labels.update(dictation["last_clips"], only_status="heard", status="ok")
                        clips = [p["clip"] for p in t["parts"]]
                        single = len(t["parts"]) == 1
                        for p in t["parts"]:
                            if p.get("spoken"):
                                labels.add(p["clip"], p["spoken"], "audio", "mic_dictation", p["hyps"])
                                memory.learn_from_speech(p["spoken"])
                                continue
                            labels.add(p["clip"], text if single else "", "heard" if single else "unlabelled",
                                       "dictation" if single else "dictation_multipart", p["hyps"])
                        dictation.update(field=dictation["field"] + t["typed"], last=t["typed"], deleted=False,
                                         words=dictation["words"] + len(text.split()),
                                         last_clips=clips if single else [])
                        conversation.append(("Typed", text))
                        print(f"lip-reader: {t['parts'][0]['hyps'][0][0]}\nTyped:      {text}  "
                              f"[first words {t.get('first', 0):.1f} s, done {now - t['sent']:.1f} s]\n", flush=True)

            # ---- TYPE-mode face gestures: smile deletes dictated words (1 per second), brow raise / wink = Enter
            gesture = False
            if ui["type"] and gestures_ok:
                if smiling:
                    gesture = True
                    smile_since = smile_since or now
                    held = now - smile_since
                    due = 0 if held < SMILE_FIRST else int((held - SMILE_FIRST) // SMILE_EVERY) + 1
                    while smile_deleted < due:
                        smile_deleted += 1
                        if not delete_dictated_word():
                            smile_deleted = due
                elif smile < SMILE_OFF:
                    smile_since, smile_deleted = None, 0
                if args.send == "brows":
                    up = (b.get("browInnerUp", 0) > BROWS_UP
                          and (b.get("browOuterUpLeft", 0) + b.get("browOuterUpRight", 0)) / 2 > 0.8 * BROWS_UP)
                    released = b.get("browInnerUp", 0) < BROWS_DOWN
                else:
                    up = (eye_l > WINK_CLOSED and eye_r < WINK_OPEN) or (eye_r > WINK_CLOSED and eye_l < WINK_OPEN)
                    released = not up
                if up:
                    gesture = True
                    send_since = send_since or now
                    if not send_fired and now - send_since >= (BROWS_HOLD if args.send == "brows" else WINK_HOLD):
                        send_fired = True
                        send_dictated()
                elif released:
                    send_since, send_fired = None, False
            else:
                smile_since, smile_deleted, send_since = None, 0, None

            # ---- overlay
            if cap is not None:
                if cap["mode"] == "blink" and cap["speech_start"] is None:
                    state, status = "waiting", "Blink seen - start mouthing"
                else:
                    state, status = "capturing", f"Listening ... {now - cap['start']:.1f} s"
            elif dictation["turn"] is not None or dictation["pending"] or (lip_jobs and ui["type"]):
                state, status = "thinking", "Typing ..."
            elif lip_jobs or pending or (turn is not None and not turn["reply"]):
                state, status = "thinking", "Thinking ..."
            elif speaker.active or turn is not None:
                state, status = "speaking", "Claude is speaking"
            else:
                verb = "type" if ui["type"] else "talk"
                state, status = "idle", (f"Blink long or hold {hotkey.label}" if ui["face"] else
                                         f"Hold {hotkey.label}") + f" to {verb}"
            if activity.landmarks is None and ui["open"]:
                state, status = "noface", "Can't see your mouth"
            if state in ("idle", "noface") and (now < voice["listen_until"] or wake.following):
                state, status = "capturing", "Listening (Hey Claude) ..."
            elif not ui["open"] and state == "idle":
                state, status = "idle", 'Say "Hey Claude" again, or press ' + key_label
            # ---- ambient: show the panel only while it's in use; speak / duck the music as fits
            if not ui["ambient"]:
                while not amb["np"].empty():
                    np_ = amb["np"].get()
                    conversation.append(("Notification", f"♫ {np_['title']} — {np_['artist']}"))
            auto_cap = cap is not None and cap["mode"] == "auto"
            auto_jobs = bool(lip_jobs) and all(m.startswith("auto:") for _, m in lip_jobs)
            busy = bool((cap is not None and not auto_cap) or hold is not None or (lip_jobs and not auto_jobs)
                        or pending or turn is not None or speaker.active or ui.get("confirm")
                        or now < voice["listen_until"] or wake.following or now - ui["question_at"] < 25)
            want_duck = (now < voice["listen_until"] or wake.following or speaker.speaking
                         or (voice["turn"] and (turn is not None or bool(pending))))
            if want_duck:
                amb["duck_off_since"] = None
                if not amb.get("ducking"):
                    amb["ducking"] = True
                    ducker.duck()
            elif amb.get("ducking"):
                amb["duck_off_since"] = amb["duck_off_since"] or now
                if now - amb["duck_off_since"] > 1.5:
                    amb["ducking"] = False
                    ducker.restore()
            # the small activity pill: what's happening right now, or nothing
            amp, ind = 0.0, None
            mouthing = level is not None and level > cal["off"]
            if mouthing:
                amp = level / (cal["on"] * 2.5)
            hearing = now < voice["listen_until"] or wake.following
            v_amp = 0.0
            if hearing and wake.levels:
                try:
                    v_amp = wake.levels[-1] / (wake.threshold() * 2)
                except Exception:
                    pass
            if speaker.speaking:
                ind = "speaking"
            elif cap is not None or hold is not None:
                ind = "mouth" if mouthing else (None if auto_cap else "listen")
            elif hearing:
                ind, amp = ("voice", v_amp) if v_amp > 0.5 else ("listen", 0.0)
            elif (lip_jobs and not auto_jobs) or dictation["pending"] or dictation["turn"] is not None:
                ind = "loading"  # (not for a hands-free start: if it turns out false, nothing should have shown)
            elif pending or turn is not None:
                ind = "working"
            if ui["ambient"]:  # the Dynamic Island (island.py); with the camera off too, replies still pop up
                if panel.alpha > 0 or panel.panel.isVisible():
                    panel.tick(False)
                while not amb["np"].empty():
                    amb["playing"], amb["playing_until"] = amb["np"].get(), now + 6
                    pres.poke(now, 6)
                pres.content(now, (tuple(conversation[-3:]), notice),
                             " ".join(t for _, t in conversation[-2:]) + " " + notice)
                gp = gazer.point(now - 0.4, now) if gazer.ready and ui["gaze"] else None
                ix, iy, iw, ih = isl.rect()
                # hovering the island (or the bare notch) opens the chat with a text box; it closes when you leave
                mp = NSEvent.mouseLocation()
                if amb.pop("unfocus", False):
                    isl.unfocus()
                over = isl.hit(mp.x, mp.y)
                if amb.get("hover_lock"):
                    if not over:
                        amb["hover_lock"] = False  # (the mouse left: hovering works again)
                    over = False
                if over:
                    amb["hover_out"] = None
                    amb["hover_in"] = amb.get("hover_in") or now
                else:
                    amb["hover_in"] = None
                    amb["hover_out"] = amb.get("hover_out") or now
                while not esc_q.empty():  # Esc: dismiss now (and forget the hover / history state)
                    esc_q.get()
                    pres.show_until, amb["hover"], amb["history_open"] = 0.0, False, False
                    amb["gaze_peek"], amb["peek_lock"] = False, True
                    if ui.get("practice"):
                        ui["practice"] = None
                        conversation.append(("Notification", "Practice stopped."))
                    amb["dismissed_at"], isl.max_text = now, 220.0
                    isl.unfocus()
                # looking up at the notch for a moment brings the recent conversation back (to re-read a reply that
                # tucked away too soon); looking away for a bit lets it go again. Only a steady stare right at the
                # notch counts (a glance at tabs or the menu bar doesn't).
                at_top = (gp is not None and abs(gp[0] - (ix + iw / 2)) <= min(iw / 2 + 40, 160)
                          and gp[1] <= iy + ih + 15)
                if at_top and not amb.get("peek_lock"):
                    amb["peek_out"] = None
                    amb["peek_in"] = amb.get("peek_in") or now
                else:
                    amb["peek_in"] = None
                    amb["peek_out"] = amb.get("peek_out") or now
                    if not at_top:
                        amb["peek_lock"] = False  # (after Esc: look away once, then looking up works again)
                if (amb.get("peek_in") and now - amb["peek_in"] > 1.0 and now - amb.get("dismissed_at", 0) > 1.0
                        and (amb.get("history") or conversation)):
                    if not amb.get("gaze_peek"):
                        print("[island] you looked up at it: showing the conversation", flush=True)
                    amb["gaze_peek"] = True
                elif amb.get("gaze_peek") and amb.get("peek_out") and now - amb["peek_out"] > 1.5:
                    amb["gaze_peek"] = False
                if amb.get("hover_in") and now - amb["hover_in"] > 0.15 and now - amb.get("dismissed_at", 0) > 1.0:
                    amb["hover"] = True  # (the text box takes the keyboard only when you click it, never by itself)
                elif amb.get("hover") and amb.get("hover_out") and now - amb["hover_out"] > 0.6 and not isl.has_text():
                    amb["hover"] = False
                    isl.unfocus()
                on_panel = gp is not None and ix - 30 <= gp[0] <= ix + iw + 30 and gp[1] <= iy + ih + 40
                struck_live = bool(amb.get("struck")) and now - amb["struck"][1] < 1.6
                deciding = bool(amb.get("deciding")) and now - amb["deciding"][1] < 5
                vis = pres.visible(now, busy or ind is not None or struck_live or deciding,
                                   ui["pinned"] or amb.get("hover", False) or amb.get("history_open", False)
                                   or amb.get("gaze_peek", False),
                                   on_panel)
                if amb.get("struck") and not struck_live and not busy:
                    amb["struck"] = None
                    vis = vis and bool(ind) or amb.get("hover", False)
                amb["noface_since"] = (amb["noface_since"] or now) if activity.landmarks is None else None
                want_field = ui["pinned"] or amb.get("hover", False) or (vis and ((amb["noface_since"] is not None and now - amb["noface_since"] > 1.5)
                                                       or now - amb["read_failed"] < 20
                                                       or (not ui["mic"] and not wake.enabled)))
                open_up = amb.get("hover") or amb.get("history_open") or amb.get("gaze_peek")
                lines = [(w, t) for w, t in (amb.get("history", []) + conversation if open_up else conversation)
                         [-(12 if amb.get("history_open") else 6 if open_up else 3):] if t and t != "..."]
                if turn is not None and turn.get("handsfree"):  # (until Claude decides it's for it)
                    lines = [x for x in lines if x[0] != "Claude"]
                if amb.get("deciding") and now - amb["deciding"][1] < 5:  # (the local model is deciding)
                    lines, ind = [("You", amb["deciding"][0])], "loading"
                struck = amb.get("struck")
                if struck and now - struck[1] < 1.6:  # a misread: struck through, then the island retracts
                    lines, ind = [("You", struck[0], struck[1])], None
                    pres.show_until = 0.0
                if notice and not ind:
                    lines.append(("Notification", notice))
                playing = None
                if amb.get("playing") and (now < amb.get("playing_until", 0) or (on_panel and vis)):
                    playing = amb["playing"]
                    amb["playing_until"] = max(amb.get("playing_until", 0), now + 1.5) if on_panel else amb["playing_until"]
                # compact (just the live activity) while you talk; it opens up when there's something to read
                show_lines = lines if (ind not in ("mouth", "voice", "listen") or amb.get("hover")) else []
                while not vote_q.empty():  # thumbs up / down on the last reply
                    v = vote_q.get()
                    if amb.get("vote_for"):
                        amb["voted"] = v
                        rec = {"at": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "vote": v, **amb["vote_for"]}
                        try:
                            with open(DATA / "feedback.jsonl", "a") as fh:
                                fh.write(json.dumps(rec) + "\n")
                        except OSError:
                            pass
                        improve.log("vote", said=rec.get("said"), reply=rec.get("reply"), ok="yes" if v == "up" else "no")
                        pres.poke(now, 1.5)
                last_reply = next((x[1] for x in reversed(show_lines) if x[0] == "Claude" and x[1] and x[1] != "..."),
                                  None)
                if last_reply and turn is None and not amb.get("struck"):
                    said = next((x[1] for x in reversed(show_lines) if x[0] in ("You", "Typed")), "")
                    if (amb.get("vote_for") or {}).get("reply") != last_reply:
                        amb["vote_for"], amb["voted"] = {"said": said, "reply": last_reply}, ""
                    votable = amb.get("voted", "")
                else:
                    votable = None
                isl.update(ind=ind, amp=amp, lines=show_lines, playing=playing, field=want_field and vis, visible=vis,
                           votable=votable if not playing else None)
                if ui.get("focus_field") and not isl.field.isHidden():  # (once it has opened)
                    ui["focus_field"] = False
                    isl.focus_field()
                if not vis:
                    amb["hidden_at"] = amb["hidden_at"] or now
                    if now - amb["hidden_at"] > 1 and (conversation or notice):
                        amb["history"] = (amb.get("history", []) + [c for c in conversation if c[1] != "..."])[-12:]
                        conversation.clear()  # (the next time it shows, it starts fresh; hovering shows the history)
                        notice = ""
                else:
                    amb["hidden_at"] = None
                continue
            if not (ui["open"] or voice["peek"]):
                continue
            panel.show(inputs=input_states(now, level, smile_squint), state=state, status=status,
                       voice=speaker.enabled, face=ui["face"], gesture=gesture, conversation=conversation,
                       notice=notice, typing=ui["type"], mic=ui["mic"], minimal=ui["ambient"], indicator=ind, amp=amp)
    except KeyboardInterrupt:
        pass
    finally:
        speaker.stop()
        mic.stop()
        session.close()
        if dictation["session"] is not None:
            dictation["session"].close()
        panel.close()
        cam.close()
        activity.close()
        reader.close()
        lip_pool.shutdown(wait=False, cancel_futures=True)
        io_pool.shutdown(wait=True)  # finish saving clips
    return 75 if ui["restart"] else 0  # 75: the app's launcher starts it again


# ----------------------------------------------------------------------------- push-to-talk + confirm mode

def wait(rec, future, header, status):
    """Keep the camera preview alive while background work runs. Returns its result, or None if Q was pressed."""
    while not future.done():
        rec._show(rec._read(), header, "", status)
        if (cv2.waitKey(30) & 0xFF) in (ord("q"), ord("Q")):
            return None
    return future.result()


def interpret(reader, clip, history, memory_block):
    """Clip -> (n-best hypotheses, up to 3 interpretations from Claude)."""
    crops, _ = reader.mouth_crops(vsr.load_video(clip))
    hyps, _ = reader.decode(crops, nbest=10)
    prompt = ((memory_block + "\n\n") if memory_block else "") + (
        f"Conversation so far:\n{transcript(history)}\n\n"
        f"Lip-reader hypotheses for the user's next message, best first:\n{format_nbest(hyps)}")
    options = []
    for line in run_claude(prompt, INTERPRET_SYSTEM)["text"].splitlines():
        text = line.strip().lstrip("0123456789.)-• ").strip().strip('"').strip()
        if text and text.lower() not in (o.lower() for o in options):
            options.append(text)
    return hyps, (options or [hyps[0][0].capitalize()])[:3]


def chat(history):
    prompt = f"Conversation so far:\n{transcript(history, last=20)}\n\nWrite Claude's next reply."
    return run_claude(prompt, CHAT_SYSTEM)["text"]


def choose(rec, header, options):
    """Show the interpretations. Returns the chosen text, "retry", "type", or None (quit)."""
    numbered = [f"{i}. {o}" for i, o in enumerate(options, 1)]
    keys = "/".join(str(i) for i in range(1, len(options) + 1))
    status = f"{keys} = send that   SPACE = mouth it again   T = type it   Q = quit"
    while True:
        rec._show(rec._read(), header + ["", "Did you say:"], numbered, status)
        key = cv2.waitKey(1) & 0xFF
        if ord("1") <= key < ord("1") + len(options):
            return options[key - ord("1")]
        if key == 32:
            return "retry"
        if key in (ord("t"), ord("T")):
            return "type"
        if key in (ord("q"), ord("Q")):
            return None


def run_confirm(args):
    print("Loading the lip-reader ...", flush=True)
    reader = vsr.LipReader(args.device)
    checker = vsr.ClipChecker()
    rec = vsr.Recorder(args.camera)
    pool = cf.ThreadPoolExecutor(1)
    speaker = Speaker(args.speak, device=args.audio)
    labels, memory = Labels(), PersonalMemory()
    history = []
    header = ["Silent chat with Claude. Hold SPACE and silently mouth a message.",
              "Keep your face in the box. Q quits."]
    print("Ready. Use the camera window: hold SPACE while you mouth a message.", flush=True)
    try:
        while True:
            clip = DATA / "clips" / f"{datetime.now():%Y%m%d_%H%M%S}.mp4"
            if rec.take(clip, header, "", max_seconds=15, check=checker) is None:
                break
            speaker.stop()
            job = pool.submit(interpret, reader, clip, history, memory.prompt_block())
            result = wait(rec, job, header, "Reading your lips ...")
            if result is None:
                break
            hyps, options = result
            print(f"lip-reader: {hyps[0][0]}\noptions:    {options}", flush=True)
            choice = choose(rec, header, options)
            if choice is None:
                break
            if choice == "retry":
                clip.unlink(missing_ok=True)
                continue
            if choice == "type":
                rec._show(rec._read(), header, "", "Type your message in the Terminal window ...")
                cv2.waitKey(1)
                text = input("\nType what you mouthed (Enter to send, empty to cancel): ").strip()
                if not text:
                    clip.unlink(missing_ok=True)
                    continue
                source = "typed"
                memory.learn_correction(hyps[0][0], options[0], text)
            else:
                text, source = choice, f"option{options.index(choice) + 1}"
            labels.add(clip, text, "confirmed", source, hyps, options)
            history.append(("User", text))
            print(f"You:    {text}", flush=True)
            reply = wait(rec, pool.submit(chat, history), [f"You: {text}"], "Claude is thinking ...")
            if reply is None:
                break
            history.append(("Claude", reply))
            print(f"Claude: {reply}\n", flush=True)
            header = [f"You: {text}", f"Claude: {reply}"]
            speaker.say(reply)
    finally:
        speaker.stop()
        rec.close()
        checker.close()
        reader.close()
        pool.shutdown(wait=False, cancel_futures=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speak", action="store_true", help="start with spoken replies on (click the window to toggle)")
    ap.add_argument("--hotkey", choices=["option", "right_option", "control"], default="option")
    ap.add_argument("--audio", default="auto", help='"auto" (AirPods when connected), "default", or part of an output name, e.g. "Speakers"')
    ap.add_argument("--confirm", action="store_true", help="older mode: SPACE in the camera window + pick 1/2/3")
    ap.add_argument("--no-face", action="store_true", help="start with face gestures off (FACE button toggles)")
    ap.add_argument("--send", choices=["brows", "wink"], default="brows",
                    help="TYPE-mode gesture for Enter: eyebrow raise (default, looks natural) or wink")
    ap.add_argument("--type", action="store_true", help="start in TYPE mode: what you mouth is typed into the app in front")
    ap.add_argument("--toggle-key", default="\\", choices=["\\", "`", "§", "none"],
                    help="global key to show/hide silent chat (double-tap types the key itself)")
    ap.add_argument("--hidden", action="store_true", help="start hidden (camera off) until the toggle key is pressed")
    ap.add_argument("--mic", action="store_true", help="start with MIC on (speak out loud, e.g. to correct names)")
    ap.add_argument("--mic-device", default="MacBook", help="microphone to use (part of its name)")
    ap.add_argument("--commands", default=None, help="named pipe to read commands from (used by the launcher)")
    ap.add_argument("--no-mac", action="store_true", help="don't let Claude open/arrange/look at apps on your Mac")
    ap.add_argument("--no-wake", action="store_true", help='don\'t listen for "Hey Claude" (the menu toggles it)')
    ap.add_argument("--no-hold-dictation", action="store_true",
                    help="while hidden, holding Option does nothing (default: dictates into the app in front)")
    ap.add_argument("--no-connectors", action="store_true",
                    help="don't give Claude your claude.ai Todoist/Spotify connectors (see connectors.py)")
    ap.add_argument("--handsfree", choices=["off", "shadow", "on"], default=None,
                    help="start listening when you mouth words while facing the screen (shadow = only measure)")
    ap.add_argument("--model", choices=["auto", "haiku", "sonnet", "opus"], default=None,
                    help="Claude model: auto = Haiku first, Opus for hard requests (default; remembered from /model)")
    ap.add_argument("--no-local", action="store_true",
                    help="send everything to Claude (default: everyday commands run instantly without a model)")
    ap.add_argument("--effort", choices=["low", "medium", "high"], default="low",
                    help="Claude effort: low is ~2x faster and was as accurate on the test sentences")
    ap.add_argument("--beam", type=int, default=10, help="lip-reader beam size (10 is ~3x faster than 40, same accuracy)")
    ap.add_argument("--no-train", action="store_true", help="don't fine-tune the lip-reader when the chat closes")
    ap.add_argument("--device", choices=vsr.DEVICES, default=vsr.DEFAULT_DEVICE)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    (DATA / "clips").mkdir(parents=True, exist_ok=True)
    pidfile = DATA / "silent_chat.pid"  # lets the launcher tell whether silent chat is running
    pidfile.write_text(str(os.getpid()))
    before = len(Labels().rows)
    try:
        code = (run_confirm if args.confirm else run_live)(args)
    finally:
        if pidfile.exists() and pidfile.read_text() == str(os.getpid()):
            pidfile.unlink()
    labels = Labels()
    print(f"\nSaved {len(labels.rows) - before} clip(s) this session; {len(labels.trainable())} usable for training.")
    if not args.no_train:  # retrain in a separate background process, so quitting/restarting never cuts it off
        import train_personal
        print(train_personal.train_in_background(DATA / "training.log"), flush=True)
    sys.exit(code or 0)


if __name__ == "__main__":
    main()
