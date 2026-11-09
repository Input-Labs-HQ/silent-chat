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
    "'mic on' / 'mic off' (out-loud speech recognition, e.g. for correcting names), 'wake on' / 'wake off' (always "
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
    "3. Only if ui can't see it (a canvas, image, game or odd app): 'see [App]' saves a small screenshot; Read it, "
    "then 'click x y' with pixel coordinates in that image ('see --zoom x y w h' to look closer at part of it).\n"
    "Batch steps you can predict into ONE call: ./macctl do \"focus Discord\" \"key cmd+k\" \"type alex\" "
    "\"wait 0.5\" \"key return\" (it stops at the first failure). Every action prints what has focus afterwards: "
    "trust that instead of checking again; 'wait-for <text>' waits for something to appear. If a name the "
    "lip-reader gave you is unclear, look at what's really there (ui --find, or the app's list) and pick the closest "
    "match. When you work out something non-obvious about an app, save it for next time: "
    "./macctl learn <App> \"<one-line tip>\".\n"
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
    "WebSearch for its open.spotify.com link), then play that; play/pause/next already report what's playing. "
    "Quote app names that contain spaces. If the request is unclear or might be a misreading, ask before acting. "
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
    "Reply with ONLY the text to type: what they most likely said, with natural punctuation and capitalisation that "
    "continue the text typed so far (no capital mid-sentence, don't repeat earlier text). Spoken commands become "
    "symbols: 'new line' -> a line break, 'new paragraph' -> two line breaks, 'comma', 'period', 'question mark' "
    "and so on -> the punctuation. Numbers as digits where natural. No quotes, labels or commentary. If you can't "
    "make out anything plausible, reply with exactly: ?\n"
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

def lipread(reader, frames, stamps, clip, io_pool, audio=None, transcriber=None, vocabulary=()):
    """Captured frames -> lip-reader n-best (+ speech-to-text if the mic was on). The clip is saved in the
    background, off the critical path."""
    spoken = transcriber(audio, vocabulary) if audio is not None and transcriber is not None else ""
    video = vsr.resample_to_fps(frames, stamps)
    io_pool.submit(vsr.write_video, video, clip, pix_fmt="bgr24")
    crops, _ = reader.mouth_crops(np.ascontiguousarray(video[..., ::-1]))
    hyps, _ = reader.decode(crops, nbest=10)
    return {"clip": clip, "hyps": hyps, "typed": None, "spoken": spoken}


def describe(p):
    """How a message part is shown to Claude."""
    if p.get("typed"):
        return f"(typed) {p['typed']}"
    if p.get("voice"):
        return f"(said out loud after 'Hey Claude'; they may not be looking at the screen) {p['spoken']}"
    if p.get("spoken"):
        return f"(said out loud) {p['spoken']}"
    return format_nbest(p["hyps"])


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


def build_message(parts, notes, memory_block, state=""):
    body = (describe(parts[0]) if len(parts) == 1 else
            "\n".join(f"Part {i}:\n{describe(p)}" for i, p in enumerate(parts, 1)))
    blocks = ([state] if state else []) + ([memory_block] if memory_block else []) + notes
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

    def mic_on(flag):
        if flag:
            name = mic.start()
            if transcriber["model"] is None:
                transcriber["model"] = speech.Transcriber()
                threading.Thread(target=transcriber["model"].warm, daemon=True).start()
            return name
        if not wake.enabled:  # "Hey Claude" keeps the mic open
            mic.stop()
    import typer
    dictation = {"session": None, "turn": None, "pending": [], "pid": None, "app": "", "field": "", "last": "",
                 "words": 0, "last_clips": [], "deleted": False, "memory_sent": None}

    def on_button(name):
        if name in ("voice", "face", "type", "mic", "wake") and not ui.get("from_chat"):
            ui["note_change"] = name  # tell Claude on your next message (it didn't make this change)
        if name == "voice":
            speaker.enabled = not speaker.enabled
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
                    dictation["session"] = ClaudeSession(DICTATE_SYSTEM, effort=args.effort)
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

    panel = Overlay(on_button, DATA / "overlay.json")

    def setup_macctl(workdir):
        """Give the chat session ./macctl (its only command) and a screens/ folder it may read."""
        wrapper = Path(workdir) / "macctl"
        wrapper.write_text(f'#!/bin/zsh\nexec "{ROOT}/.venv/bin/python" "{ROOT}/mac.py" "$@"\n')
        wrapper.chmod(0o755)
        (Path(workdir) / "screens").mkdir(exist_ok=True)

    if args.no_mac:
        session = ClaudeSession(LIVE_SYSTEM, effort=args.effort)
    else:  # Claude may run only ./macctl and read only its own screenshots: the CLI refuses anything else
        found = {}
        if not args.no_connectors:  # Todoist / Spotify connected on your claude.ai account (connectors.py)
            disc.join(timeout=25)
            found = found_box or connectors.cached()
        mcp, extra_allowed, disallowed, extra_prompt = connectors.session_options(found)
        print(f"Connectors: {', '.join(found) or 'none'}.", flush=True)
        session = ClaudeSession(LIVE_SYSTEM + "\n" + MAC_CONTROL + ("\n" + extra_prompt if extra_prompt else ""),
                                effort=args.effort, tools="Bash,Read,WebSearch",
                                allowed=["Bash(./macctl:*)", "Read(./screens/**)", "WebSearch", *extra_allowed],
                                setup=setup_macctl, mcp=mcp, disallowed=disallowed)
    typed = queue.Queue()
    if args.commands:
        threading.Thread(target=fifo_reader, args=(typed, args.commands), daemon=True).start()
    else:
        threading.Thread(target=stdin_reader, args=(typed,), daemon=True).start()
    lip_pool, io_pool = cf.ThreadPoolExecutor(1), cf.ThreadPoolExecutor(1)

    conversation, pending, lip_jobs, notes = [], [], [], []
    hold = None  # Option held while hidden: dictation into the app in front
    voice = {"quiet_since": 0.0, "listen_until": 0.0, "turn": False, "peek": False, "peek_until": None,
             "restore_voice": False}  # "Hey Claude" conversations
    turn, last_turn, memory_sent = None, None, None
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
    menu = StatusMenu(typed.put, lambda: {"open": ui["open"], "type": ui["type"], "voice": speaker.enabled,
                                          "face": ui["face"], "mic": ui["mic"], "wake": wake.enabled}, args.toggle_key)
    if args.mic:
        on_button("mic")
    if settings.get("wake", True) and not args.no_wake:
        set_wake(True)
    if args.hidden:
        set_open(False)

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
        """Type what you said out loud straight into the app in front (no Claude round trip needed)."""
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
        print(f"Typed (said out loud) into {app}: {text}", flush=True)

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
        mode, cap, last_capture_end = cap["mode"], None, now
        if len(items) < 12:
            notice = f"Too short: hold {hotkey.label} for the whole message."
            return
        clip = DATA / "clips" / f"{datetime.now():%Y%m%d_%H%M%S_%f}.mp4"
        audio = mic.segment(items[0][0] - 0.15, items[-1][0] + 0.15) if mic.on and ui["mic"] else None
        vocab = sorted(memory.data["vocabulary"], key=lambda w: -memory.data["vocabulary"][w])[:40]
        job = lip_pool.submit(lipread, reader, [f for _, f in items], [t for t, _ in items], clip, io_pool,
                              audio, transcriber["model"], vocab)
        lip_jobs.append((job, "type" if ui["type"] else "chat"))

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
                f"Hey Claude {'on' if wake.enabled else 'off'}.)")

    def apply_action(text, from_chat=False):
        """Carry out one 'ACTION: ...' line from Claude. Returns what changed, or ''."""
        m = re.search(r"\b(type|typing|voice|face|mic|wake|hide)\b\W*(on|off)?", text.lower())
        if not m:
            return ""
        name, value = {"typing": "type"}.get(m.group(1), m.group(1)), m.group(2)
        if name == "hide":
            ui["toggle"] = False
            return "hidden"
        current = {"type": ui["type"], "voice": speaker.enabled, "face": ui["face"], "mic": ui["mic"],
                   "wake": wake.enabled}[name]
        want = (not current) if value is None else value == "on"
        if want != current:
            ui["from_chat"] = from_chat
            on_button(name)
            ui["from_chat"] = False
        label = {"type": "TYPE mode", "voice": "voice", "face": "face gestures", "mic": "MIC",
                 "wake": "Hey Claude"}[name]
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
            if not final and " " not in head:
                return  # wait for a whole first word (also rules out a bare "?")
            if final and head.strip().strip(quotes) in ("", "?"):
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
        heard, fix = t["heard"] or "", t["fix"]
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
                elif action == "toggle":
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
            if ui["open"] and cam.cap is None and now - retry["camera"] > 3:
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
                # ignore the talk key just after we pressed keys ourselves (e.g. Option+Delete when smiling to delete)
                down = hotkey.held() and now - typer.LAST_SYNTHETIC[0] > 0.5
            else:  # hidden (camera off): only Option dictation and "Hey Claude"; replies in flight still finish
                time.sleep(0.03)
                frame, level = None, None
                down = (not ui["open"] and not args.no_hold_dictation and hotkey.held()
                        and now - typer.LAST_SYNTHETIC[0] > 0.5)
                activity.blend, activity.landmarks = {}, None
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

            # ---- capture: hold Option, or a long blink
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
                while preroll and now - preroll[0][0] > 0.4:
                    preroll.popleft()
                if down:
                    start_capture("key", now)
                    log_face("listening started by Option key", now)
                elif long_blink and gestures_ok:
                    start_capture("blink", now)
                    log_face("listening started by LONG BLINK", now)
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
                elif cmd == "/restart":  # the app's launcher starts it again (exit code 75)
                    ui["quit"] = ui["restart"] = True
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
                          f"toggle key {key_label}; {wake.level_report()}", flush=True)
                elif cmd == "/fix" and rest.strip():
                    apply_fix(rest.strip(), tell_claude=True)
                elif cmd == "/learn" and rest.strip():
                    added = memory.add_words(rest.replace(",", " ").split())
                    notice = f"Added to your vocabulary: {', '.join(added) or '(already known)'}"
                elif line.startswith("/"):
                    notice = "Commands: /fix <text>, /learn <words>, /voice, /face, /type, /mic, /wake, /hide, /show, /status, /restart, /quit"
                else:
                    pending.append({"clip": None, "hyps": None, "typed": line})

            if ui.get("note_change"):
                notes[:] = [n for n in notes if not n.startswith("(The user changed the app's settings")]
                notes.append(f"(The user changed the app's settings themselves; now {app_state()[1:-1]}.)")
                ui["note_change"] = None

            # ---- "Hey Claude": talk out loud, even while hidden (it pops up without the camera)
            for ev in wake.poll():
                if ev[0] == "wake":
                    voice["listen_until"], voice["peek_until"] = now + 8, None
                    if not ui["open"] and not voice["peek"]:
                        conversation.clear()
                        voice["peek"] = True
                        panel.unhide()
                    notice = 'Listening ("Hey Claude") ...'
                    subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])
                    print("[voice] Hey Claude", flush=True)
                elif ev[0] == "said" and turn is not None and turn.get("acting") and \
                        re.match(r"\W*(stop|cancel|never ?mind|wait)\b", ev[1], re.I):
                    stop_agent(f"said {ev[1]!r}")
                elif ev[0] == "said":
                    voice["listen_until"], voice["turn"], notice = 0.0, True, ""
                    pending.append({"clip": None, "hyps": None, "typed": None, "spoken": ev[1], "voice": True})
                    print(f"[voice] said: {ev[1]}", flush=True)
                elif ev[0] == "idle":
                    voice["listen_until"], notice = 0.0, ""

            # ---- lip-reading results
            while lip_jobs and lip_jobs[0][0].done():
                job, mode = lip_jobs.pop(0)
                try:
                    result = job.result()
                    notice = ""
                    if mode != "hold":
                        (dictation["pending"] if mode == "type" else pending).append(result)
                    elif result.get("spoken"):
                        type_now(result["spoken"])
                    else:  # mouthed: Claude picks the right words from the lip-reader's guesses, then it's typed
                        if dictation["session"] is None:
                            dictation["session"] = ClaudeSession(DICTATE_SYSTEM, effort=args.effort)
                        dictation["pending"].append(result)
                except Exception as e:  # e.g. mouth not visible
                    notice = f"Couldn't read that ({type(e).__name__}). Keep your mouth in view and try again."
                    print(notice, e, flush=True)
                    if mode == "hold":
                        subprocess.Popen(["afplay", "-v", "0.4", "/System/Library/Sounds/Funk.aiff"])

            # ---- Claude: send when you've finished; stream the reply
            if pending and turn is None and not session.busy and not lip_jobs and cap is None:
                mem = memory.prompt_block()
                message = build_message(pending, notes, mem if mem != memory_sent else "", app_state())
                memory_sent = mem
                turn = {"parts": pending, "buf": "", "in_reply": False, "heard": None, "fix": None, "reply": "",
                        "speech": "", "muted": False, "shown": False, "sent": time.monotonic()}
                if any(p.get("voice") for p in pending) and not speaker.enabled:
                    speaker.enabled, voice["restore_voice"] = True, True  # you asked out loud: answer out loud
                pending, notes = [], []
                speaker.begin_reply()
                stop_file.unlink(missing_ok=True)  # a new request: Claude may act again
                session.send(message)
            for ev in session.poll():
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
                elif ev[0] == "done" and turn is not None:
                    finish_turn(turn, ev[1], ev[2])
                    turn = None
                elif ev[0] == "exit":
                    notice = "The Claude session ended; starting a new one (earlier context is lost)."
                    if turn is not None:
                        pending, turn = turn["parts"] + pending, None
                    memory_sent = None
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
                    speaker.enabled, voice["restore_voice"] = False, False
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
                            conversation.append(("Typed", f"(not typed) {t['buf'].strip()}"))
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
            if not (ui["open"] or voice["peek"]):
                continue
            panel.show(inputs=input_states(now, level, smile_squint), state=state, status=status,
                       voice=speaker.enabled, face=ui["face"], gesture=gesture, conversation=conversation,
                       notice=notice, typing=ui["type"], mic=ui["mic"])
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
