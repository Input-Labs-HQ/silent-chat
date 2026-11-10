"""Everyday commands handled instantly on the Mac, with no model at all (free, ~0.2 s instead of 2-4 s).

Volume, play/pause/next, "open Chrome", typing mode, voice/mic/face on/off, hide, scroll, new/close tab, back.
Anything that doesn't clearly match goes to Claude as usual. For lip-reading, the top guess must match (or the top
two must agree), since a misread command should fall through to Claude rather than do the wrong thing.

  parse(text, front_bundle) -> Command(kind, args, say) or None
    kind "action": args is an app-mode change for silent chat ("type on", "voice off", "hide" ...)
    kind "macctl": args is a macctl command line (list)
"""
import difflib
import re
from dataclasses import dataclass
from pathlib import Path

BROWSERS = {"com.google.Chrome", "com.apple.Safari", "company.thebrowser.Browser", "com.brave.Browser",
            "com.microsoft.edgemac", "org.mozilla.firefox"}
ALIASES = {"chrome": "Google Chrome", "settings": "System Settings", "system preferences": "System Settings",
           "vs code": "Visual Studio Code", "vscode": "Visual Studio Code", "messages": "Messages",
           "imessage": "Messages", "mail": "Mail", "terminal": "Terminal", "music": "Music", "photos": "Photos",
           "facetime": "FaceTime", "app store": "App Store", "word": "Microsoft Word", "excel": "Microsoft Excel"}
NUMBERS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                       "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
        "ninety": 90}
_apps = None


@dataclass
class Command:
    kind: str
    args: list
    say: str


def apps():
    """Names of installed apps (cached)."""
    global _apps
    if _apps is None:
        names = set()
        for d in ("/Applications", "/Applications/Utilities", "/System/Applications",
                  "/System/Applications/Utilities", str(Path.home() / "Applications")):
            for p in Path(d).glob("*.app"):
                names.add(p.stem)
        _apps = sorted(names)
    return _apps


def find_app(name):
    name = name.strip().lower()
    if name in ALIASES:
        return ALIASES[name]
    by_lower = {a.lower(): a for a in apps()}
    if name in by_lower:
        return by_lower[name]
    close = difflib.get_close_matches(name, list(by_lower), n=1, cutoff=0.82)
    return by_lower[close[0]] if close else None


def last_number(text):
    """The last number in a sentence: '... to like 20' / '... twenty five' -> int, else None."""
    digits = re.findall(r"\b(\d{1,3})\b", text)
    if digits:
        return int(digits[-1])
    words, found = text.replace("-", " ").split(), None
    for i, w in enumerate(words):
        if w in TENS:
            found = TENS[w] + (NUMBERS[words[i + 1]] if i + 1 < len(words) and NUMBERS.get(words[i + 1], 10) < 10 else 0)
        elif w in NUMBERS and not (i and words[i - 1] in TENS):
            found = NUMBERS[w]
        elif w == "hundred":
            found = 100
    return found


def number(text):
    """'20', 'twenty', 'twenty five', 'a hundred' -> int (0-100), else None."""
    text = text.strip().replace("-", " ")
    if text.isdigit():
        return int(text)
    if text in ("hundred", "a hundred", "one hundred", "max", "full"):
        return 100
    words = text.split()
    if len(words) == 1 and words[0] in NUMBERS:
        return NUMBERS[words[0]]
    if words and words[0] in TENS:
        return TENS[words[0]] + (NUMBERS.get(words[1], 0) if len(words) == 2 and NUMBERS.get(words[1], 10) < 10 else 0)
    return None


def clean(text):
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^\w\s'%-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^(hey claude |claude |okay |ok |um |uh |actually |so |and )+", "", t)
    t = re.sub(r"^(please |can you |could you |would you |go ahead and )+", "", t)
    t = re.sub(r"( please| for me| now| thanks| thank you)+$", "", t)
    return t.strip()


def parse(text, front_bundle=""):
    t = clean(text)
    if not t or len(t.split()) > 8:
        return None
    # silent chat's own modes
    m = re.fullmatch(r"(?:turn on |switch to |start |go to )?typ(?:e|ing) mode(?: on)?", t)
    if m:
        return Command("action", "type on", "Typing mode on.")
    if re.fullmatch(r"(?:turn off |stop |exit |leave )typ(?:e|ing) mode|typ(?:e|ing) mode off|(?:switch to )?chat mode", t):
        return Command("action", "type off", "Chat mode.")
    feature = r"(voice|mic|microphone|face|face gestures|wake word|hey claude|hands free|hands-free|handsfree)"
    m = re.fullmatch(r"(?:turn )?" + feature + r"(?: mode)? (on|off)", t) or \
        re.fullmatch(r"turn (on|off) (?:the )?" + feature + r"(?: mode)?", t)
    if m:
        a, b = m.groups()
        what, state = (a, b) if b in ("on", "off") else (b, a)
        what = {"microphone": "mic", "face gestures": "face", "wake word": "wake", "hey claude": "wake",
                "hands free": "handsfree", "hands-free": "handsfree"}.get(what, what)
        label = {"voice": "Voice", "mic": "Mic", "face": "Face gestures", "wake": "Hey Claude",
                 "handsfree": "Hands-free"}[what]
        return Command("action", f"{what} {state}", f"{label} {state}.")
    if re.fullmatch(r"hide|hide yourself|go away|hide the overlay", t):
        return Command("action", "hide", "")
    # Spotify
    if re.fullmatch(r"(?:play|resume|unpause)(?: the)?(?: music| spotify| song)?", t):
        return Command("macctl", ["spotify", "play"], "")
    if re.fullmatch(r"pause(?: the)?(?: music| spotify| song)?|stop the music", t):
        return Command("macctl", ["spotify", "pause"], "Paused.")
    if re.fullmatch(r"(?:next|skip)(?: the)?(?: song| track)?|skip (?:this|it)(?: song)?", t):
        return Command("macctl", ["spotify", "next"], "")
    if re.fullmatch(r"(?:previous|last|go back a)(?: song| track)?|play the last song", t):
        return Command("macctl", ["spotify", "previous"], "")
    if "volume" in t or t.startswith("turn it ") or t in ("louder", "quieter", "softer"):
        n = last_number(t)
        if n is not None and 0 <= n <= 100:
            return Command("macctl", ["spotify", "volume", str(n)], f"Volume {n}.")
        if re.search(r"\b(up|louder)\b", t):
            return Command("macctl", ["spotify", "volume", "up"], "Louder.")
        if re.search(r"\b(down|quieter|softer|lower)\b", t):
            return Command("macctl", ["spotify", "volume", "down"], "Quieter.")
        return None
    # apps
    m = re.fullmatch(r"(?:open|launch|start|switch to|go to) (?:up )?(?:the )?(.+?)(?: app)?", t)
    if m:
        app = find_app(m.group(1))
        if app:
            return Command("macctl", ["open", app], f"Opened {app}.")
        return None  # "open my DMs with ..." etc.: Claude's job
    if re.fullmatch(r"scroll (up|down)(?: a bit| more| a little)?", t):
        return Command("macctl", ["scroll", t.split()[1]], "")
    # a web search ("search for flights to Tokyo", "google best ramen near me"), not a search *in* something
    m = re.fullmatch(r"(?:search(?: the web| google)?(?: for)?|google|look up) (.+)", t)
    if m and not re.search(r"\b(my|in|on|inside|from)\b", m.group(1)):
        from urllib.parse import quote_plus
        return Command("macctl", ["url", "https://www.google.com/search?q=" + quote_plus(m.group(1))],
                       f"Searching for {m.group(1)}.")
    # browser tabs (only when a browser is in front)
    if front_bundle in BROWSERS:
        if re.fullmatch(r"(?:close|close this|close the) tab", t):
            return Command("macctl", ["key", "cmd+w"], "Closed the tab.")
        if re.fullmatch(r"(?:open )?(?:a )?new tab", t):
            return Command("macctl", ["key", "cmd+t"], "New tab.")
        if re.fullmatch(r"(?:go )?back(?: a page)?", t):
            return Command("macctl", ["key", "cmd+["], "")
        if re.fullmatch(r"reload|refresh(?: the page)?", t):
            return Command("macctl", ["key", "cmd+r"], "")
    return None


REQUEST_START = re.compile(  # commands to the computer itself (not things you'd say to a person in a chat)
    r"^\W*(?:please\s+|can you\s+|could you\s+)?"
    r"(?:open|launch|quit|play|pause|skip|search(?: for)?|google|look up|remind me|schedule|"
    r"turn (?:on|off|up|down)|switch to)\b", re.I)
WAKE = re.compile(r"^\W*(?:hey,?\s+)?claude\b", re.I)  # "Hey Claude, ..." / "Claude, ...": always for Claude
REQUEST_ANYWHERE = re.compile(  # "... message Marianne on Discord that ..." (a capitalised name, then where or what)
    r"\b(?i:message|text|dm|email|reply to|send a message to)\s+[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)?"
    r"\s+(?i:on|in|via|that|saying|to say|about)\b")


def is_request(text):
    """Something said while dictating that's really a request for Claude, not words to type
    ("message Marianne on Discord that this works", "open Spotify", "hey Claude, what's ...")."""
    t = (text or "").strip()
    return bool(WAKE.search(t) or REQUEST_START.search(t) or REQUEST_ANYWHERE.search(t))


def parse_part(part, front_bundle=""):
    """A message part (typed / said out loud / lip-reader n-best) -> Command, only if it's clearly one."""
    if part.get("typed") or part.get("spoken"):
        return parse(part.get("typed") or part.get("spoken"), front_bundle)
    hyps = part.get("hyps") or []
    if not hyps:
        return None
    first = parse(hyps[0][0], front_bundle)
    if first:
        return first
    if len(hyps) > 1:  # the top two guesses agree on a command
        a, b = parse(hyps[0][0], front_bundle), parse(hyps[1][0], front_bundle)
        return a if a and b and (a.kind, a.args) == (b.kind, b.args) else None
    return None


# ----------------------------------------------------------------------------- closed-set lip-reading candidates

FIXED = (["next song", "previous song", "pause", "play", "volume up", "volume down", "scroll down", "scroll up",
          "typing mode", "chat mode", "hide", "voice on", "voice off", "mic on", "mic off", "hands free off"]
         + [f"volume {w}" for w in ("ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
                                    "ninety", "a hundred")])
BROWSER_FIXED = ["close tab", "new tab", "go back", "reload"]
# never done instantly: these go to Claude, which asks first
RISKY = re.compile(r"\b(send|post|delete|remove|buy|purchase|pay|submit|publish|confirm|order|archive|leave|block|"
                   r"unsubscribe|sign out|log ?out|uninstall|quit|erase|reset|transfer|accept|decline|trash)\b", re.I)
CLICK_ROLES = {"AXButton", "AXCheckBox", "AXRadioButton", "AXTab", "AXMenuItem", "AXPopUpButton", "AXMenuButton"}
OPEN_ROLES = {"AXLink", "AXRow"}


def _label(text):
    text = (text or "").split(",")[0]  # "Nick (direct message), Online" -> "Nick"
    text = re.sub(r"\(.*?\)|[^\w\s'&-]", " ", text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    return text if 1 <= len(words) <= 4 and len(text) >= 2 and not text.isdigit() else ""


def candidates(front_bundle="", nodes=(), apps=(), shortcuts=()):
    """[(phrase, action)] the lip-reader can choose from. action: ("local", phrase) for local_commands.parse, or
    ("press", label) for an on-screen control. Phrases are spelled the way they'd be said."""
    out = [(p, ("local", p)) for p in FIXED + (BROWSER_FIXED if front_bundle in BROWSERS else [])]
    out += [(f"open {a.lower()}", ("local", f"open {a}")) for a in apps]
    out += list(shortcuts)
    seen = {p for p, _ in out}
    for n in nodes:
        lab = _label(n.get("label"))
        if not lab or RISKY.search(lab):
            continue
        verb = "click" if n["role"] in CLICK_ROLES else "open" if n["role"] in OPEN_ROLES else None
        if verb and f"{verb} {lab}" not in seen:
            seen.add(f"{verb} {lab}")
            out.append((f"{verb} {lab}", ("press", n.get("label"))))
    return out[:90]
