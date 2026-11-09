"""Saved shortcuts: tasks Claude finished once, replayed instantly on the Mac without a model.

Claude saves one after doing a task you're likely to repeat, with {blanks} for the parts that change:
  ./macctl remember "open my DMs with {name}" "focus Discord" "key cmd+k" "type {name}" "wait 0.5" "key return"
A step starting "ask:" (sending, deleting, buying ...) still asks you first when replayed. Stored in
data/silent_chat/shortcuts.json (personal: git-ignored).

  ./macctl shortcuts            list them        ./macctl forget <number|phrase>
  ./macctl rename <number|phrase> <new phrase>
"""
import json
import re
from datetime import datetime
from pathlib import Path

FILE = Path(__file__).resolve().parent / "data" / "silent_chat" / "shortcuts.json"
BLANK = re.compile(r"\{(\w+)\}")


def load():
    try:
        return json.loads(FILE.read_text())
    except (OSError, ValueError):
        return []


def save(items):
    FILE.parent.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps(items, indent=1))


def norm(text):
    t = re.sub(r"[^\w\s{}'-]", " ", text.lower())
    return re.sub(r"\s+", " ", t).strip()


def remember(phrase, steps):
    phrase = norm(phrase)
    if not phrase or not steps:
        raise ValueError('usage: remember "<phrase with {blanks}>" "<step>" ["<step>" ...]')
    missing = set(BLANK.findall(" ".join(steps))) - set(BLANK.findall(phrase))
    if missing:
        raise ValueError(f"The steps use {{{', '.join(sorted(missing))}}}, which the phrase doesn't have.")
    items = [s for s in load() if s["phrase"] != phrase]
    items.append({"phrase": phrase, "steps": steps, "uses": 0, "values": {},
                  "created": f"{datetime.now():%Y-%m-%d %H:%M}"})
    save(items)
    asks = sum(s.startswith("ask:") for s in steps)
    return f"Saved shortcut {len(items)}: \"{phrase}\" ({len(steps)} steps{f', asks before {asks}' if asks else ''})."


def find(ref, items):
    if ref.isdigit() and 1 <= int(ref) <= len(items):
        return int(ref) - 1
    ref = norm(ref)
    for i, s in enumerate(items):
        if s["phrase"] == ref or ref in s["phrase"]:
            return i
    return None


def listing():
    items = load()
    if not items:
        return "No saved shortcuts yet."
    return "\n".join(f"{i}. \"{s['phrase']}\" -> {' / '.join(s['steps'])} (used {s['uses']}x)"
                     for i, s in enumerate(items, 1))


def forget(ref):
    items = load()
    i = find(ref, items)
    if i is None:
        return f"No shortcut matching {ref!r}."
    gone = items.pop(i)
    save(items)
    return f"Deleted \"{gone['phrase']}\"."


def rename(ref, new):
    items = load()
    i = find(ref, items)
    if i is None:
        return f"No shortcut matching {ref!r}."
    new = norm(new)
    if set(BLANK.findall(new)) != set(BLANK.findall(items[i]["phrase"])):
        return f"The new phrase must keep the same blanks: {', '.join(BLANK.findall(items[i]['phrase'])) or 'none'}."
    items[i]["phrase"] = new
    save(items)
    return f"Renamed to \"{new}\"."


def match(text):
    """A request -> (shortcut, {blank: value}) if it matches a saved phrase, else None."""
    t = re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", " ", text)).strip()  # (keeps your capitals for typed parts)
    for s in load():
        pattern = re.escape(s["phrase"]).replace(r"\{", "{").replace(r"\}", "}")
        pattern = BLANK.sub(lambda m: f"(?P<{m.group(1)}>.+?)", pattern)
        m = re.fullmatch(pattern, t, re.IGNORECASE)
        if m:
            return s, m.groupdict()
    return None


def fill(steps, values):
    return [BLANK.sub(lambda m: values.get(m.group(1), m.group(0)), step) for step in steps]


def used(phrase, values):
    """Count a replay and remember the values (they become closed-set candidates next time)."""
    items = load()
    for s in items:
        if s["phrase"] == phrase:
            s["uses"] += 1
            s["last_used"] = f"{datetime.now():%Y-%m-%d %H:%M}"  # (what the self-improver keeps or removes by)
            for k, v in values.items():
                seen = s["values"].setdefault(k, [])
                if v not in seen:
                    seen.insert(0, v)
                    del seen[8:]
    save(items)


def candidates(names=()):
    """Phrases to offer the closed-set lip-reader: each shortcut filled with values used before (and, for a
    {name} blank, names on screen)."""
    out = []
    for s in load():
        blanks = BLANK.findall(s["phrase"])
        if not blanks:
            out.append((s["phrase"], ("shortcut", (s["phrase"], {}))))
            continue
        if len(blanks) > 1:
            continue  # (two blanks would multiply the list; those go through the text matcher only)
        b = blanks[0]
        values = list(dict.fromkeys(s["values"].get(b, []) + list(names)))[:12]
        for v in values:
            out.append((s["phrase"].replace("{" + b + "}", v), ("shortcut", (s["phrase"], {b: v}))))
    return out


def failed(phrase):
    """Count a replay that failed (the self-improver tweaks shortcuts that keep failing)."""
    items = load()
    for s in items:
        if s["phrase"] == phrase:
            s["fails"] = s.get("fails", 0) + 1
            s["last_used"] = f"{datetime.now():%Y-%m-%d %H:%M}"
    save(items)
