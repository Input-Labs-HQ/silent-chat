"""Computer control for silent chat's agent: use any app the way you would, the cheapest way first.

  ui [App] [--find words] [--more]    the window's controls and text as a short numbered list (Accessibility)
  press <#id | "label">               press a button / link / row / tab (no mouse movement)
  set <#id | "label"> <text>          replace the text in a field
  click <#id | x y> [--right|--double]  mouse click: an element, or a point in the last `see` image (pixels)
  type <text>                         type at the cursor          key <combo> ...   e.g. cmd+k  return  down*3
  menu <App> ["File > Export…"]       list a menu (with shortcuts) or choose an item
  scroll <up|down|left|right> [n] [--at #id | x y]
  see [App] [--zoom x y w h]          small screenshot of the window (or screen); --zoom: a part of the last one
  wait <seconds>                      wait-for <text> [seconds]: until that text shows up in the window
  skill [App]                         tips for an app        learn <App> <tip>: save a tip for next time

Every action prints where the focus ended up, so you rarely need to look again. Password fields are refused, and
after the user presses \\ (stop) every action is refused until their next message.
"""
import json
import re
import subprocess
import time
from pathlib import Path

import ApplicationServices as AX
import CoreFoundation as CF
import Quartz
from AppKit import NSRunningApplication, NSWorkspace

ROOT = Path(__file__).resolve().parent
STOP_FILE = ROOT / "data" / "silent_chat" / "agent_stop"
SKILLS = ROOT / "skills"
SCREENS = Path.cwd() / "screens"
STATE = SCREENS / ".control.json"  # ids from the last `ui`, and the last screenshot's coordinates

ATTRS = ["AXRole", "AXSubrole", "AXTitle", "AXDescription", "AXValue", "AXPlaceholderValue", "AXChildren",
         "AXPosition", "AXSize", "AXFocused", "AXEnabled"]
ROLES = {"AXButton": "button", "AXLink": "link", "AXTextField": "field", "AXTextArea": "text area",
         "AXSearchField": "search", "AXComboBox": "combo box", "AXCheckBox": "checkbox", "AXRadioButton": "option",
         "AXPopUpButton": "popup", "AXMenuButton": "menu button", "AXMenuItem": "menu item", "AXSlider": "slider",
         "AXDisclosureTriangle": "disclosure", "AXRow": "row", "AXTab": "tab", "AXIncrementor": "stepper",
         "AXSwitch": "switch", "AXDockItem": "dock item", "AXMenuBarItem": "menu"}
FIELDS = {"AXTextField", "AXTextArea", "AXSearchField", "AXComboBox"}
TEXTS = {"AXStaticText", "AXHeading"}
CHROMIUM = {"com.google.Chrome", "com.google.Chrome.canary", "com.brave.Browser", "com.microsoft.edgemac",
            "company.thebrowser.Browser", "com.vivaldi.Vivaldi", "com.operasoftware.Opera"}
_AXVALUE = AX.AXValueGetTypeID()
AX.AXUIElementSetMessagingTimeout(AX.AXUIElementCreateSystemWide(), 1.5)  # a hung app can't hang us


class Fail(Exception):
    pass


# ----------------------------------------------------------------------------- small helpers

def _val(v):
    """AX values -> Python: points/sizes as tuples, errors/unsupported as None."""
    if v is None:
        return None
    try:
        if CF.CFGetTypeID(v) == _AXVALUE:
            t = AX.AXValueGetType(v)
            if t == AX.kAXValueCGPointType:
                p = AX.AXValueGetValue(v, t, None)[1]
                return (p.x, p.y)
            if t == AX.kAXValueCGSizeType:
                s = AX.AXValueGetValue(v, t, None)[1]
                return (s.width, s.height)
            return None
    except Exception:
        pass
    return v


def _get(el, attr):
    err, v = AX.AXUIElementCopyAttributeValue(el, attr, None)
    return _val(v) if err == 0 else None


def _attrs(el):
    err, vals = AX.AXUIElementCopyMultipleAttributeValues(el, ATTRS, 0, None)
    if err != 0 or vals is None:
        return {}
    return {k: _val(v) for k, v in zip(ATTRS, vals)}


def _text(v, n=90):
    if not isinstance(v, str):
        return ""
    v = " ".join(re.sub("[\u200b-\u200f\ufeff]", "", v).split())  # (web text fields carry zero-width marks)
    return v if len(v) <= n else v[:n - 1] + "…"


def stop_check():
    if STOP_FILE.exists():
        raise Fail("STOPPED: the user pressed \\ to stop. Don't do anything more; tell them you stopped.")


def load_state():
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def save_state(**kw):
    st = load_state()
    st.update(kw)
    SCREENS.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(st))


def app_named(name=None):
    """A running app by name (exact > starts with > contains), or the frontmost app."""
    ws = NSWorkspace.sharedWorkspace()
    if not name:
        return NSRunningApplication.runningApplicationWithProcessIdentifier_(front_pid())
    apps = [a for a in ws.runningApplications() if a.activationPolicy() == 0 and a.localizedName()]
    low = name.lower()
    for test in (lambda n: n == low, lambda n: n.startswith(low), lambda n: low in n):
        hits = [a for a in apps if test(a.localizedName().lower())]
        if hits:
            return hits[0]
    raise Fail(f"{name} isn't running (open it first).")


def enable_web_tree(a):
    """Chrome and Electron apps (Discord, Slack, VS Code ...) only build their accessibility tree when asked, and
    then lazily. True the first time for this app (the caller should give it a moment)."""
    seen = load_state().get("web", [])
    if a.processIdentifier() in seen:
        return False
    el = AX.AXUIElementCreateApplication(a.processIdentifier())
    bundle = a.bundleIdentifier() or ""
    path = a.bundleURL().path() if a.bundleURL() else ""
    ok = False
    if bundle in CHROMIUM:
        ok = AX.AXUIElementSetAttributeValue(el, "AXEnhancedUserInterface", True) == 0
    elif Path(path, "Contents/Frameworks/Electron Framework.framework").exists():
        ok = AX.AXUIElementSetAttributeValue(el, "AXManualAccessibility", True) == 0
    save_state(web=(seen + [a.processIdentifier()])[-30:])
    return ok


def front_pid():
    """The pid of the app that really has keyboard focus now (live, unlike NSWorkspace in a short-lived process)."""
    app = _get(AX.AXUIElementCreateSystemWide(), "AXFocusedApplication")
    if app is not None:
        err, pid = AX.AXUIElementGetPid(app, None)
        if err == 0:
            return pid
    return NSWorkspace.sharedWorkspace().frontmostApplication().processIdentifier()


def bring_front(a):
    """Bring an app to the front and check that it really is: macOS often ignores plain activation requests from
    background processes. Remembers it as the target for key presses and typing."""
    pid = a.processIdentifier()
    steps = [lambda: AX.AXUIElementSetAttributeValue(AX.AXUIElementCreateApplication(pid), "AXFrontmost", True),
             lambda: a.activateWithOptions_(2),
             lambda: subprocess.run(["open", "-b", a.bundleIdentifier()] if a.bundleIdentifier() else
                                    ["open", "-a", a.bundleURL().path()], capture_output=True)]
    ok = front_pid() == pid
    for step in ([] if ok else steps):
        step()
        for _ in range(12):
            if front_pid() == pid:
                ok = True
                break
            time.sleep(0.05)
        if ok:
            break
    if ok:
        save_state(target={"pid": pid, "name": a.localizedName(), "t": time.time()})
    return ok


def check_target():
    """Keys and typing only go to the app Claude last brought forward (if it did so in the last 2 minutes)."""
    t = load_state().get("target")
    if t and time.time() - t["t"] < 120 and front_pid() != t["pid"]:
        now = NSRunningApplication.runningApplicationWithProcessIdentifier_(front_pid())
        raise Fail(f"{t['name']} isn't in front ({now.localizedName() if now else 'another app'} is), so I sent "
                   f"nothing. Run focus {t['name']!r} first.")


def window_of(a):
    app = AX.AXUIElementCreateApplication(a.processIdentifier())
    return _get(app, "AXFocusedWindow") or _get(app, "AXMainWindow") or (_get(app, "AXWindows") or [None])[0]


def focused():
    """(app name, window title, focused element description) right now: the cheap way to check an action."""
    a = NSRunningApplication.runningApplicationWithProcessIdentifier_(front_pid())
    app = AX.AXUIElementCreateApplication(a.processIdentifier())
    win = _get(app, "AXFocusedWindow")
    el = _get(app, "AXFocusedUIElement")
    what = ""
    if el is not None:
        n = _attrs(el)
        role = ROLES.get(n.get("AXRole"), (n.get("AXRole") or "").removeprefix("AX").lower())
        label = _text(n.get("AXTitle") or n.get("AXDescription") or n.get("AXPlaceholderValue"), 50)
        what = f"{role} {label!r}" if label else role
        if n.get("AXRole") in FIELDS and isinstance(n.get("AXValue"), str) and n["AXValue"]:
            what += f" = {_text(n['AXValue'], 60)!r}"
    title = _text(_get(win, "AXTitle"), 70) if win is not None else ""
    return a.localizedName(), title, what


def report(prefix="ok"):
    time.sleep(0.25)  # let the app react
    app, title, what = focused()
    return f"{prefix}. Now: {app}" + (f" - {title!r}" if title else "") + (f"; focus: {what}" if what else "")


# ----------------------------------------------------------------------------- reading the window

def walk(a, budget_nodes=3500, budget_s=2.5):
    """The focused window as a list of nodes in reading order: dicts with role, label, value, path, center."""
    win = window_of(a)
    if win is None:
        raise Fail(f"{a.localizedName()} has no window open.")
    wa = _attrs(win)
    wx, wy = wa.get("AXPosition") or (0, 0)
    ww, wh = wa.get("AXSize") or (10000, 10000)
    nodes, t_end = [], time.monotonic() + budget_s
    pruned = []  # (debugging) subtrees skipped as out of view

    def visit(el, path, depth, inside):
        if len(nodes) >= budget_nodes or time.monotonic() > t_end or depth > 60:
            return
        n = _attrs(el)
        if not n:
            return
        pos, size = n.get("AXPosition"), n.get("AXSize")
        if pos and size and size[0] > 0 and size[1] > 0:  # skip what's scrolled out of the window
            if pos[0] > wx + ww or pos[1] > wy + wh or pos[0] + size[0] < wx or pos[1] + size[1] < wy:
                pruned.append((n.get("AXRole"), [round(v) for v in pos + size]))
                return
        role = n.get("AXRole") or ""
        node = {"role": role, "sub": n.get("AXSubrole") or "", "title": _text(n.get("AXTitle")),
                "desc": _text(n.get("AXDescription")), "value": n.get("AXValue"),
                "ph": _text(n.get("AXPlaceholderValue")), "focused": bool(n.get("AXFocused")),
                "enabled": n.get("AXEnabled") is not False, "path": path, "inside": inside, "texts": [],
                "center": (round(pos[0] + size[0] / 2), round(pos[1] + size[1] / 2)) if pos and size else None}
        nodes.append(node)
        acting = inside if inside is not None else (len(nodes) - 1 if role in ROLES else None)
        if role in TEXTS and inside is not None and isinstance(node["value"], str):
            nodes[inside]["texts"].append(_text(node["value"], 60))  # e.g. a button's label lives in its text
        for i, child in enumerate(n.get("AXChildren") or []):
            visit(child, path + [i], depth + 1, acting)

    visit(win, [], 0, None)
    walk.pruned, walk.window = pruned, (wx, wy, ww, wh)
    return wa, nodes


WINDOW_BUTTONS = {"AXCloseButton": "close window", "AXMinimizeButton": "minimize window",
                  "AXZoomButton": "zoom window", "AXFullScreenButton": "full screen"}


def label(n):
    for k in ("title", "desc"):
        if n[k]:
            return n[k]
    if n["sub"] in WINDOW_BUTTONS:
        return WINDOW_BUTTONS[n["sub"]]
    if n["texts"]:
        return _text(" ".join(n["texts"]), 70)
    if n["role"] in TEXTS and isinstance(n["value"], str):
        return _text(n["value"], 100)
    return n["ph"]


def cmd_ui(args):
    more, stats = "--more" in args, "--stats" in args
    args = [x for x in args if x not in ("--more", "--stats")]
    find = ""
    if "--find" in args:
        i = args.index("--find")
        find, args = " ".join(args[i + 1:]).lower(), args[:i]
    a = app_named(" ".join(args) if args else None)
    fresh = enable_web_tree(a)
    if fresh:
        time.sleep(1.0)  # first time for this app: it builds its web tree in the background
    t0 = time.monotonic()
    wa, nodes = walk(a)
    for _ in range(2):  # a web view that's still filling in
        if not any(n["role"] == "AXWebArea" for n in nodes) or len(nodes) >= 120:
            break
        time.sleep(0.6)
        wa, nodes = walk(a)
    lines, ids, texts_shown, last_text = [], {}, 0, None
    for n in nodes:
        lab = label(n)
        if n["role"] in ROLES and n["inside"] is None or n["role"] in ROLES and n["role"] in FIELDS:
            desc = ROLES[n["role"]]
            if n["sub"] == "AXSecureTextField":
                desc = "password field"
            if n["role"] in FIELDS:
                val = _text(n["value"], 60) if isinstance(n["value"], str) else ""
                line = f"{desc} {lab!r}" + (f" = {val!r}" if val else "")
            elif n["role"] in ("AXCheckBox", "AXRadioButton", "AXSwitch") and n["value"] in (0, 1):
                line = f"{desc} {lab!r} [{'x' if n['value'] else ' '}]"
            else:
                line = f"{desc} {lab!r}" if lab else desc
            if not n["enabled"]:
                line += " (disabled)"
            if n["focused"]:
                line += " (focused)"
            if find and not all(w in line.lower() for w in find.split()):
                continue
            k = str(len(ids) + 1)
            ids[k] = {"path": n["path"], "role": n["role"], "label": lab, "center": n["center"]}
            lines.append(f"#{k} {line}")
        elif n["role"] in TEXTS and n["inside"] is None and lab and lab != last_text:
            last_text = lab
            if find and not all(w in lab.lower() for w in find.split()):
                continue
            if texts_shown < (400 if more else 120):
                lines.append(f"   {lab!r}")
                texts_shown += 1
    save_state(ui={"pid": a.processIdentifier(), "app": a.localizedName(), "items": ids})
    limit = 400 if more else 140
    if stats:  # (debugging: sizes only, no content)
        from collections import Counter
        roles = Counter(n["role"] for n in nodes)
        shown = "\n".join(lines[:limit])
        return (f"{a.localizedName()}: {len(nodes)} nodes in {time.monotonic() - t0:.2f} s (first time: {fresh}); "
                f"{len(ids)} controls, {len(lines)} lines; output {len(shown)} chars (~{len(shown) // 4} tokens); "
                f"roles {dict(roles.most_common(8))}")
    head = f"{a.localizedName()} - {_text(wa.get('AXTitle'), 60)!r}: {len(ids)} controls"
    if not lines:
        return head + (f", nothing matching {find!r}" if find else
                       ". Nothing readable here (a canvas or image?): use `see` instead.")
    extra = f"\n(+{len(lines) - limit} more lines: use --find <words> or --more)" if len(lines) > limit else ""
    return head + "\n" + "\n".join(lines[:limit]) + extra


# ----------------------------------------------------------------------------- what's in front, in brief

SKIP_PIDS = set()  # never described (silent chat's own windows)
KINDS = [("field", "Fields", 5), ("button", "Buttons", 12), ("tab", "Tabs", 6), ("menu button", "Menus", 4),
         ("popup", "Menus", 4), ("checkbox", "Toggles", 4), ("switch", "Toggles", 4), ("option", "Options", 4),
         ("row", "Rows", 4), ("link", "Links", 8)]
NOISE = re.compile(r"^(more options for|more actions for|advertisement|report ad|adchoices)", re.I)


def _selected(el, n):
    """Selected text in the focused element: plain fields, or a web page (text markers)."""
    sel = _get(el, "AXSelectedText")
    if isinstance(sel, str) and sel.strip():
        return sel
    try:
        err, rng = AX.AXUIElementCopyAttributeValue(el, "AXSelectedTextMarkerRange", None)
        if err == 0 and rng is not None:
            err, s = AX.AXUIElementCopyParameterizedAttributeValue(el, "AXStringForTextMarkerRange", rng, None)
            if err == 0 and isinstance(s, str):
                return s
    except Exception:
        pass
    return ""


def looking_at(x, y, radius):
    """What's at screen point (x, y), give or take `radius` points: the nearest controls by distance. -> line or ''."""
    err, el = AX.AXUIElementCopyElementAtPosition(AX.AXUIElementCreateSystemWide(), x, y, None)
    if err != 0 or el is None:
        return ""
    err, pid = AX.AXUIElementGetPid(el, None)
    a = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid) if err == 0 else None
    if a is None or pid in SKIP_PIDS:
        return ""
    under = _attrs(el)
    under_text = _text(under.get("AXValue") if isinstance(under.get("AXValue"), str) else
                       under.get("AXTitle") or under.get("AXDescription"), 80)
    near = []
    try:
        _, nodes = walk(a, budget_nodes=1500, budget_s=0.3)
    except Fail:
        nodes = []
    for n in nodes:
        if n["role"] not in ROLES or n["center"] is None or n["sub"] in WINDOW_BUTTONS:
            continue
        if n["inside"] is not None and n["role"] not in FIELDS:
            continue
        lab = _text(label(n), 40)
        d = ((n["center"][0] - x) ** 2 + (n["center"][1] - y) ** 2) ** 0.5
        if lab and d <= radius * 1.5:
            near.append((d, f"{ROLES[n['role']]} {lab!r}"))
    near = [w for _, w in sorted(near)[:4]]
    parts = [f"Looking at ({a.localizedName()}, give or take {radius:.0f} pt):"]
    if under_text:
        parts.append(f"under their gaze {under_text!r};")
    parts.append(("nearest controls, closest first: " + "; ".join(near)) if near else "no control nearby")
    return " ".join(parts)


def context(budget=650, walk_s=0.35, pid=None, gaze=None):
    """What's in front of the user right now in at most ~budget characters (~budget/2.5 tokens, so <= ~260): the app and window,
    the focused field, any selected text, the page address, and the window's main controls by label. Sent with
    each message so Claude can act on "this" without looking first. Read-only; '' when there's nothing to say."""
    pid = pid or front_pid()
    a = None if pid in SKIP_PIDS else NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
    if a is None:
        return ""
    app = AX.AXUIElementCreateApplication(pid)
    win = window_of(a)
    title = _text(_get(win, "AXTitle"), 80) if win is not None else ""
    head = [f"Front: {a.localizedName()}" + (f" - {title!r}" if title else "" if win is not None else " (no window)")]
    el = _get(app, "AXFocusedUIElement")
    if el is not None:
        n = _attrs(el)
        secure = n.get("AXSubrole") == "AXSecureTextField"
        if n.get("AXRole") in ROLES:
            lab = _text(n.get("AXTitle") or n.get("AXDescription") or n.get("AXPlaceholderValue"), 40)
            focus = "password field" if secure else f"{ROLES[n['AXRole']]} {lab!r}" if lab else ROLES[n["AXRole"]]
            if not secure and n.get("AXRole") in FIELDS and isinstance(n.get("AXValue"), str) and n["AXValue"].strip():
                focus += f" containing {_text(n['AXValue'], 80)!r}"
            head.append(f"Focus: {focus}")
        sel = "" if secure else _selected(el, n)
        if sel.strip():
            head.append(f"Selected: {_text(sel, 200)!r}")
    if gaze is not None:  # (x, y, radius): what they were looking at as they spoke
        try:
            g = looking_at(*gaze)
        except Exception:
            g = ""
        if g:
            head.append(g)
    if win is None:
        return "\n".join(head)
    enable_web_tree(a)  # (Chrome / Electron: the first time, their web content shows from the next message on)
    try:
        _, nodes = walk(a, budget_nodes=1500, budget_s=walk_s)
    except Fail:
        nodes = []
    groups, seen, headings, url, total = {}, set(), [], "", 0
    for n in nodes:
        r = n["role"]
        if r == "AXWebArea" and not url:
            u = _get(follow(win, n["path"]), "AXURL")
            u = str(u.absoluteString()) if hasattr(u, "absoluteString") else str(u or "")
            url = re.sub(r"[?#].*", "", u)[:100] if u.startswith("http") else ""  # (no query: it can hold tokens)
        if r == "AXHeading" and len(headings) < 3:
            h = _text(label(n), 50)
            if h and h not in headings:
                headings.append(h)
        if r not in ROLES or n["sub"] in WINDOW_BUTTONS or not n["enabled"] or (n["inside"] is not None and r not in FIELDS):
            continue
        kind = "field" if r in FIELDS else ROLES[r]
        lab = _text(label(n), 40)
        if n["sub"] == "AXSecureTextField":
            lab = (lab or "password") + " (password)"
        elif kind == "field" and isinstance(n["value"], str) and n["value"].strip():
            lab = f"{lab!r} = {_text(n['value'], 40)!r}" if lab else repr(_text(n["value"], 40))
        if not lab or (kind, lab) in seen or NOISE.match(lab):
            continue
        seen.add((kind, lab))
        total += 1
        groups.setdefault(kind, []).append(lab)
    lines = head + ([f"Page: {url}"] if url else [])
    used, shown = sum(len(x) + 1 for x in lines), 0
    merged = {}
    for kind, name, cap in KINDS:
        merged.setdefault(name, []).extend(groups.get(kind, [])[:cap])
    for name, labs in merged.items():
        if not labs:
            continue
        line = f"{name}: "
        for lab in labs:
            if used + len(line) + len(lab) + 2 > budget:
                break
            line += ("; " if not line.endswith(": ") else "") + lab
            shown += 1
        if not line.endswith(": "):
            lines.append(line)
            used += len(line) + 1
    more = f"(+{total - shown} more controls: ui --find <words>)" if total > shown else ""
    if headings and used + len(more) + 60 < budget:
        lines.append("Headings: " + "; ".join(headings))
    text = "\n".join(lines + ([more] if more else []))
    return text if len(text) <= budget + 60 else text[:budget + 59] + "…"  # (hard cap: ~290 tokens at most)


def cmd_context(args):
    """context [App | --all]: the short summary of what's in front (what's sent with each message)."""
    if args[:1] == ["--all"]:  # (measuring: every app with a window)
        out = []
        for a in NSWorkspace.sharedWorkspace().runningApplications():
            if a.activationPolicy() == 0 and a.processIdentifier() not in SKIP_PIDS:
                t0 = time.monotonic()
                c = context(pid=a.processIdentifier())
                out.append(f"== {a.localizedName()}: {len(c)} chars (~{len(c) * 2 // 5} tokens), "
                           f"{(time.monotonic() - t0) * 1000:.0f} ms\n{c}")
        return "\n".join(out)
    t0 = time.monotonic()
    c = context(pid=app_named(" ".join(args)).processIdentifier() if args else None)
    return f"{c}\n({len(c)} chars, ~{len(c) * 2 // 5} tokens, {(time.monotonic() - t0) * 1000:.0f} ms)"


# ----------------------------------------------------------------------------- finding an element again

def resolve(ref):
    """#id from the last `ui`, or a label to look for in the front app's window. -> (element, info)."""
    st = load_state().get("ui", {})
    if ref.startswith("#"):
        info = st.get("items", {}).get(ref[1:])
        if not info:
            raise Fail(f"No {ref} in the last `ui` list: run `ui` again.")
        a = NSRunningApplication.runningApplicationWithProcessIdentifier_(st["pid"])
        if a is None:
            raise Fail("That app has quit.")
        el = follow(window_of(a), info["path"])
        if el is not None and _get(el, "AXRole") == info["role"]:
            return el, info, a
        ref = info["label"] or ref  # the window changed: look for it by its label instead
        if not ref or ref.startswith("#"):
            raise Fail("That element is gone: run `ui` again.")
    a = NSRunningApplication.runningApplicationWithProcessIdentifier_(st["pid"]) if st.get("pid") else None
    a = a or NSRunningApplication.runningApplicationWithProcessIdentifier_(front_pid())
    enable_web_tree(a)
    _, nodes = walk(a)
    want = ref.strip("'\"").lower()
    best = None
    for n in nodes:
        if n["role"] not in ROLES:
            continue
        lab = (label(n) or "").lower()
        score = 3 if lab == want else 2 if lab.startswith(want) else 1 if want in lab else 0
        if score and (best is None or score > best[0]):
            best = (score, n)
    if best is None:
        raise Fail(f"Nothing called {ref!r} in {a.localizedName()}: check `ui --find`.")
    n = best[1]
    el = follow(window_of(a), n["path"])
    if el is None:
        raise Fail(f"{ref!r} disappeared: try again.")
    return el, {"role": n["role"], "label": label(n), "center": n["center"]}, a


def follow(el, path):
    """The element at these child indices under `el` (None if the window changed)."""
    for i in path:
        kids = (_get(el, "AXChildren") or []) if el is not None else []
        el = kids[i] if i < len(kids) else None
    return el


# ----------------------------------------------------------------------------- acting

def _post_mouse(kind, x, y, button=Quartz.kCGMouseButtonLeft, clicks=1):
    ev = Quartz.CGEventCreateMouseEvent(None, kind, (x, y), button)
    Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGMouseEventClickState, clicks)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)


def mouse_click(x, y, right=False, double=False):
    """Click a screen point (top-left origin, in points), then put the pointer back where you had it."""
    home = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    down, up = ((Quartz.kCGEventRightMouseDown, Quartz.kCGEventRightMouseUp) if right else
                (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp))
    button = Quartz.kCGMouseButtonRight if right else Quartz.kCGMouseButtonLeft
    _post_mouse(Quartz.kCGEventMouseMoved, x, y)
    time.sleep(0.03)
    for c in ((1, 2) if double else (1,)):
        _post_mouse(down, x, y, button, c)
        _post_mouse(up, x, y, button, c)
        time.sleep(0.05)
    time.sleep(0.05)
    Quartz.CGWarpMouseCursorPosition(home)
    Quartz.CGAssociateMouseAndMouseCursorPosition(True)


def cmd_press(args):
    stop_check()
    if not args:
        raise Fail("usage: press <#id | label>")
    el, info, a = resolve(" ".join(args))
    if AX.AXUIElementPerformAction(el, "AXPress") != 0:  # (works without bringing the app forward)
        if info.get("role") == "AXRow" and AX.AXUIElementSetAttributeValue(el, "AXSelected", True) == 0:
            return report(f"selected {info['label']!r}")
        if not info.get("center") or not bring_front(a):
            raise Fail(f"Can't press {info['label']!r}.")
        mouse_click(*info["center"])
    return report(f"pressed {info['label']!r}")


def _secure_focus():
    el = _get(AX.AXUIElementCreateApplication(front_pid()), "AXFocusedUIElement")
    return el is not None and _get(el, "AXSubrole") == "AXSecureTextField"


def cmd_type(args):
    stop_check()
    text = " ".join(args)
    if _secure_focus():
        raise Fail("Refused: the cursor is in a password field. The user must type passwords themselves.")
    check_target()
    import typer
    typer.type_text(text)
    return report(f"typed {len(text)} characters")


def cmd_set(args):
    stop_check()
    if len(args) < 2:
        raise Fail("usage: set <#id | label> <text>")
    el, info, a = resolve(args[0])
    if _get(el, "AXSubrole") == "AXSecureTextField":
        raise Fail("Refused: that's a password field. The user must type passwords themselves.")
    text = " ".join(args[1:])
    if AX.AXUIElementSetAttributeValue(el, "AXValue", text) == 0 and _get(el, "AXValue") == text:
        return report(f"set {info['label']!r}")  # written directly: no keyboard, no focus change
    if not bring_front(a):  # web-style fields: click in and type, but only into the right app
        raise Fail(f"Couldn't bring {a.localizedName()} to the front, so I didn't type.")
    if AX.AXUIElementSetAttributeValue(el, "AXFocused", True) != 0 and info.get("center"):
        mouse_click(*info["center"])
    time.sleep(0.1)
    if front_pid() != a.processIdentifier():
        raise Fail(f"{a.localizedName()} lost focus, so I didn't type.")
    press_keys(["cmd+a"])
    import typer
    typer.type_text(text)
    return report(f"set {info['label']!r}")


KEYS = {"a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11, "q": 12, "w": 13,
        "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25,
        "7": 26, "-": 27, "8": 28, "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35, "return": 36,
        "enter": 36, "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44, "n": 45, "m": 46,
        ".": 47, "tab": 48, "space": 49, "`": 50, "delete": 51, "backspace": 51, "escape": 53, "esc": 53,
        "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100, "f9": 101, "f10": 109,
        "f11": 103, "f12": 111, "home": 115, "pageup": 116, "forwarddelete": 117, "end": 119, "pagedown": 121,
        "left": 123, "right": 124, "down": 125, "up": 126}
MODS = {"cmd": Quartz.kCGEventFlagMaskCommand, "command": Quartz.kCGEventFlagMaskCommand,
        "shift": Quartz.kCGEventFlagMaskShift, "opt": Quartz.kCGEventFlagMaskAlternate,
        "option": Quartz.kCGEventFlagMaskAlternate, "alt": Quartz.kCGEventFlagMaskAlternate,
        "ctrl": Quartz.kCGEventFlagMaskControl, "control": Quartz.kCGEventFlagMaskControl,
        "fn": Quartz.kCGEventFlagMaskSecondaryFn}


def press_keys(combos):
    import typer
    for combo in combos:
        combo, _, times = combo.lower().partition("*")
        *mods, key = combo.split("+") if combo != "+" else ["+"]
        if key not in KEYS or any(m not in MODS for m in mods):
            raise Fail(f"Unknown key {combo!r} (e.g. cmd+k, return, escape, tab, down, f5).")
        flags = 0
        for m in mods:
            flags |= MODS[m]
        for _ in range(int(times or 1)):
            typer._key(KEYS[key], True, flags)
            typer._key(KEYS[key], False, flags)
            time.sleep(0.03)


def cmd_key(args):
    stop_check()
    if not args:
        raise Fail("usage: key <combo> [combo ...]   e.g. cmd+k  return  down*3  cmd+shift+t")
    check_target()
    press_keys(args)
    return report("pressed " + " ".join(args))


def _scaled(x, y):
    """A point in the last `see` image (pixels) -> screen points."""
    shot = load_state().get("shot")
    if not shot:
        raise Fail("Take a `see` screenshot first (click x y uses its pixels), or use click #id.")
    return shot["x"] + float(x) * shot["scale"], shot["y"] + float(y) * shot["scale"]


def cmd_click(args):
    stop_check()
    right, double = "--right" in args, "--double" in args
    args = [x for x in args if not x.startswith("--")]
    if len(args) == 1:
        el, info, a = resolve(args[0])
        if not info.get("center"):
            raise Fail("That element has no position.")
        bring_front(a)
        x, y = info["center"]
        what = repr(info["label"])
    elif len(args) == 2:
        x, y = _scaled(*args)
        what = f"at {args[0]},{args[1]}"
    else:
        raise Fail("usage: click <#id | x y> [--right|--double]")
    mouse_click(x, y, right, double)
    return report(f"clicked {what}")


def cmd_scroll(args):
    stop_check()
    direction = args[0] if args else "down"
    n = int(args[1]) if len(args) > 1 and args[1].isdigit() else 5
    if "--at" in args:
        rest = args[args.index("--at") + 1:]
        x, y = resolve(rest[0])[1]["center"] if rest[0].startswith("#") else _scaled(rest[0], rest[1])
    else:
        a = NSRunningApplication.runningApplicationWithProcessIdentifier_(front_pid())
        wa = _attrs(window_of(a))
        (px, py), (w, h) = wa.get("AXPosition", (0, 0)), wa.get("AXSize", (800, 600))
        x, y = px + w / 2, py + h / 2
    dy = {"up": n, "down": -n}.get(direction, 0)
    dx = {"left": n, "right": -n}.get(direction, 0)
    home = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    Quartz.CGWarpMouseCursorPosition((x, y))
    time.sleep(0.03)
    ev = Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitLine, 2, dy, dx)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
    time.sleep(0.1)
    Quartz.CGWarpMouseCursorPosition(home)
    Quartz.CGAssociateMouseAndMouseCursorPosition(True)
    return f"scrolled {direction} {n}"


def cmd_menu(args):
    if not args:
        raise Fail('usage: menu <App> ["File > Export..."]')
    a = app_named(args[0])
    bar = _get(AX.AXUIElementCreateApplication(a.processIdentifier()), "AXMenuBar")
    if bar is None:
        raise Fail(f"{a.localizedName()} has no menu bar I can read.")
    tops = (_get(bar, "AXChildren") or [])[1:]  # skip the Apple menu

    def norm(s):
        return re.sub(r"[.…]+$", "", (s or "").strip().lower())

    def items_of(el):
        menus = _get(el, "AXChildren") or []
        return _get(menus[0], "AXChildren") or [] if menus else []

    if len(args) == 1:
        return f"{a.localizedName()} menus: " + ", ".join(_get(t, "AXTitle") or "?" for t in tops)
    parts = [p for p in re.split(r"\s*>\s*", " ".join(args[1:])) if p]
    level, chosen = tops, None
    for depth, part in enumerate(parts):
        want = norm(part)
        match = [el for el in level if norm(_get(el, "AXTitle")) == want] or \
                [el for el in level if norm(_get(el, "AXTitle")).startswith(want)] or \
                [el for el in level if want in norm(_get(el, "AXTitle"))]
        if not match:
            names = [t for t in (_get(el, "AXTitle") for el in level) if t]
            raise Fail(f"No {part!r} in that menu. It has: {', '.join(names)[:600]}")
        chosen = match[0]
        if depth < len(parts) - 1:
            level = items_of(chosen)
    subitems = items_of(chosen)
    if len(parts) == 1 or subitems:  # a menu or submenu: list it, with keyboard shortcuts
        rows = []
        for el in subitems:
            t = _get(el, "AXTitle")
            if not t:
                continue  # separator
            key = _get(el, "AXMenuItemCmdChar")
            mods = _get(el, "AXMenuItemCmdModifiers") or 0
            sc = ""
            if key:
                glyph = key.upper() if key.isprintable() and key.strip() else "⌫"
                sc = ("^" if mods & 4 else "") + ("⌥" if mods & 2 else "") + ("⇧" if mods & 1 else "") + \
                     ("" if mods & 8 else "⌘") + glyph
            rows.append(f"{t}{' ▸' if items_of(el) else ''}{'  ' + sc if sc else ''}"
                        f"{' (disabled)' if _get(el, 'AXEnabled') is False else ''}")
        return f"{' > '.join(parts)}: " + " | ".join(rows)
    stop_check()
    bring_front(a)
    time.sleep(0.1)
    if _get(chosen, "AXEnabled") is False:
        raise Fail(f"{parts[-1]!r} is disabled right now.")
    if AX.AXUIElementPerformAction(chosen, "AXPress") != 0:
        raise Fail(f"Couldn't choose {parts[-1]!r}.")
    return report(f"chose {' > '.join(parts)}")


# ----------------------------------------------------------------------------- looking (last resort)

def _window_bounds(a):
    info = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements, Quartz.kCGNullWindowID)
    best = None
    for w in info or []:
        if w.get("kCGWindowOwnerPID") == a.processIdentifier() and w.get("kCGWindowLayer", 1) == 0:
            b = w["kCGWindowBounds"]
            if best is None or b["Width"] * b["Height"] > best[1]["Width"] * best[1]["Height"]:
                best = (int(w["kCGWindowNumber"]), b)
    return best


def _shrink(path, max_side=1024):
    """Downscale (fewer tokens: ~900 for 1024x640) and return the image size."""
    subprocess.run(["sips", "-Z", str(max_side), "-s", "format", "jpeg", "-s", "formatOptions", "60", str(path)],
                   capture_output=True)
    r = subprocess.run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)], capture_output=True, text=True)
    nums = re.findall(r"pixel(?:Width|Height): (\d+)", r.stdout)
    return int(nums[0]), int(nums[1])


def cmd_see(args):
    SCREENS.mkdir(exist_ok=True)
    for old in sorted(SCREENS.glob("see_*.jpg"))[:-4]:  # keep the last few
        old.unlink()
    path = SCREENS / f"see_{int(time.time() * 1000)}.jpg"
    if "--zoom" in args:
        x, y, w, h = map(float, args[args.index("--zoom") + 1:][:4])
        (x0, y0), (x1, y1) = _scaled(x, y), _scaled(x + w, y + h)
        region = (x0, y0, x1 - x0, y1 - y0)
        r = subprocess.run(["screencapture", "-x", "-t", "jpg", "-R", ",".join(str(int(v)) for v in region),
                            str(path)], capture_output=True)
        origin, width_pt = (region[0], region[1]), region[2]
    else:
        a = app_named(" ".join(args)) if args else None
        found = _window_bounds(a) if a else None
        if found:
            wid, b = found
            r = subprocess.run(["screencapture", "-x", "-o", "-t", "jpg", "-l", str(wid), str(path)],
                               capture_output=True)
            origin, width_pt = (b["X"], b["Y"]), b["Width"]
        else:
            r = subprocess.run(["screencapture", "-x", "-m", "-t", "jpg", str(path)], capture_output=True)
            frame = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
            origin, width_pt = (0, 0), frame.size.width
    if r.returncode or not path.exists():
        raise Fail("Couldn't take a screenshot: Screen Recording permission is missing for Silent Chat.")
    w_px, h_px = _shrink(path)
    save_state(shot={"x": origin[0], "y": origin[1], "scale": width_pt / w_px})
    return f"Read {path.relative_to(Path.cwd())} ({w_px}x{h_px}). `click x y` takes pixels in this image."


def cmd_wait(args):
    time.sleep(min(10.0, float(args[0]) if args else 1.0))
    return "waited"


def cmd_wait_for(args):
    if not args:
        raise Fail("usage: wait-for <text> [seconds]")
    limit = float(args[-1]) if len(args) > 1 and re.fullmatch(r"[\d.]+", args[-1]) else 6.0
    text = " ".join(args[:-1] if len(args) > 1 and re.fullmatch(r"[\d.]+", args[-1]) else args).lower()
    end = time.monotonic() + min(limit, 20)
    while time.monotonic() < end:
        a = NSRunningApplication.runningApplicationWithProcessIdentifier_(front_pid())
        try:
            _, nodes = walk(a, budget_s=1.5)
        except Fail:
            nodes = []
        title = (_get(window_of(a), "AXTitle") or "").lower() if window_of(a) is not None else ""
        if text in title or any(text in (label(n) or "").lower() or
                                (isinstance(n["value"], str) and text in n["value"].lower()) for n in nodes):
            return report(f"{text!r} is showing")
        time.sleep(0.3)
    raise Fail(f"{text!r} didn't show up within {limit:.0f} s.")


# ----------------------------------------------------------------------------- app tips

def _skill_file(app):
    SKILLS.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", app.lower()).strip("-")
    return SKILLS / f"{slug}.md"


def cmd_skill(args):
    if not args:
        names = sorted(p.stem for p in SKILLS.glob("*.md"))
        return "Tips exist for: " + ", ".join(names) if names else "No app tips yet."
    path = _skill_file(" ".join(args))
    general = SKILLS / "macos.md"
    if not path.exists():
        return f"No tips for {' '.join(args)} yet." + (f" General macOS tips:\n{general.read_text()}"
                                                        if general.exists() else "")
    return path.read_text()


def cmd_learn(args):
    if len(args) < 2:
        raise Fail("usage: learn <App> <tip>")
    path = _skill_file(args[0])
    tip = " ".join(args[1:]).strip()
    if path.exists() and tip.lower() in path.read_text().lower():
        return "Already known."
    with open(path, "a") as f:
        if f.tell() == 0:
            f.write(f"# {args[0]}\n")
        f.write(f"- {tip}\n")
    return f"Saved a tip for {args[0]}."


COMMANDS = {"ui": cmd_ui, "press": cmd_press, "set": cmd_set, "type": cmd_type, "key": cmd_key, "click": cmd_click,
            "scroll": cmd_scroll, "menu": cmd_menu, "see": cmd_see, "wait": cmd_wait, "wait-for": cmd_wait_for,
            "skill": cmd_skill, "learn": cmd_learn, "context": cmd_context}
