#!/usr/bin/env python
"""macctl: the Mac actions silent chat's Claude is allowed to use.

Deliberately small and non-destructive: it can open, focus and arrange apps, look at a window, and open a web link.
It cannot type into apps, send, delete, buy, quit apps or run anything else. Claude runs it as `./macctl ...`.

  macctl apps                      running apps and their windows
  macctl open <App> [<App> ...]    open (or bring forward) apps, e.g. Todoist, Calendar, Discord
  macctl focus <App>               bring an app to the front
  macctl place <App> left|right|full|center|top|bottom
  macctl arrange <App1> <App2>     side by side: App1 left, App2 right
                                   (place/arrange first bring the app forward, take it out of full screen,
                                   un-minimise it and reopen a window if it has none)
  macctl fullscreen <App> on|off   enter or leave macOS full screen
  macctl minimize <App>            minimise its front window
  macctl restore <App>             un-minimise / bring back its window
  macctl windows <App>             where its windows are and their state (diagnostic)
  macctl look [<App>]              screenshot of the app's front window (or the whole screen); prints the PNG path.
                                   Read that file to see it.
  macctl url <https://...>         open a web link in the default browser
  macctl spotify now               what's playing (track, artist, position, volume, shuffle/repeat)
  macctl spotify play [<link>]     resume, or play a Spotify link (open.spotify.com/... or spotify:track:...,
                                   album, playlist, artist, episode, show)
  macctl spotify pause|toggle|next|previous
  macctl spotify volume <0-100|up|down>
  macctl spotify shuffle on|off    /  macctl spotify repeat on|off
  macctl spotify search <words>    open Spotify's search for those words (it can't auto-play a search)

Window placement keeps clear of the silent chat panel and uses the macOS Accessibility API directly, then reads
the window back to check it really moved. It needs Accessibility; `look` needs Screen Recording.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import ApplicationServices as AX
import Quartz
from AppKit import NSScreen, NSWorkspace

HOST = "Silent Chat" if os.environ.get("SILENT_CHAT_APP") else "Terminal"  # who needs the permissions

ROOT = Path(__file__).resolve().parent
PANEL_STATE = ROOT / "data" / "silent_chat" / "overlay.json"
PANEL_W = 300 + 24  # silent chat panel width + margins
SCREENS = Path.cwd() / "screens"  # inside Claude's working directory, where it may read them


def out(msg):
    print(msg, flush=True)


def regular_apps():
    return [a for a in NSWorkspace.sharedWorkspace().runningApplications() if a.activationPolicy() == 0]


def find_app(name):
    name_l = name.lower()
    apps = regular_apps()
    names = [(a, (a.localizedName() or "").lower()) for a in apps]
    for test in (lambda n: n == name_l, lambda n: n.startswith(name_l), lambda n: name_l in n):  # exact first
        match = next((a for a, n in names if test(n)), None)
        if match:
            return match
    return None


def wait_for(name, seconds=8):
    a = None
    for _ in range(int(seconds * 10)):
        a = find_app(name)
        if a and windows(a.localizedName()):
            return a
        time.sleep(0.1)
    return a


def windows(owner=None):
    info = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements, Quartz.kCGNullWindowID)
    wins = []
    for w in info or []:
        if w.get("kCGWindowLayer", 1) != 0:
            continue
        if owner and (w.get("kCGWindowOwnerName") or "").lower() != owner.lower():
            continue
        b = w.get("kCGWindowBounds", {})
        wins.append({"id": int(w["kCGWindowNumber"]), "app": w.get("kCGWindowOwnerName", ""),
                     "title": w.get("kCGWindowName", ""), "w": int(b.get("Width", 0)), "h": int(b.get("Height", 0))})
    return wins


def cmd_apps(args):
    for a in regular_apps():
        name = a.localizedName()
        titles = [w["title"] for w in windows(name) if w["title"]]
        out(f"{name}{' (front)' if a.isActive() else ''}" + (f": {'; '.join(titles[:4])}" if titles else ""))


def cmd_open(args):
    if not args:
        return out("usage: macctl open <App> [<App> ...]")
    for name in args:
        r = subprocess.run(["open", "-a", name], capture_output=True, text=True)
        if r.returncode:
            out(f"Couldn't open {name}: {r.stderr.strip() or 'not found'}")
            continue
        a = wait_for(name)
        if a and len(args) == 1:
            import control
            control.bring_front(a)  # (and remember it as where keys/typing should go)
        out(f"Opened {a.localizedName() if a else name}.")


def cmd_focus(args):
    a = find_app(" ".join(args)) if args else None
    if not a:
        return out(f"{' '.join(args)} isn't running.")
    import control
    if control.bring_front(a):
        out(f"{a.localizedName()} is in front.")
    else:
        raise control.Fail(f"Couldn't bring {a.localizedName()} to the front.")


def visible_area():
    """Main screen's usable area in top-left coordinates, minus the side the silent chat panel is on."""
    screen = NSScreen.mainScreen()
    full, vf = screen.frame(), screen.visibleFrame()
    x, y = vf.origin.x, full.size.height - (vf.origin.y + vf.size.height)
    w, h = vf.size.width, vf.size.height
    try:
        side = json.loads(PANEL_STATE.read_text()).get("side", "right")
    except (OSError, ValueError):
        side = "right"
    if side == "left":
        x += PANEL_W
    w -= PANEL_W
    return int(x), int(y), int(w), int(h)


def _get(el, attr):
    err, value = AX.AXUIElementCopyAttributeValue(el, attr, None)
    return value if err == 0 else None


def _set(el, attr, value):
    return AX.AXUIElementSetAttributeValue(el, attr, value) == 0


def ax_window(a):
    """The app's main window as an Accessibility element (main, else focused, else first standard window)."""
    app = AX.AXUIElementCreateApplication(a.processIdentifier())
    win = _get(app, AX.kAXMainWindowAttribute) or _get(app, AX.kAXFocusedWindowAttribute)
    if win is None:
        wins = _get(app, AX.kAXWindowsAttribute) or []
        std = [w for w in wins if _get(w, AX.kAXSubroleAttribute) == AX.kAXStandardWindowSubrole]
        win = (std or list(wins) or [None])[0]
    return win


def frame_of(win):
    _, p = AX.AXValueGetValue(_get(win, AX.kAXPositionAttribute), AX.kAXValueCGPointType, None)
    _, z = AX.AXValueGetValue(_get(win, AX.kAXSizeAttribute), AX.kAXValueCGSizeType, None)
    return round(p.x), round(p.y), round(z.width), round(z.height)


def set_frame(win, x, y, w, h):
    size = AX.AXValueCreate(AX.kAXValueCGSizeType, Quartz.CGSize(w, h))
    _set(win, AX.kAXSizeAttribute, size)  # size, position, size: some apps clamp the size to the current position
    _set(win, AX.kAXPositionAttribute, AX.AXValueCreate(AX.kAXValueCGPointType, Quartz.CGPoint(x, y)))
    _set(win, AX.kAXSizeAttribute, size)


def prepare_window(a):
    """Bring the app forward and get a usable window: reopen one if needed, un-minimise, leave full screen."""
    if not AX.AXIsProcessTrusted():
        return None, "Accessibility permission is missing for silent chat (System Settings > Privacy & Security)."
    import control
    control.bring_front(a)  # (switches to its Space if it's full screen)
    win = None
    for attempt in range(30):
        win = ax_window(a)
        if win is not None:
            break
        if attempt == 3:  # no window: reopen one (like clicking its Dock icon)
            subprocess.run(["open", "-a", a.bundleURL().path() if a.bundleURL() else a.localizedName()],
                           capture_output=True)
        time.sleep(0.1)
    if win is None:
        return None, f"{a.localizedName()} has no window I can use."
    if _get(win, AX.kAXMinimizedAttribute):
        _set(win, AX.kAXMinimizedAttribute, False)
        time.sleep(0.4)
    if _get(win, "AXFullScreen"):
        _set(win, "AXFullScreen", False)
        for _ in range(30):  # wait for the exit-full-screen animation
            time.sleep(0.1)
            if not _get(win, "AXFullScreen"):
                break
        time.sleep(0.8)
        win = ax_window(a) or win
    return win, ""


def window_action(a, action, *frame):
    """place (with frame), fullscreen-on, fullscreen-off, minimize, restore. Returns (ok, message)."""
    win, err = prepare_window(a)
    if win is None:
        return False, err
    name = a.localizedName()
    if action == "minimize":
        return _set(win, AX.kAXMinimizedAttribute, True), f"{name} minimised."
    if action == "fullscreen-on":
        return _set(win, "AXFullScreen", True), f"{name} is full screen."
    if action in ("fullscreen-off", "restore"):
        return True, f"{name} {'left full screen' if action == 'fullscreen-off' else 'restored'}."
    x, y, w, h = frame
    for attempt in range(2):  # set it, then read it back: report what really happened
        set_frame(win, x, y, w, h)
        time.sleep(0.3)
        ax, ay, aw, ah = frame_of(win)
        if abs(ax - x) <= 12 and abs(ay - y) <= 12 and abs(aw - w) <= 12 and abs(ah - h) <= 12:
            return True, ""
    if (ax, ay) == (x, y) or (abs(ax - x) <= 12 and abs(ay - y) <= 12):
        return True, f"(it can't be smaller than {aw}x{ah} pt, so it's {aw} wide instead of {w})"
    return False, f"macOS didn't let me move {name}: it's still at {ax},{ay} ({aw}x{ah})."


def place(name, where, dry_run=False):
    a = find_app(name)
    if not a and not dry_run:  # not running yet: open it first (a dry run never changes anything)
        subprocess.run(["open", "-a", name], capture_output=True)
        a = wait_for(name)
    if not a:
        return f"Couldn't find {name}." if not dry_run else f"{name} isn't running (would open it first)."
    x, y, w, h = visible_area()
    gap = 8
    frames = {"left": (x, y, w // 2 - gap // 2, h), "right": (x + w // 2 + gap // 2, y, w // 2 - gap // 2, h),
              "full": (x, y, w, h), "top": (x, y, w, h // 2 - gap // 2),
              "bottom": (x, y + h // 2 + gap // 2, w, h // 2 - gap // 2),
              "center": (x + w // 6, y + h // 8, w * 2 // 3, h * 3 // 4)}
    if where not in frames:
        return f"Unknown place '{where}'. Use left, right, full, center, top or bottom."
    fx, fy, fw, fh = frames[where]
    if dry_run:
        return f"{a.localizedName()} -> {where} {fx},{fy} {fw}x{fh}"
    ok, msg = window_action(a, "place", fx, fy, fw, fh)
    return f"{a.localizedName()} placed {where}{' ' + msg if msg else ''}." if ok else msg


def cmd_place(args):
    if len(args) < 2:
        return out("usage: macctl place <App> left|right|full|center|top|bottom")
    out(place(" ".join(args[:-1]), args[-1].lower(), "--dry-run" in sys.argv))


def cmd_arrange(args):
    if len(args) != 2:
        return out("usage: macctl arrange <App1> <App2>   (quote names with spaces)")
    dry = "--dry-run" in sys.argv
    out(place(args[0], "left", dry))
    if dry:
        return out(place(args[1], "right", dry))
    first = find_app(args[0])
    win = ax_window(first) if first and AX.AXIsProcessTrusted() else None
    if win is None:
        return out(place(args[1], "right"))
    x, y, w, h = visible_area()
    lx, ly, lw, lh = frame_of(win)
    gap = 8
    rx, rw = lx + lw + gap, x + w - (lx + lw + gap)  # whatever is left to the right of the first window
    second = find_app(args[1])
    if not second:
        subprocess.run(["open", "-a", args[1]], capture_output=True)
        second = wait_for(args[1])
    if not second:
        return out(f"Couldn't find {args[1]}.")
    ok, msg = window_action(second, "place", rx, y, rw, h)
    out(f"{second.localizedName()} placed right{' ' + msg if msg else ''}." if ok else msg)


def _simple(args, action, done, usage):
    if not args:
        return out(usage)
    a = find_app(" ".join(args))
    if not a:
        return out(f"{' '.join(args)} isn't running.")
    ok, err = window_action(a, action)
    out(f"{a.localizedName()} {done}." if ok else f"Couldn't do that for {a.localizedName()}: {err}")


def cmd_fullscreen(args):
    if len(args) < 2 or args[-1].lower() not in ("on", "off"):
        return out("usage: macctl fullscreen <App> on|off")
    on = args[-1].lower() == "on"
    _simple(args[:-1], "fullscreen-on" if on else "fullscreen-off", "is full screen" if on else "left full screen",
            "usage: macctl fullscreen <App> on|off")


def cmd_minimize(args):
    _simple(args, "minimize", "minimised", "usage: macctl minimize <App>")


def cmd_restore(args):
    _simple(args, "restore", "restored", "usage: macctl restore <App>")


def cmd_windows(args):
    a = find_app(" ".join(args)) if args else None
    if not a:
        return out("usage: macctl windows <App>  (it must be running)")
    if not AX.AXIsProcessTrusted():
        return out("Accessibility permission is missing.")
    app = AX.AXUIElementCreateApplication(a.processIdentifier())
    wins = _get(app, AX.kAXWindowsAttribute) or []
    main = _get(app, AX.kAXMainWindowAttribute)
    out(f"{a.localizedName()}: {len(wins)} window(s)")
    for w in wins:
        x, y, ww, hh = frame_of(w)
        out(f"  {'*' if main is not None and w == main else ' '} '{_get(w, AX.kAXTitleAttribute) or ''}' at {x},{y} "
            f"{ww}x{hh} subrole={_get(w, AX.kAXSubroleAttribute)} minimised={bool(_get(w, AX.kAXMinimizedAttribute))} "
            f"fullscreen={bool(_get(w, 'AXFullScreen'))}")


def cmd_look(args):
    SCREENS.mkdir(exist_ok=True)
    path = SCREENS / f"look_{datetime.now():%H%M%S}.png"
    if args:
        a = find_app(" ".join(args))
        wins = windows(a.localizedName()) if a else []
        if not wins:
            return out(f"No visible window for {' '.join(args)}. Open it first.")
        win = max(wins, key=lambda w: w["w"] * w["h"])
        r = subprocess.run(["screencapture", "-x", "-o", "-l", str(win["id"]), str(path)], capture_output=True)
    else:
        r = subprocess.run(["screencapture", "-x", "-m", str(path)], capture_output=True)
    if r.returncode or not path.exists():
        return out(f"Couldn't take a screenshot: allow {HOST} under Privacy & Security > Screen Recording.")
    subprocess.run(["sips", "-Z", "1400", str(path)], capture_output=True)  # keep it small enough to read quickly
    out(f"Screenshot saved: {path.relative_to(Path.cwd())}  (Read this file to see it)")


def cmd_url(args):
    if not args or not args[0].startswith(("https://", "http://")):
        return out("usage: macctl url <https://...>")
    subprocess.run(["open", args[0]])
    out(f"Opened {args[0]}")


SPOTIFY_NOW = '''
tell application "Spotify"
    set pstate to player state as string
    if pstate is "stopped" then return "stopped"
    set t to current track
    return pstate & "|" & (name of t) & "|" & (artist of t) & "|" & (album of t) & "|" & ((player position) as integer) & ¬
        "|" & ((duration of t) div 1000) & "|" & sound volume & "|" & shuffling & "|" & repeating
end tell
'''
SPOTIFY_ARG = '''
on run argv
    set cmd to item 1 of argv
    tell application "Spotify"
        if cmd is "play-uri" then play track (item 2 of argv)
        if cmd is "volume" then set sound volume to ((item 2 of argv) as integer)
        if cmd is "shuffle" then set shuffling to ((item 2 of argv) is "on")
        if cmd is "repeat" then set repeating to ((item 2 of argv) is "on")
    end tell
end run
'''
SIMPLE = {"play": "play", "pause": "pause", "toggle": "playpause", "next": "next track", "previous": "previous track"}


def spotify_uri(link):
    """open.spotify.com/<type>/<id>?si=... or spotify:<type>:<id> -> spotify:<type>:<id> (None if it isn't one)."""
    import re
    m = re.search(r"open\.spotify\.com/(?:intl-[a-z-]+/)?(track|album|playlist|artist|episode|show)/([A-Za-z0-9]+)",
                  link) or re.fullmatch(r"spotify:(track|album|playlist|artist|episode|show):([A-Za-z0-9]+)", link)
    return f"spotify:{m.group(1)}:{m.group(2)}" if m else None


def osa(script, *args):
    r = subprocess.run(["osascript", "-e", script, *args], capture_output=True, text=True)
    if r.returncode and "not allowed" in r.stderr.lower():
        raise SystemExit(f"Not allowed to control Spotify: allow {HOST} under Privacy & Security > Automation.")
    return r.returncode == 0, r.stdout.strip(), r.stderr.strip()[:160]


def spotify_now():
    if not find_app("Spotify"):
        return "Spotify isn't running."
    ok, res, err = osa(SPOTIFY_NOW)
    if not ok:
        return f"Couldn't read Spotify: {err}"
    if res == "stopped":
        return "Spotify is stopped."
    state, name, artist, album, pos, dur, vol, shuf, rep_ = res.split("|")
    fmt = lambda t: f"{int(t) // 60}:{int(t) % 60:02d}"
    return (f"{state}: {name} by {artist} ({album}), {fmt(pos)} of {fmt(dur)}; volume {vol}; "
            f"shuffle {'on' if shuf == 'true' else 'off'}, repeat {'on' if rep_ == 'true' else 'off'}")


def cmd_spotify(args):
    dry = "--dry-run" in sys.argv
    if not args:
        return out("usage: macctl spotify now|play [<link>]|pause|toggle|next|previous|volume <n|up|down>|"
                   "shuffle on|off|repeat on|off|search <words>")
    cmd, rest = args[0].lower(), args[1:]
    if cmd == "now":
        return out(spotify_now())
    if cmd == "play" and rest:
        uri = spotify_uri(rest[0])
        if not uri:
            return out("That isn't a Spotify link (open.spotify.com/... or spotify:track:...). "
                       "Use 'macctl spotify search <words>' to open Spotify's search instead.")
        if dry:
            return out(f"(dry run) would play {uri}")
        ok, _, err = osa(SPOTIFY_ARG, "play-uri", uri)
        time.sleep(1.0)
        return out(spotify_now() if ok else f"Couldn't play it: {err}")
    if cmd in SIMPLE:
        if dry:
            return out(f"(dry run) would {SIMPLE[cmd]}")
        ok, _, err = osa(f'tell application "Spotify" to {SIMPLE[cmd]}')
        time.sleep(0.6)
        return out(spotify_now() if ok else f"Couldn't: {err}")
    if cmd == "volume" and rest:
        v = rest[0].lower()
        if v in ("up", "down"):
            ok, cur, _ = osa('tell application "Spotify" to return sound volume')
            v = str(max(0, min(100, int(cur or 50) + (10 if v == "up" else -10))))
        if not v.isdigit():
            return out("usage: macctl spotify volume <0-100|up|down>")
        v = str(max(0, min(100, int(v))))
        if dry:
            return out(f"(dry run) would set volume {v}")
        ok, _, err = osa(SPOTIFY_ARG, "volume", v)
        return out(f"Spotify volume {v}." if ok else f"Couldn't: {err}")
    if cmd in ("shuffle", "repeat") and rest and rest[0].lower() in ("on", "off"):
        if dry:
            return out(f"(dry run) would set {cmd} {rest[0]}")
        ok, _, err = osa(SPOTIFY_ARG, cmd, rest[0].lower())
        return out(f"{cmd.capitalize()} {rest[0].lower()}." if ok else f"Couldn't: {err}")
    if cmd == "search" and rest:
        from urllib.parse import quote
        query = " ".join(rest)
        if not dry:
            subprocess.run(["open", f"spotify:search:{quote(query)}"])
        return out(f"Opened Spotify's search for '{query}' (it can't auto-play a search; pick a result there).")
    out("usage: macctl spotify now|play [<link>]|pause|toggle|next|previous|volume <n|up|down>|"
        "shuffle on|off|repeat on|off|search <words>")


COMMANDS = {"apps": cmd_apps, "open": cmd_open, "focus": cmd_focus, "place": cmd_place, "arrange": cmd_arrange,
            "fullscreen": cmd_fullscreen, "windows": cmd_windows, "minimize": cmd_minimize, "restore": cmd_restore,
            "look": cmd_look, "url": cmd_url, "spotify": cmd_spotify}


def dispatch(argv):
    """Run one macctl command. True if it worked."""
    import control
    name, args = argv[0], argv[1:]
    try:
        if name in control.COMMANDS:
            out(control.COMMANDS[name](args))
            return True
        if name in COMMANDS:
            if name in ("open", "focus", "url"):
                control.stop_check()
            COMMANDS[name](args)
            return True
    except control.Fail as e:
        out(f"FAILED: {e}")
        return False
    out(f"Unknown command {name!r}. Run ./macctl with no arguments for the list.")
    return False


def cmd_do(args):
    """Several steps in one call (each a quoted command line); stops at the first one that fails."""
    import shlex
    for step in args:
        parts = shlex.split(step)
        if not parts:
            continue
        out(f"> {step}")
        if parts[0] == "do" or not dispatch(parts):
            out("(stopped there; later steps not run)")
            return False
    return True


def main():
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    if not args:
        import control
        return out(__doc__ + "\n" + control.__doc__)
    ok = cmd_do(args[1:]) if args[0] == "do" else dispatch(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
