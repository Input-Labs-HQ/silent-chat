"""New notifications as they arrive, read from the Notification Center banners (Accessibility, no extra permission).

macOS has no public way for an app to read other apps' notifications, but the banners are ordinary on-screen UI,
so silent chat reads them the way VoiceOver does. Polls a few times a second (one cheap Accessibility call while
nothing is showing). Codes and passwords are never read out or passed on.
"""
import re
import threading
import time

from datetime import datetime
from pathlib import Path

import ApplicationServices as AX
from AppKit import NSRunningApplication

import control

DEBUG_LOG = Path(__file__).resolve().parent / "data" / "silent_chat" / "notify_debug.log"

NC_BUNDLE = "com.apple.notificationcenterui"
PANEL_WORDS = {"clear all", "notification center", "show less", "notifications", "widgets", "edit widgets"}
SECRET = re.compile(r"\b(code|passcode|verification|verify|otp|2fa|one[- ]time|password|pin|login)\b", re.I)
DIGITS = re.compile(r"\b\d{4,8}\b|\b[A-Z0-9]{3}-[A-Z0-9]{3}\b")


def _nc():
    apps = NSRunningApplication.runningApplicationsWithBundleIdentifier_(NC_BUNDLE)
    return apps[0].processIdentifier() if apps else None


def _texts(el, depth=0, out=None):
    out = [] if out is None else out
    n = control._attrs(el)
    if not n or depth > 6:
        return out
    if n.get("AXRole") == "AXStaticText":
        v = control._text(n.get("AXValue") if isinstance(n.get("AXValue"), str) else n.get("AXTitle"), 240)
        if v and v not in out:
            out.append(v)
    for c in n.get("AXChildren") or []:
        _texts(c, depth + 1, out)
    return out


def banners(pid=None, debug=None):
    """The notifications on screen now: [{'app', 'title', 'body', 'private', 'key'}]. Each is an element whose
    subrole names it a notification (AXNotificationCenterBanner / Alert ...); its texts are title, (subtitle,) body.
    debug: a list that gets one line per element seen (structure, for tuning on a new macOS)."""
    pid = pid or _nc()
    if not pid:
        return []
    app = AX.AXUIElementCreateApplication(pid)
    wins = control._get(app, "AXWindows") or []
    found = []

    panel = [False]

    def visit(el, depth):
        n = control._attrs(el)
        if not n or depth > 14 or len(found) >= 30:
            return
        sub, desc = n.get("AXSubrole") or "", control._text(n.get("AXDescription"), 240)
        name = (control._text(n.get("AXTitle")) or desc).lower()
        if n.get("AXRole") in ("AXButton", "AXHeading", "AXStaticText") and name in PANEL_WORDS:
            panel[0] = True  # the Notification Center panel (opened from the clock), not a new banner
        if debug is not None:
            debug.append("  " * depth + f"{n.get('AXRole')} {sub} {desc[:80]!r}")
        if "Notification" in sub and "Stack" not in sub and "Window" not in sub:
            texts = _texts(el)
            if texts or desc:
                found.append((sub, desc, texts))
                return
        for c in n.get("AXChildren") or []:
            visit(c, depth + 1)

    for w in wins:
        visit(w, 0)
    out = []
    for sub, desc, texts in found:
        parts = [p.strip() for p in desc.split(", ")] if desc else []  # "App, Title, Body"
        app_name = parts[0] if parts and parts[0] not in texts and len(parts) > len(texts) else ""
        if texts:
            title, body = texts[0], " — ".join(texts[1:])
        else:
            rest = parts[1:] if app_name else parts
            title, body = (rest[0] if rest else ""), ", ".join(rest[1:])
        whole = f"{app_name} {title} {body}"
        out.append({"app": app_name, "title": title, "body": body, "kind": sub, "panel": panel[0] or len(found) > 4,
                    "private": bool(SECRET.search(whole) and DIGITS.search(whole)),
                    "key": (app_name, title, body)})
    return out


def describe(b, for_claude=False):
    """One line for the overlay, speech, or Claude."""
    who = " / ".join(x for x in (b["app"], b["title"]) if x) or "Notification"
    if b["private"]:
        return f"{who}: (a code or password, not read out)"
    return f"{who}: {b['body']}" if b["body"] else who


class Watcher:
    """Calls on_new(banner) from a background thread for each notification that appears. Not for ones already
    showing at start, and not when the whole Notification Center panel is opened (those aren't new)."""

    def __init__(self, on_new, every=0.6):
        self.on_new, self.every, self.enabled = on_new, every, True
        self.seen, self.pid, self.checked = {}, None, 0.0
        self.errors, self.dumped = 0, 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        first = True
        while True:
            time.sleep(self.every)
            if not self.enabled:
                first = True  # (don't announce what piled up while it was off)
                continue
            now = time.monotonic()
            if self.pid is None or now - self.checked > 30:
                self.pid, self.checked = _nc(), now
            dbg = []
            try:
                items = banners(self.pid, debug=dbg)
            except Exception:
                self.errors += 1
                items = []
            if dbg and not items and now - self.dumped > 60:  # something showed that isn't understood: keep its
                self.dumped = now                               # layout, to adjust banners() for this macOS
                try:
                    with open(DEBUG_LOG, "a") as f:
                        f.write(f"--- {datetime.now():%Y-%m-%d %H:%M:%S}\n" + "\n".join(dbg[:80]) + "\n")
                except OSError:
                    pass
            fresh = [b for b in items if b["key"] not in self.seen]
            for b in items:
                self.seen[b["key"]] = now
            for k in [k for k, t in self.seen.items() if now - t > 6 * 3600]:
                del self.seen[k]
            if first:  # start-up: what's showing already isn't new
                first = False
                continue
            for b in fresh:
                if b["panel"]:
                    continue
                try:
                    self.on_new(b)
                except Exception:
                    pass
