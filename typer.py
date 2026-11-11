"""Type text into whatever app is in front, as if from the keyboard (for silent dictation).

Needs Accessibility permission for the app running this (Terminal): System Settings > Privacy & Security >
Accessibility. Without it macOS silently drops the synthetic key presses, so check `trusted()` first.
"""
import time

import Quartz
from AppKit import NSWorkspace
from ApplicationServices import AXIsProcessTrusted, AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt

RETURN, DELETE = 36, 51  # macOS virtual key codes
TERMINALS = {"com.apple.Terminal", "com.googlecode.iterm2", "dev.warp.Warp-Stable", "com.mitchellh.ghostty"}


def trusted(prompt=False):
    """True if we may send key presses to other apps. With prompt=True, macOS shows its permission dialog."""
    if prompt:
        return bool(AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True}))
    return bool(AXIsProcessTrusted())


def frontmost():
    """(bundle id, pid, name) of the app that would receive typing."""
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return app.bundleIdentifier() or "", app.processIdentifier(), app.localizedName() or ""


LAST_SYNTHETIC = [0.0]  # when we last sent a key press ourselves (time.monotonic)


def _key(code, down, flags=0, text=None):
    ev = Quartz.CGEventCreateKeyboardEvent(None, code, down)
    if text:
        Quartz.CGEventKeyboardSetUnicodeString(ev, len(text.encode("utf-16-le")) // 2, text)
    Quartz.CGEventSetFlags(ev, flags)  # no stray modifiers (e.g. the Option you might still be holding)
    # Session level, not HID level: HID-level presses show up as physically held keys, so our own Option+Delete
    # looked like you holding Option (the talk key) and started listening.
    Quartz.CGEventPost(Quartz.kCGSessionEventTap, ev)
    LAST_SYNTHETIC[0] = time.monotonic()


def type_text(text):
    """Type `text` into the frontmost app. Line breaks are sent as Return."""
    for i, line in enumerate(text.split("\n")):
        if i:
            _key(RETURN, True)
            _key(RETURN, False)
        for j in range(0, len(line), 16):  # macOS takes at most ~20 characters per synthetic key event
            chunk = line[j:j + 16]
            _key(0, True, text=chunk)
            _key(0, False, text=chunk)
            time.sleep(0.004)


def delete_word():
    """Option+Delete: removes the word before the cursor, in any standard macOS text field."""
    _key(DELETE, True, Quartz.kCGEventFlagMaskAlternate)
    _key(DELETE, False, Quartz.kCGEventFlagMaskAlternate)


# ----------------------------------------------------------------------------- global show/hide key

TOGGLE_KEYS = {"\\": 42, "`": 50, "§": 10}  # character -> macOS virtual key code (ANSI layout)


def request_listen_access():
    """Input Monitoring permission (needed to see key presses in other apps). Shows the macOS prompt once."""
    if Quartz.CGPreflightListenEventAccess():
        return True
    Quartz.CGRequestListenEventAccess()
    return bool(Quartz.CGPreflightListenEventAccess())


class ToggleKey:
    """A global key that shows/hides silent chat, whichever app is in front.

    Tap it to toggle; double-tap it to type the character itself (the key is swallowed, so this is how you still
    type it). With Shift or Cmd/Ctrl/Option it passes through untouched (Shift+\\ is still |).
    Runs an event tap on its own thread, so it never delays typing in other apps. Needs Accessibility + Input
    Monitoring for Terminal; `ok` is False if macOS refused the tap.
    """

    def __init__(self, char="\\", double_tap=0.35):
        import threading
        self.char, self.keycode, self.window = char, TOGGLE_KEYS[char], double_tap
        self.presses, self.pending, self.ok, self.tap = [], None, False, None
        self.last_key = 0.0  # when any other key was last pressed (tells Option dictation from Option shortcuts)
        import gaze_click
        self.rcmd = gaze_click.TapDetector()  # Right Command taps: click what you're looking at
        self._ready = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()
        self._ready.wait(3)

    def _run(self):
        mask = (Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown) | Quartz.CGEventMaskBit(Quartz.kCGEventKeyUp)
                | Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged)
                | Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDown)
                | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDown))
        self.tap = Quartz.CGEventTapCreate(Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
                                           Quartz.kCGEventTapOptionDefault, mask, self._callback, None)
        if self.tap is None:
            self._ready.set()
            return
        source = Quartz.CFMachPortCreateRunLoopSource(None, self.tap, 0)
        Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(self.tap, True)
        self.ok = True
        self._ready.set()
        Quartz.CFRunLoopRun()

    def _callback(self, proxy, etype, event, refcon):
        if etype in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
            Quartz.CGEventTapEnable(self.tap, True)
            return event
        if etype == Quartz.kCGEventFlagsChanged:
            self.rcmd.flags(Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode),
                            bool(Quartz.CGEventGetFlags(event) & Quartz.kCGEventFlagMaskCommand), time.monotonic())
            return event
        if etype in (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventRightMouseDown):
            self.rcmd.other()
            return event
        if etype == Quartz.kCGEventKeyDown:
            self.rcmd.other()
        if Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode) != self.keycode:
            if etype == Quartz.kCGEventKeyDown:
                self.last_key = time.monotonic()
            return event
        mods = (Quartz.kCGEventFlagMaskShift | Quartz.kCGEventFlagMaskCommand | Quartz.kCGEventFlagMaskControl
                | Quartz.kCGEventFlagMaskAlternate)
        if Quartz.CGEventGetFlags(event) & mods:
            return event
        if etype == Quartz.kCGEventKeyDown and not Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventAutorepeat):
            self.presses.append(time.monotonic())
        return None  # swallow it

    def poll(self, now=None):
        """Call every loop. Returns actions: "toggle" (single tap) or "type" (double tap)."""
        now = time.monotonic() if now is None else now
        actions = []
        while self.presses:
            t = self.presses.pop(0)
            if self.pending is not None and t - self.pending <= self.window:
                actions.append("type")
                self.pending = None
            else:
                if self.pending is not None:
                    actions.append("toggle")
                self.pending = t
        if self.pending is not None and now - self.pending > self.window:
            actions.append("toggle")
            self.pending = None
        return actions


def press_return():
    """Enter: sends the message in chat apps (Messages, Discord, Slack ...)."""
    _key(RETURN, True)
    _key(RETURN, False)
