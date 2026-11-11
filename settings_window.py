"""Two small native windows: Settings (everything you can switch, in one place) and first-run Setup (the
permissions silent chat needs, each with a button to the right System Settings page, and a live check mark).

Both talk to the app the way the menu bar does: on_command("/screen off") etc., and state() for what's on now.
"""
import objc
from AppKit import (NSApp, NSBackingStoreBuffered, NSButton, NSColor, NSFont, NSMakeRect, NSPopUpButton, NSSwitch,
                    NSTextField, NSTimer, NSView, NSWindow, NSWindowStyleMaskClosable, NSWindowStyleMaskTitled,
                    NSWorkspace)
from Foundation import NSObject, NSURL

W = 520


def _label(text, x, y, w, size=13, bold=False, color=None, h=18):
    t = NSTextField.labelWithString_(text)
    t.setFrame_(NSMakeRect(x, y, w, h))
    t.setFont_(NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size))
    if color is not None:
        t.setTextColor_(color)
    t.setLineBreakMode_(4)  # (truncate tail)
    return t


class _Target(NSObject):
    def initWithOwner_(self, owner):
        self = objc.super(_Target, self).init()
        if self is not None:
            self.owner = owner
        return self

    def switched_(self, sender):
        self.owner.switched(sender)

    def pressed_(self, sender):
        self.owner.pressed(sender)

    def picked_(self, sender):
        self.owner.picked(sender)

    def tick_(self, timer):
        self.owner.refresh()


def _window(title, h):
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(0, 0, W, h), NSWindowStyleMaskTitled | NSWindowStyleMaskClosable, NSBackingStoreBuffered, False)
    win.setTitle_(title)
    win.setReleasedWhenClosed_(False)
    win.center()
    return win


# (state key, label, what it does, command to turn it on, command to turn it off; None = toggle command)
SWITCHES = [
    ("handsfree", "Hands-free", "Just mouth at the screen; a local model filters out chatter.",
     "/handsfree on", "/handsfree off"),
    ("wake", "“Hey Claude”", "Always listen for “Hey Claude” (speech is checked on this Mac).", None, None),
    ("voice", "Speak replies", "Read replies aloud when you mouth or type (with headphones in).", None, None),
    ("ambient", "Live in the notch", "Show only while in use, in the Dynamic Island. Off: a floating panel.",
     "/ambient on", "/ambient off"),
    ("screen", "Screen awareness", "Send a short, privacy-checked summary of the window in front.", "/screen on",
     "/screen off"),
    ("gaze", "Gaze awareness", "Tell Claude roughly what you were looking at as you spoke.", "/gaze on", "/gaze off"),
    ("lens", "Glass lens", "A subtle lens that follows your gaze.", "/lens on", "/lens off"),
    ("warp", "Lens bends the screen", "Needs Screen Recording; nothing is kept or sent.", "/lens warp on",
     "/lens warp off"),
    ("notify", "Notifications", "Show new notifications and let Claude reply (it asks first).", "/notify on",
     "/notify off"),
    ("nudges", "Suggest replies", "When a message notification arrives, offer to reply (off by default).",
     "/nudges on", "/nudges off"),
    ("improve", "Build its own features", "Learn shortcuts and tips from how you use it, once a day.",
     "/improve on", "/improve off"),
    ("face", "Face gestures", "Long blink to listen, smile to delete, eyebrows for Enter.", None, None),
    ("clips", "Save mouth clips for training", "Keep clips of your mouth on this Mac to improve the lip-reader.",
     "/privacy clips on", "/privacy clips off"),
    ("remember", "Remember across restarts", "Keep your last few exchanges on this Mac so a restart keeps context.",
     "/privacy remember on", "/privacy remember off"),
    ("clicks", "Learn gaze as I work", "Eye tracking calibrates itself from what you click, type and look at.",
     "/privacy clicks on", "/privacy clicks off"),
]
MODELS = [("auto", "Auto: Haiku, Opus when needed"), ("haiku", "Haiku (cheapest)"), ("opus", "Opus (smartest)")]
BUTTONS = [("Practice (2 min)", "/practice"), ("Recalibrate Gaze…", "/gaze calibrate"), ("Improve Now", "/improve"),
           ("Show Where I'm Looking", "/gaze show 20"), ("Show Stats", "/stats window"), ("Setup…", "/setup")]


class SettingsWindow:
    def __init__(self, on_command, state):
        self.on_command, self.state = on_command, state
        self.target = _Target.alloc().initWithOwner_(self)
        h = 70 + 46 * len(SWITCHES) + 144
        self.win = _window("Silent Chat Settings", h)
        v = self.win.contentView()
        y = h - 40
        v.addSubview_(_label("Silent Chat", 24, y, 300, 17, bold=True, h=22))
        self.switches = {}
        y -= 16
        for i, (key, label, desc, _, _) in enumerate(SWITCHES):
            y -= 46
            v.addSubview_(_label(label, 24, y + 20, 360, 13, bold=True))
            v.addSubview_(_label(desc, 24, y + 2, 400, 11, color=NSColor.secondaryLabelColor()))
            sw = NSSwitch.alloc().initWithFrame_(NSMakeRect(W - 70, y + 6, 44, 24))
            sw.setTarget_(self.target)
            sw.setAction_("switched:")
            sw.setTag_(i)
            v.addSubview_(sw)
            self.switches[key] = sw
        y -= 44
        v.addSubview_(_label("Claude model", 24, y + 4, 140, 13, bold=True))
        self.model = NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(170, y, W - 194, 26), False)
        for _, name in MODELS:
            self.model.addItemWithTitle_(name)
        self.model.setTarget_(self.target)
        self.model.setAction_("picked:")
        v.addSubview_(self.model)
        y -= 46
        x = 24
        for i, (title, cmd) in enumerate(BUTTONS):
            b = NSButton.buttonWithTitle_target_action_(title, self.target, "pressed:")
            b.sizeToFit()
            if x + b.frame().size.width > W - 20:  # (wrap to a second row)
                x, y = 24, y - 34
            b.setFrameOrigin_((x, y))
            b.setTag_(i)
            v.addSubview_(b)
            x += b.frame().size.width + 8
        self.timer = None

    def show(self):
        self.refresh()
        self.win.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)
        if self.timer is None:
            self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1.0, self.target, "tick:", None, True)

    def refresh(self):
        if not self.win.isVisible():
            return
        s = self.state()
        for key, sw in self.switches.items():
            on = s.get(key)
            on = on == "on" if key == "handsfree" else bool(on)
            sw.setState_(1 if on else 0)
        names = [m for m, _ in MODELS]
        self.model.selectItemAtIndex_(names.index(s.get("model", "auto")) if s.get("model", "auto") in names else 0)

    def switched(self, sw):
        row = SWITCHES[sw.tag()]
        key = row[0]
        on = sw.state() == 1
        cmd = (row[3] if on else row[4]) or f"/{key}"
        self.on_command(cmd)

    def picked(self, popup):
        self.on_command("/model " + MODELS[popup.indexOfSelectedItem()][0])

    def pressed(self, b):
        self.on_command(BUTTONS[b.tag()][1])


PANES = "x-apple.systempreferences:com.apple.preference.security?"


class SetupWindow:
    """First run: what silent chat needs, each with a live check and a button to fix it."""

    def __init__(self, checks, on_command, on_done):
        """checks: [(name, why, check() -> bool, privacy pane anchor or URL)]"""
        self.checks, self.on_command, self.on_done = checks, on_command, on_done
        self.target = _Target.alloc().initWithOwner_(self)
        h = 166 + 56 * len(checks) + 24
        self.win = _window("Set Up Silent Chat", h)
        v = self.win.contentView()
        y = h - 46
        v.addSubview_(_label("Welcome to Silent Chat", 24, y, 400, 18, bold=True, h=24))
        y -= 26
        sub = NSTextField.wrappingLabelWithString_("Talk to Claude by mouthing words. It needs a few macOS "
                                                   "permissions; nothing they give access to leaves this Mac "
                                                   "except what you send Claude.")
        sub.setFrame_(NSMakeRect(24, y - 16, W - 48, 34))
        sub.setFont_(NSFont.systemFontOfSize_(12))
        sub.setTextColor_(NSColor.secondaryLabelColor())
        v.addSubview_(sub)
        y -= 16
        self.marks, self.cmds = [], []
        for name, why, check, pane in checks:
            y -= 56
            mark = _label("…", 24, y + 14, 24, 16, h=22)
            v.addSubview_(mark)
            v.addSubview_(_label(name, 54, y + 22, 300, 13, bold=True))
            v.addSubview_(_label(why, 54, y + 4, 330, 11, color=NSColor.secondaryLabelColor()))
            b = NSButton.buttonWithTitle_target_action_("Open Settings", self.target, "pressed:")
            b.sizeToFit()
            b.setFrameOrigin_((W - b.frame().size.width - 24, y + 10))
            b.setTag_(len(self.cmds))
            self.cmds.append(pane)
            v.addSubview_(b)
            self.marks.append((mark, check))
        y -= 60
        for title, cmd, x in (("Calibrate Gaze (optional)…", "/gaze calibrate", 24), ("Done", "done", None)):
            b = NSButton.buttonWithTitle_target_action_(title, self.target, "pressed:")
            b.sizeToFit()
            b.setFrameOrigin_((x if x is not None else W - b.frame().size.width - 24, y))
            b.setTag_(len(self.cmds))
            self.cmds.append(cmd)
            if cmd == "done":
                b.setKeyEquivalent_("\r")
            v.addSubview_(b)
        self.timer = None

    def show(self):
        self.refresh()
        self.win.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)
        if self.timer is None:
            self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1.0, self.target, "tick:", None, True)

    def refresh(self):
        if not self.win.isVisible():
            return
        for mark, check in self.marks:
            try:
                ok = bool(check())
            except Exception:
                ok = False
            mark.setStringValue_("✅" if ok else "⚪️")

    def pressed(self, b):
        cmd = self.cmds[b.tag()]
        if cmd == "done":
            self.win.orderOut_(None)
            self.on_done()
        elif cmd.startswith("/"):
            self.on_command(cmd)
        else:
            url = cmd if "://" in cmd or cmd.startswith("x-apple") else PANES + cmd
            NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(url))

    def switched(self, sw):
        pass

    def picked(self, p):
        pass
