"""Silent chat's menu bar icon: show/hide, the modes, Hey Claude, Open at Login, a console, Restart and Quit.

Menu choices become the same commands the console accepts ("/show", "/type", "/quit", ...), so they go through
silent chat's normal command handling. The icon is a mouth: outlined while hidden, filled while the overlay shows;
a waveform while you hold Option to dictate, and dots while that's being turned into text.
"""
import os
import subprocess
import threading
from pathlib import Path

import objc
from AppKit import NSImage, NSMenu, NSMenuItem, NSObject, NSStatusBar, NSVariableStatusItemLength

ROOT = Path(__file__).resolve().parent
APP = Path.home() / "Applications" / "Silent Chat.app"
ITEMS = [("toggle", "Show Silent Chat"), None,
         ("type", "TYPE: type what I mouth into apps"), ("voice", "VOICE: speak replies"),
         ("face", "FACE: blink / smile / eyebrow gestures"), ("mic", "MIC: speak out loud for corrections"),
         ("wake", 'HEY CLAUDE: always listen for "Hey Claude"'),
         ("handsfree", "HANDS-FREE: start when you mouth words at the screen"),
         ("screen", "SCREEN: Claude sees what's in front with each message"),
         ("notify", "NOTIFICATIONS: show them here, read out when VOICE is on"),
         ("lens", "GLASS CIRCLE: show where you're looking"),
         ("ambient", "AUTO-HIDE: show only while I'm using it"),
         ("gazecal", "Calibrate Gaze…"), ("gazeshow", "Show Where I'm Looking"), None,
         ("settings", "Settings…"),
         ("model:auto", "Model: Auto (Haiku, Opus when needed)"), ("model:haiku", "Model: Haiku (cheapest)"),
         ("model:opus", "Model: Opus (smartest)"), None,
         ("login", "Open at Login"), ("console", "Open Console (type, /fix, log)"), None,
         ("restart", "Restart Silent Chat"), ("quit", "Quit Silent Chat")]


def launcher():
    """The app's launcher binary (it manages Open at Login), or None if silent chat isn't installed as an app."""
    path = os.environ.get("SILENT_CHAT_LAUNCHER") or str(APP / "Contents" / "MacOS" / "SilentChat")
    return path if os.path.exists(path) else None


class _MenuTarget(NSObject):
    def initWithOwner_(self, owner):
        self = objc.super(_MenuTarget, self).init()
        if self is not None:
            self.owner = owner
        return self

    @objc.IBAction
    def clicked_(self, sender):
        self.owner._clicked(sender.representedObject())

    def menuNeedsUpdate_(self, menu):
        self.owner._refresh()


class StatusMenu:
    def __init__(self, on_command, state, key_label="\\"):
        """on_command("/show") etc. state() -> {"open", "type", "voice", "face", "mic"} (current settings)."""
        self.on_command, self.state, self.key_label = on_command, state, key_label
        self.target = _MenuTarget.alloc().initWithOwner_(self)
        self.item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        self.menu = NSMenu.alloc().init()
        self.menu.setAutoenablesItems_(False)
        self.menu.setDelegate_(self.target)
        self.items = {}
        for entry in ITEMS:
            if entry is None:
                self.menu.addItem_(NSMenuItem.separatorItem())
                continue
            name, title = entry
            mi = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, "clicked:", "")
            mi.setTarget_(self.target)
            mi.setRepresentedObject_(name)
            self.menu.addItem_(mi)
            self.items[name] = mi
        self.item.setMenu_(self.menu)
        self.launcher, self.login, self.shown, self.activity = launcher(), None, True, None
        if self.launcher:
            threading.Thread(target=self._login, daemon=True).start()
        else:
            self.items["login"].setHidden_(True)
        self.set_shown(True)

    def set_shown(self, shown):
        self.shown = shown
        self._draw()

    def set_activity(self, activity, tip=""):
        """"listening" (Option held), "thinking" (turning it into text) or None. tip: shown on hover."""
        if (activity, tip) != (self.activity, getattr(self, "tip", "")):
            self.activity, self.tip = activity, tip
            self._draw()

    def _draw(self):
        symbol = {"listening": "waveform", "thinking": "ellipsis"}.get(self.activity,
                                                                         "mouth.fill" if self.shown else "mouth")
        img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, "Silent Chat")
        button = self.item.button()
        if img is None:
            button.setTitle_("Silent")
            return
        img.setTemplate_(True)  # drawn in the menu bar's own colour (light/dark)
        button.setImage_(img)
        button.setToolTip_(getattr(self, "tip", "") or
                           f"Silent Chat ({'shown' if self.shown else 'hidden'}): press {self.key_label} to show/hide")

    def _refresh(self):
        s = self.state()
        self.items["toggle"].setTitle_(f"{'Hide' if s['open'] else 'Show'} Silent Chat      {self.key_label}")
        for name in ("type", "voice", "face", "mic", "wake", "screen", "notify", "lens", "ambient"):
            self.items[name].setState_(1 if s.get(name) else 0)
        hf = s.get("handsfree", "off")
        self.items["handsfree"].setState_(1 if hf == "on" else 0)
        self.items["handsfree"].setTitle_("HANDS-FREE: start when you mouth words at the screen" +
                                          ("  (measuring only)" if hf == "shadow" else ""))
        for name in ("auto", "haiku", "opus"):
            self.items["model:" + name].setState_(1 if s.get("model", "auto") == name else 0)
        login = self.items["login"]
        login.setState_(1 if self.login in ("on", "needs-approval") else 0)
        login.setTitle_("Open at Login (approve it in System Settings > Login Items)"
                        if self.login == "needs-approval" else "Open at Login")

    def _clicked(self, name):
        if name == "toggle":
            self.on_command("/hide" if self.state()["open"] else "/show")
        elif name.startswith("model:"):
            self.model = name.split(":")[1]
            self.on_command("/model " + self.model)
        elif name == "settings":
            self.on_command("/settings")
        elif name == "gazecal":
            self.on_command("/gaze calibrate")
        elif name == "gazeshow":
            self.on_command("/gaze show 20")
        elif name == "handsfree":  # (/handsfree alone prints its stats)
            self.on_command("/handsfree " + ("off" if self.state().get("handsfree") == "on" else "on"))
        elif name in ("type", "voice", "face", "mic", "wake", "screen", "notify", "lens", "ambient", "restart", "quit"):
            self.on_command("/" + name)
        elif name == "login":
            change = "off" if self.login in ("on", "needs-approval") else "on"
            threading.Thread(target=self._login, args=(change,), daemon=True).start()
        elif name == "console":
            subprocess.Popen(["open", "-a", "Terminal", str(ROOT / "silent_chat.command")])

    def _login(self, change="status"):
        try:
            r = subprocess.run([self.launcher, "--login", change], capture_output=True, text=True, timeout=15)
            self.login = (r.stdout.strip().splitlines() or [None])[-1]
            if r.returncode or change != "status":
                print(f"[login] {change}: {self.login} {r.stderr.strip()}", flush=True)
        except Exception as e:
            print(f"[login] {change} failed: {e}", flush=True)
