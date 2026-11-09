// Silent Chat.app's launcher. It starts silent chat (Python) as a child process and stays alive as its parent, so
// macOS asks for permissions (camera, microphone, Accessibility, Input Monitoring, ...) as "Silent Chat", not
// Terminal. It has no window or Dock icon: the menu bar icon and the overlay belong to the Python process.
//
//   - Exit code 0 (Quit in the menu) quits the app; 75 (Restart in the menu) restarts silent chat right away.
//   - If silent chat crashes it is restarted (hidden, camera off), at most 3 times in 10 minutes.
//   - Opening the app while it runs (Finder, Spotlight, `open -a`) shows the overlay (SIGUSR1).
//   - Launched at login it starts hidden, so the camera stays off until you press the toggle key.
//   - `SilentChat --login on|off|status` adds/removes it from Login Items (used by the menu's "Open at Login").
//
// Built by make_app.py, which fills in @ROOT@ (the project folder). Rebuilding changes the app's signature, and
// macOS then asks for its permissions again, so make_app.py only rebuilds when this file changes.
import AppKit
import ServiceManagement

let root = "@ROOT@"
let data = root + "/data/silent_chat"
let pidFile = data + "/silent_chat.pid"
let logPath = data + "/session.log"
let fifo = data + "/commands.fifo"
let agentPlist = NSHomeDirectory() + "/Library/LaunchAgents/local.silentchat.plist"

// ---------------------------------------------------------------- Login Items

func loginItem(_ change: String) -> Int32 {
    let service = SMAppService.mainApp
    var problem = ""
    do {
        if change == "on" && service.status != .enabled { try service.register() }
        if change == "off" && service.status != .notRegistered { try service.unregister() }
    } catch let e {
        problem = e.localizedDescription
    }
    // Fallback if Login Items refuses this (ad-hoc signed) app: a LaunchAgent that opens it at login.
    if change == "on" && !problem.isEmpty {
        let app = Bundle.main.bundlePath
        let plist = """
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
            <plist version="1.0"><dict>
              <key>Label</key><string>local.silentchat</string>
              <key>ProgramArguments</key>
              <array><string>/usr/bin/open</string><string>-a</string><string>\(app)</string>
                     <string>--args</string><string>--login-launch</string></array>
              <key>RunAtLoad</key><true/>
            </dict></plist>
            """
        try? FileManager.default.createDirectory(atPath: NSHomeDirectory() + "/Library/LaunchAgents",
                                                 withIntermediateDirectories: true)
        if (try? plist.write(toFile: agentPlist, atomically: true, encoding: .utf8)) != nil { problem = "" }
    }
    if change == "off" { try? FileManager.default.removeItem(atPath: agentPlist) }
    if !problem.isEmpty { FileHandle.standardError.write(("error: " + problem + "\n").data(using: .utf8)!) }
    switch service.status {
    case .enabled: print("on")
    case .requiresApproval: print("needs-approval")
    default: print(FileManager.default.fileExists(atPath: agentPlist) ? "on" : "off")
    }
    return problem.isEmpty ? 0 : 1
}

// ---------------------------------------------------------------- the app

func shell(_ path: String, _ args: [String]) -> String {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: path)
    p.arguments = args
    let out = Pipe()
    p.standardOutput = out
    p.standardError = FileHandle.nullDevice
    guard (try? p.run()) != nil else { return "" }
    p.waitUntilExit()
    return String(data: out.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
}

/// A silent chat started some other way (e.g. the old Terminal launcher), if one is running.
func otherInstance() -> pid_t? {
    guard let text = try? String(contentsOfFile: pidFile, encoding: .utf8),
          let pid = pid_t(text.trimmingCharacters(in: .whitespacesAndNewlines)), kill(pid, 0) == 0,
          shell("/bin/ps", ["-p", String(pid), "-o", "command="]).contains("silent_chat.py") else { return nil }
    return pid
}

class Launcher: NSObject, NSApplicationDelegate {
    var child: Process?
    var hidden = CommandLine.arguments.contains("--login-launch")
    var stopping = false
    var crashes: [Date] = []

    func launchedAtLogin() -> Bool {
        // the "open application" Apple Event says when Login Items opened us ('oapp' with prdt = 'lgit')
        guard let ev = NSAppleEventManager.shared().currentAppleEvent, ev.eventID == 0x6F61_7070 else { return false }
        return ev.paramDescriptor(forKeyword: 0x7072_6474)?.enumCodeValue == 0x6C67_6974
    }

    func applicationWillFinishLaunching(_ n: Notification) {
        if launchedAtLogin() { hidden = true }
    }

    func applicationDidFinishLaunching(_ n: Notification) {
        if launchedAtLogin() || ProcessInfo.processInfo.systemUptime < 180 { hidden = true }
        if let pid = otherInstance() {  // only one silent chat (it owns the camera and the toggle key)
            kill(pid, SIGTERM)
            for _ in 0..<80 where kill(pid, 0) == 0 { usleep(100_000) }
            if kill(pid, 0) == 0 { kill(pid, SIGKILL) }
        }
        start()
    }

    func start() {
        try? FileManager.default.createDirectory(atPath: data, withIntermediateDirectories: true)
        if !FileManager.default.fileExists(atPath: fifo) { mkfifo(fifo, 0o600) }
        if let size = (try? FileManager.default.attributesOfItem(atPath: logPath))?[.size] as? Int, size > 20_000_000 {
            try? FileManager.default.removeItem(atPath: logPath + ".1")
            try? FileManager.default.moveItem(atPath: logPath, toPath: logPath + ".1")
        }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: root + "/.venv/bin/python")
        p.arguments = [root + "/silent_chat.py", "--speak", "--commands", fifo] + (hidden ? ["--hidden"] : [])
        p.currentDirectoryURL = URL(fileURLWithPath: root)
        var env = ProcessInfo.processInfo.environment
        let home = NSHomeDirectory()
        env["PATH"] = "\(home)/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        env["PYTHONUNBUFFERED"] = "1"
        env["LANG"] = env["LANG"] ?? "en_US.UTF-8"
        env["SILENT_CHAT_APP"] = Bundle.main.bundlePath
        env["SILENT_CHAT_LAUNCHER"] = Bundle.main.executablePath ?? ""
        p.environment = env
        let fd = open(logPath, O_WRONLY | O_APPEND | O_CREAT, 0o644)
        let log = fd >= 0 ? FileHandle(fileDescriptor: fd, closeOnDealloc: true) : FileHandle.nullDevice
        p.standardOutput = log
        p.standardError = log
        p.standardInput = FileHandle.nullDevice
        p.terminationHandler = { proc in DispatchQueue.main.async { self.ended(proc) } }
        do {
            try p.run()
            child = p
        } catch {
            fail("Couldn't start silent chat: \(error.localizedDescription)")
        }
    }

    func ended(_ p: Process) {
        child = nil
        if stopping {
            NSApp.reply(toApplicationShouldTerminate: true)
            return
        }
        let code = p.terminationStatus
        let bySignal = p.terminationReason == .uncaughtSignal
        if !bySignal && code == 75 {  // Restart (menu)
            hidden = false
            start()
            return
        }
        if (!bySignal && code == 0) || (bySignal && [SIGTERM, SIGKILL, SIGINT, SIGHUP].contains(code)) {
            NSApp.terminate(nil)  // quit on purpose
            return
        }
        crashes = crashes.filter { $0.timeIntervalSinceNow > -600 } + [Date()]
        if crashes.count <= 3 {
            hidden = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) { self.start() }
        } else {
            fail("Silent chat keeps stopping (\(bySignal ? "signal" : "exit code") \(code)).")
        }
    }

    func fail(_ message: String) {
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.messageText = "Silent Chat"
        alert.informativeText = message + "\n\nThe log is " + logPath
        alert.addButton(withTitle: "Open Log")
        alert.addButton(withTitle: "Quit")
        if alert.runModal() == .alertFirstButtonReturn {
            NSWorkspace.shared.open(URL(fileURLWithPath: logPath))
        }
        NSApp.terminate(nil)
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if let p = child, p.isRunning { kill(p.processIdentifier, SIGUSR1) }  // show the overlay
        return false
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let p = child, p.isRunning else { return .terminateNow }
        stopping = true
        kill(p.processIdentifier, SIGTERM)  // silent chat closes the camera, saves clips and exits
        DispatchQueue.main.asyncAfter(deadline: .now() + 10) {
            if let p = self.child, p.isRunning { kill(p.processIdentifier, SIGKILL) }
        }
        return .terminateLater
    }
}

let args = CommandLine.arguments
if args.count >= 3 && args[1] == "--login" {
    exit(loginItem(args[2]))
}
let app = NSApplication.shared
let launcher = Launcher()
app.delegate = launcher
app.setActivationPolicy(.accessory)
app.run()
