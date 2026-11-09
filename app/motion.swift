// "Silent Chat Motion": streams AirPods head motion to silent chat (for nod-to-confirm / shake-to-cancel and
// gaze). It's its own tiny background app, so macOS asks for Motion permission for it alone and Silent Chat's
// other permissions stay as they are.
//
//   open -g -a "Silent Chat Motion" --args --fifo <path>
//
// Writes one JSON line per event into the named pipe silent chat reads: {"status": ...} when AirPods connect or
// disconnect, then {"t", "pitch", "yaw", "roll" (radians), "rx", "ry", "rz" (rad/s)} at ~25 Hz while they're in
// your ears. Quits when silent chat stops reading. Built by make_app.py.
import AppKit
import CoreMotion

let args = CommandLine.arguments
guard let at = args.firstIndex(of: "--fifo"), at + 1 < args.count else {
    FileHandle.standardError.write("usage: --fifo <path>\n".data(using: .utf8)!)
    exit(2)
}
signal(SIGPIPE, SIG_IGN)
let fd = open(args[at + 1], O_WRONLY)  // waits until silent chat opens the pipe for reading
if fd < 0 { exit(3) }

func send(_ line: String) {
    let bytes = Array((line + "\n").utf8)
    if write(fd, bytes, bytes.count) < 0 { exit(0) }  // silent chat went away
}

class Watcher: NSObject, CMHeadphoneMotionManagerDelegate {
    let manager = CMHeadphoneMotionManager()

    func start() {
        manager.delegate = self
        let auth = CMHeadphoneMotionManager.authorizationStatus().rawValue
        send("{\"status\": \"started\", \"available\": \(manager.isDeviceMotionAvailable), \"auth\": \(auth)}")
        manager.startDeviceMotionUpdates(to: OperationQueue.main) { motion, error in
            if let error = error {
                send("{\"status\": \"error\", \"message\": \"\(error.localizedDescription)\"}")
                return
            }
            guard let m = motion else { return }
            let a = m.attitude, r = m.rotationRate
            send(String(format: "{\"t\": %.3f, \"pitch\": %.4f, \"yaw\": %.4f, \"roll\": %.4f, \"rx\": %.3f, " +
                        "\"ry\": %.3f, \"rz\": %.3f}", m.timestamp, a.pitch, a.yaw, a.roll, r.x, r.y, r.z))
        }
    }

    func headphoneMotionManagerDidConnect(_ manager: CMHeadphoneMotionManager) {
        send("{\"status\": \"connected\"}")
    }

    func headphoneMotionManagerDidDisconnect(_ manager: CMHeadphoneMotionManager) {
        send("{\"status\": \"disconnected\"}")
    }
}

let watcher = Watcher()
watcher.start()
Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { _ in send("{\"ping\": 1}") }  // notices a quit reader
NSApplication.shared.setActivationPolicy(.prohibited)
RunLoop.main.run()
