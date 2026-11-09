"""Build "Silent Chat.app": silent chat as a menu bar app (no Terminal window), in ~/Applications.

  .venv/bin/python make_app.py              # build it (only if app/launcher.swift changed) and install it
  .venv/bin/python make_app.py --login      # ... and open it at login, so the toggle key works from start-up
  .venv/bin/python make_app.py --force      # rebuild anyway (macOS then asks for the permissions again)

The app is a tiny native launcher (app/launcher.swift) that runs silent_chat.py from this folder with the venv's
Python and stays its parent, so macOS asks for permissions as "Silent Chat". Python changes need no rebuild:
choose Restart in the menu bar. It's signed ad hoc (just for this Mac): a rebuild gets a new signature, and macOS
forgets the permissions it had, which is why it only rebuilds when the launcher's source changes.
"""
import argparse
import hashlib
import plistlib
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = Path.home() / "Applications" / "Silent Chat.app"
BUNDLE_ID = "local.silentchat"
INFO = {
    "CFBundleIdentifier": BUNDLE_ID,
    "CFBundleName": "Silent Chat",
    "CFBundleDisplayName": "Silent Chat",
    "CFBundleExecutable": "SilentChat",
    "CFBundleIconFile": "AppIcon",
    "CFBundlePackageType": "APPL",
    "CFBundleShortVersionString": "1.0",
    "CFBundleVersion": "1",
    "LSMinimumSystemVersion": "14.0",
    "LSUIElement": True,  # menu bar only: no Dock icon, no app menu
    "NSHighResolutionCapable": True,
    "NSCameraUsageDescription": "Silent Chat reads your lips through the camera while its overlay is showing.",
    "NSMicrophoneUsageDescription": "Silent Chat hears your voice only while MIC is on and you hold the talk key.",
    "NSAppleEventsUsageDescription": "Silent Chat controls apps like Spotify and arranges windows when you ask it to.",
}


def draw_icon(png, size=1024):
    """A mouth on a blue-violet rounded square, in the macOS icon grid."""
    from AppKit import (NSBitmapImageFileTypePNG, NSBitmapImageRep, NSBezierPath, NSColor, NSGradient,
                        NSGraphicsContext, NSImage, NSImageSymbolConfiguration, NSMakeRect)
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, size, size, 8, 4, True, False, "NSCalibratedRGBColorSpace", 0, 0)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))
    shape = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(100, 100, 824, 824), 186, 186)
    NSGradient.alloc().initWithStartingColor_endingColor_(
        NSColor.colorWithSRGBRed_green_blue_alpha_(0.50, 0.30, 0.93, 1.0),
        NSColor.colorWithSRGBRed_green_blue_alpha_(0.38, 0.66, 1.00, 1.0)).drawInBezierPath_angle_(shape, 90)
    cfg = NSImageSymbolConfiguration.configurationWithPointSize_weight_(430, 0.3).configurationByApplyingConfiguration_(
        NSImageSymbolConfiguration.configurationWithPaletteColors_([NSColor.whiteColor()]))
    sym = NSImage.imageWithSystemSymbolName_accessibilityDescription_("mouth.fill", None).imageWithSymbolConfiguration_(cfg)
    w, h = sym.size().width, sym.size().height
    sym.drawInRect_(NSMakeRect((size - w) / 2, (size - h) / 2, w, h))
    NSGraphicsContext.restoreGraphicsState()
    rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, {}).writeToFile_atomically_(str(png), True)


def make_icns(dest, work):
    png = work / "icon.png"
    draw_icon(png)
    iconset = work / "AppIcon.iconset"
    iconset.mkdir()
    for s in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            name = f"icon_{s}x{s}{'@2x' if scale == 2 else ''}.png"
            subprocess.run(["sips", "-z", str(s * scale), str(s * scale), str(png), "--out", str(iconset / name)],
                           check=True, capture_output=True)
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(dest)], check=True)


def build(force=False):
    source = (ROOT / "app" / "launcher.swift").read_text().replace("@ROOT@", str(ROOT))
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    stamp = APP / "Contents" / "Resources" / "build.txt"
    if APP.exists() and stamp.exists() and stamp.read_text() == digest and not force:
        print(f"{APP} is up to date (no rebuild, so its permissions stay).")
        return False
    work = Path(tempfile.mkdtemp(prefix="silent_chat_app_"))
    app = work / APP.name
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "Resources").mkdir()
    (work / "launcher.swift").write_text(source)
    print("Compiling the launcher ...")
    subprocess.run(["swiftc", "-O", "-swift-version", "5", "-target", "arm64-apple-macos14.0",
                    "-o", str(app / "Contents" / "MacOS" / "SilentChat"), str(work / "launcher.swift")], check=True)
    with open(app / "Contents" / "Info.plist", "wb") as f:
        plistlib.dump(INFO, f)
    make_icns(app / "Contents" / "Resources" / "AppIcon.icns", work)
    (app / "Contents" / "Resources" / "build.txt").write_text(digest)
    subprocess.run(["codesign", "--force", "--sign", "-", "--identifier", BUNDLE_ID, str(app)], check=True)
    subprocess.run(["codesign", "--verify", "--strict", str(app)], check=True)
    APP.parent.mkdir(exist_ok=True)
    if APP.exists():
        shutil.rmtree(APP)
    shutil.move(str(app), str(APP))
    shutil.rmtree(work, ignore_errors=True)
    subprocess.run(["/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/"
                    "lsregister", "-f", str(APP)], capture_output=True)  # so Spotlight/Launchpad know it now
    print(f"Built {APP}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild even if the launcher didn't change")
    ap.add_argument("--login", action="store_true", help="open Silent Chat at login")
    args = ap.parse_args()
    build(args.force)
    if args.login:
        r = subprocess.run([str(APP / "Contents" / "MacOS" / "SilentChat"), "--login", "on"],
                           capture_output=True, text=True)
        print(f"Open at login: {r.stdout.strip()} {r.stderr.strip()}".strip())


if __name__ == "__main__":
    main()
