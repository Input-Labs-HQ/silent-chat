#!/bin/zsh
# Silent Chat setup. Double-click it (or run ./setup.command). Apple Silicon Mac, macOS 14 or later. No sudo.
#
# Downloads about 2.5 GB in total: Python packages (~1 GB), the Auto-AVSR lip-reading model (~1 GB, research and
# non-commercial use only: see THIRD_PARTY.md), MediaPipe face models (~4 MB). Whisper (~1.6 GB) downloads the
# first time you use the mic. Then it builds "Silent Chat.app" in ~/Applications and opens it.
set -e
cd "$(dirname "$0")"

missing=0
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1. $2"; missing=1; }; }
need uv "Install it: brew install uv (or see https://docs.astral.sh/uv/)."
need ffmpeg "Install it: brew install ffmpeg."
need git "Install the Xcode Command Line Tools: xcode-select --install."
xcrun --find swiftc >/dev/null 2>&1 || { echo "Missing: the Swift compiler. Run: xcode-select --install"; missing=1; }
[ "$(uname -m)" = "arm64" ] || { echo "Silent Chat needs an Apple Silicon Mac (it uses Metal and MLX)."; missing=1; }
[ $missing = 0 ] || exit 1
command -v claude >/dev/null 2>&1 || [ -x "$HOME/.local/bin/claude" ] || echo "Note: install Claude Code (https://claude.com/claude-code) and run 'claude' once to sign in. Silent Chat talks to Claude through it, on your own plan."

echo "== Python environment"
[ -d .venv ] || uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python -r requirements.txt

echo "== Auto-AVSR (lip-reading code)"
if [ ! -d third_party/auto_avsr ]; then
  git clone --quiet https://github.com/mpc001/auto_avsr third_party/auto_avsr
  git -C third_party/auto_avsr checkout --quiet 182b628
  git -C third_party/auto_avsr apply "$PWD/patches/auto_avsr-ctc-device.patch"
fi

echo "== Models"
mkdir -p models
[ -f models/vsr_trlrs2lrs3vox2avsp_base.pth ] || .venv/bin/gdown 1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_ -O models/vsr_trlrs2lrs3vox2avsp_base.pth
[ -f models/blaze_face_short_range.tflite ] || curl -fsSL -o models/blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
[ -f models/face_landmarker.task ] || curl -fsSL -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

echo "== The app"
.venv/bin/python make_app.py --login
open "$HOME/Applications/Silent Chat.app"
echo
echo "Done. Silent Chat is in your menu bar (the mouth icon) and opens at login."
echo "macOS will ask for Camera, Accessibility and Input Monitoring for Silent Chat: allow them. Then tap \\ anywhere."
