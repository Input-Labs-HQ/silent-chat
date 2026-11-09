#!/bin/zsh
# Double-click (or `open record_test.command`) to record one test clip in Terminal and transcribe it.
cd "$(dirname "$0")"
.venv/bin/python vsr.py --record --save-crop 2>&1 | grep --line-buffered -v "^W0000\|^I0000" | tee recordings/last_run.txt
echo; echo "Done. You can close this window."
