#!/bin/zsh
# Double-click (or `open run_test.command`) to run the full lip-reading test in Terminal:
# all three conditions, then transcription and scoring. Safe to quit (Q) and re-run; it resumes.
cd "$(dirname "$0")"
mkdir -p results
.venv/bin/python run_test.py "$@" 2>&1 | grep --line-buffered -v "^W0000\|^I0000" | tee -a results/run_log.txt
echo; echo "Done. Results are in the results folder. You can close this window."
