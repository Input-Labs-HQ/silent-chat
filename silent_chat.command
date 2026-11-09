#!/bin/zsh
# Silent chat console. Silent chat normally runs as the menu bar app "Silent Chat" (~/Applications; it opens at
# login, so tap \ anywhere to show/hide it). This window shows its live log and lets you type a message, or
# /fix <text>, /learn <words>, /voice, /face, /type, /mic, /hide, /show, /status, /restart, /quit.
# Closing this window doesn't stop silent chat. If the app isn't built yet (make_app.py), this starts silent chat
# in the background from here instead (Terminal then needs the camera etc. permissions).
#
# Talk: hold Option (in any app) while you mouth, or blink long to start; in TYPE mode a smile deletes words (1/s)
# and an eyebrow raise presses Enter. Double-tap \ to type a backslash.
cd "$(dirname "$0")"
mkdir -p data/silent_chat
LOG=data/silent_chat/session.log
FIFO=data/silent_chat/commands.fifo
[ -p "$FIFO" ] || { rm -f "$FIFO"; mkfifo "$FIFO"; }

PIDFILE=data/silent_chat/silent_chat.pid
running() { [ -f "$PIDFILE" ] && ps -p "$(cat "$PIDFILE")" -o command= 2>/dev/null | grep -q "python.*silent_chat.py"; }

APP="$HOME/Applications/Silent Chat.app"
if running; then
  echo "Silent chat is running."
elif [ -d "$APP" ]; then
  echo "Starting the Silent Chat app ..."
  open "$APP"
else
  echo "Starting silent chat in the background ..."
  PYTHONUNBUFFERED=1 nohup .venv/bin/python silent_chat.py --speak --commands "$FIFO" "$@" < /dev/null >> "$LOG" 2>&1 &
  disown 2>/dev/null
fi
echo "Tap \\ anywhere to show/hide it (or use the mouth icon in the menu bar). Closing this window doesn't stop it."
echo "Type a message or a command here: /fix <text>, /learn <words>, /voice, /face, /type, /mic, /hide, /show, /status, /restart, /quit"
echo

tail -n 3 -F "$LOG" 2>/dev/null | grep --line-buffered -v "^W0000\|^I0000" &
TAIL=$!
trap "kill $TAIL 2>/dev/null" EXIT
while IFS= read -r line; do
  [ -z "$line" ] && continue
  if ! running; then echo "Silent chat isn't running. Open the Silent Chat app (or double-click this file again) to start it."; continue; fi
  printf '%s\n' "$line" > "$FIFO"
  if [ "$line" = "/quit" ]; then sleep 4; echo "Silent chat has quit. You can close this window."; break; fi
done
