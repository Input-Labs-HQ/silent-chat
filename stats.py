"""How well silent chat is working for you, from its own logs (all on this Mac).

  lip-reading   of the clips you went on to use, how many you had to correct
  instant       how often a mouthed instant command was matched clearly enough to run
  hands-free    starts per hour, and how many turned out real vs false
  speed         time from sending to Claude's first words and to done; instant commands
  replies       your thumbs up / down

/stats prints it; Settings > Show Stats shows it in a window. `python stats.py` from Terminal too.
"""
import json
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "silent_chat"


def _jsonl(path):
    out = []
    try:
        for line in open(path):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def _since(rows, days, key="at"):
    cut = datetime.now() - timedelta(days=days)
    keep = []
    for r in rows:
        try:
            if datetime.strptime(str(r.get(key, ""))[:19], "%Y-%m-%d %H:%M:%S") >= cut:
                keep.append(r)
        except ValueError:
            pass
    return keep


def summary(days=7):
    lines = [f"Silent Chat, last {days} days"]
    # lip-reading: confirmed (ok / confirmed / audio) vs corrected (fixed)
    try:
        import csv
        rows = list(csv.DictReader(open(DATA / "labels.csv")))
        rows = [r for r in rows if r.get("time") and
                datetime.strptime(r["time"][:19], "%Y-%m-%d %H:%M:%S") >= datetime.now() - timedelta(days=days)]
        ok = sum(r["status"] in ("ok", "confirmed") for r in rows)
        fixed = sum(r["status"] == "fixed" for r in rows)
        if ok + fixed:
            lines.append(f"Lip-reading: {ok} reads you went on with, {fixed} you corrected "
                         f"({100 * fixed / (ok + fixed):.0f}% needed a correction; Claude also fixes many misreads "
                         "silently).")
    except (OSError, ValueError, KeyError):
        pass
    # instant commands (closed-set)
    cs = _since(_jsonl(DATA / "commands.log"), days, "time")
    if cs:
        acc = sum(bool(r.get("accepted")) for r in cs)
        lines.append(f"Instant commands: {acc} of {len(cs)} mouthed tries matched clearly enough to run.")
    # hands-free
    hf = []
    try:
        cut = datetime.now() - timedelta(days=days)
        for line in open(DATA / "handsfree.log"):
            p = line.split()
            if len(p) > 4 and datetime.strptime(" ".join(p[:2])[:19], "%Y-%m-%d %H:%M:%S") >= cut:
                hf.append(p[4])
    except (OSError, ValueError):
        pass
    starts = hf.count("auto-start")
    if starts:
        used = sum(e in ("auto-used", "auto-answer") or e in ("local-act", "local-claude") for e in hf)
        false = sum(e in ("auto-rejected", "local-ignore", "claude-ignored", "auto-dropped") for e in hf)
        lines.append(f"Hands-free: {starts} starts; {used} were real requests, {false} false starts (dropped "
                     "quietly).")
    # speed
    use = _since(_jsonl(DATA / "usage.log"), days)
    if use:
        ms = [r["ms"] for r in use if r.get("ms")]
        lines.append(f"Claude replies: {len(use)}, typically done in {statistics.median(ms) / 1000:.1f} s "
                     f"(slowest 10%: {sorted(ms)[int(len(ms) * 0.9)] / 1000:.1f} s).")
    first = _since(_jsonl(DATA / "metrics.jsonl"), days)
    fw = [r["first_ms"] for r in first if r.get("first_ms")]
    if fw:
        lines.append(f"First words after sending: typically {statistics.median(fw) / 1000:.1f} s.")
    # votes
    fb = _since(_jsonl(DATA / "feedback.jsonl"), days)
    if fb:
        up = sum(r.get("vote") == "up" for r in fb)
        lines.append(f"Your ratings: {up} 👍, {len(fb) - up} 👎.")
    if len(lines) == 1:
        lines.append("Nothing measured yet.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
