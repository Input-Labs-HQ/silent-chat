"""Measure what the screen summary costs and saves: the same questions about the front window, asked with and
without it, each in a fresh Haiku session with the live system prompt. Reports tokens, model calls, tool calls and
time per question. Claude may only look (ui, see, apps, skill), never act, so it's safe to run while you work.

Needs Silent Chat's Accessibility permission, so run it through the app: type  /eval context  in its console (or
printf '/eval context\\n' > data/silent_chat/commands.fifo). Results: data/eval_context.txt
"""
import json
import statistics
import sys
import time

import control
from claude_session import MODELS, ClaudeSession
from silent_chat import LIVE_SYSTEM, MAC_CONTROL, build_message

QUESTIONS = [  # about what's in front ...
    "What app and window am I looking at?",
    "Which buttons can I press here?",
    "Is there a text field here, and what's in it?",
    "What page or conversation is this?",
    "Summarise what's on my screen in one sentence.",
    "What would I press to go back or close this?",
    # ... and two that have nothing to do with the screen (the summary is pure overhead there)
    "What's 18 times 7?",
    "Tell me a one-line fun fact.",
]
LOOK_ONLY = ["Bash(./macctl ui:*)", "Bash(./macctl see:*)", "Bash(./macctl apps:*)", "Bash(./macctl skill:*)",
             "Bash(./macctl context:*)", "Read(./screens/**)"]


def setup(workdir):
    from pathlib import Path
    root = Path(__file__).resolve().parent
    w = Path(workdir) / "macctl"
    w.write_text(f'#!/bin/zsh\nexec "{root}/.venv/bin/python" "{root}/mac.py" "$@"\n')
    w.chmod(0o755)
    (Path(workdir) / "screens").mkdir(exist_ok=True)


def ask(question, screen):
    s = ClaudeSession(LIVE_SYSTEM + "\n" + MAC_CONTROL, model=MODELS["haiku"], effort="low", tools="Bash,Read",
                      allowed=LOOK_ONLY, setup=setup)
    part = {"clip": None, "hyps": None, "typed": question}
    s.send(build_message([part], [], "", "", screen))
    t0, tools, text, info = time.monotonic(), [], "", {}
    while time.monotonic() - t0 < 120:
        for ev in s.poll():
            if ev[0] == "tool":
                tools.append((ev[2].get("command") or ev[2].get("file_path") or "")[:80])
            elif ev[0] == "done":
                text, info = ev[1], ev[3] if len(ev) > 3 else {}
            elif ev[0] == "exit":
                info = info or {"error": "exited"}
        if info:
            break
        time.sleep(0.05)
    s.close()
    u = info.get("usage") or {}
    return {"q": question, "screen_chars": len(screen), "new_in": (u.get("input_tokens") or 0) +
            (u.get("cache_creation_input_tokens") or 0), "cached_in": u.get("cache_read_input_tokens") or 0,
            "out": u.get("output_tokens") or 0, "calls": info.get("num_turns") or 0, "tools": tools,
            "s": round(time.monotonic() - t0, 1), "answer": " ".join(text.split())[:160]}


CONDS = ("without", "with")


def main():
    global CONDS
    if "--with-only" in sys.argv:  # (re-measuring a smaller summary: the "without" side doesn't change)
        sys.argv.remove("--with-only")
        CONDS = ("with",)
    out = open(sys.argv[1] if len(sys.argv) > 1 else "data/eval_context.txt", "w")

    def say(line):
        print(line, flush=True)
        out.write(line + "\n")
        out.flush()

    ask("Say OK.", "")  # (warm-up: the shared system prompt gets cached, so neither side pays for that)
    rows = {"without": [], "with": []}
    say(f"front window: {control.context().splitlines()[0] if control.context() else '?'}")
    for q in QUESTIONS:
        ctx = control.context()
        for name, screen in (("without", ""), ("with", "Screen when they said it:\n" + ctx if ctx else "")):
            if name not in CONDS:
                continue
            r = ask(q, screen)
            rows[name].append(r)
            say(json.dumps({"cond": name, **r}))
    say("")
    for name, sel in rows.items():
        if not sel:
            continue
        for label, part in (("screen questions", sel[:6]), ("other questions", sel[6:])):
            m = lambda k: statistics.mean(r[k] for r in part)
            say(f"{name:>7} screen summary, {label}: new input {m('new_in'):.0f} tokens, cached {m('cached_in'):.0f}, "
                f"output {m('out'):.0f}, model calls {m('calls'):.1f}, tool calls "
                f"{statistics.mean(len(r['tools']) for r in part):.1f}, {m('s'):.1f} s")
    sizes = [r["screen_chars"] for r in rows["with"]]
    say(f"screen summary size: {min(sizes)}-{max(sizes)} chars (~{min(sizes) // 4}-{max(sizes) // 4} tokens)")


if __name__ == "__main__":
    main()
