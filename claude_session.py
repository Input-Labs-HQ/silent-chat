"""A long-running Claude conversation through the logged-in `claude` CLI (your claude.ai plan, no API key).

One `claude -p` process stays open with streaming JSON in and out, so each turn skips the CLI start-up (~1-4 s),
the conversation stays in the session's context, and reply text arrives as it is generated (first words in ~1 s).
Skills/plugins, user settings and MCP servers are switched off (except connectors passed in explicitly), so the
context stays small.
"""
import json
import queue
import subprocess
import tempfile
import threading

MODEL = "claude-opus-5-5"
MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
FLAGS = ["--input-format", "stream-json", "--output-format", "stream-json", "--include-partial-messages", "--verbose",
         "--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands", "--setting-sources", "project"]


class ClaudeSession:
    def __init__(self, system, model=MODEL, effort="low", tools="", allowed=(), setup=None, mcp=None, disallowed=()):
        """tools: the CLI tools available ("" = none). allowed: permission rules for tools that may run without
        asking (anything else is refused, since nobody can approve it). setup(workdir) prepares the working folder.
        mcp: {"mcpServers": ...} to load (e.g. claude.ai connectors); disallowed: tools left out of the context."""
        self.system, self.model, self.effort, self.tools, self.allowed = system, model, effort, tools, list(allowed)
        self.mcp, self.disallowed = mcp, list(disallowed)
        self.events, self.busy, self.proc = queue.Queue(), False, None
        self.workdir = tempfile.mkdtemp(prefix="claude_session_")  # neutral cwd: no project files to pick up
        if setup:
            setup(self.workdir)
        self.start()

    def start(self):
        cmd = ["claude", "-p", "--model", self.model, *FLAGS, "--tools", self.tools, "--effort", self.effort,
               "--system-prompt", self.system]
        if self.mcp:
            cmd += ["--mcp-config", json.dumps(self.mcp)]
        if self.allowed:
            cmd += ["--allowedTools", *self.allowed]
        if self.disallowed:
            cmd += ["--disallowedTools", *self.disallowed]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1, cwd=self.workdir)
        self.busy = False
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _read(self, proc):
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "stream_event":
                e = ev.get("event", {})
                if e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta":
                    self.events.put(("text", e["delta"]["text"]))
                elif e.get("type") == "content_block_start" and e.get("content_block", {}).get("type") == "text":
                    self.events.put(("text", "\n"))  # text before and after a tool call are separate lines
            elif ev.get("type") == "assistant":
                for c in ev.get("message", {}).get("content", []):
                    if c.get("type") == "tool_use":
                        self.events.put(("tool", c.get("name", ""), c.get("input", {})))
            elif ev.get("type") == "result":  # (with what the turn used: tokens, model calls, time)
                info = {"usage": ev.get("usage") or {}, "num_turns": ev.get("num_turns"),
                        "duration_ms": ev.get("duration_ms"), "model": self.model}
                self.events.put(("done", ev.get("result") or "", ev.get("is_error", False), info))
        self.events.put(("exit", proc.poll()))

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def send(self, text, full=None):
        """Start a turn. Its reply arrives through poll() as ("text", delta) events, then ("done", full_text, is_error,
        info). full: (Router only) the message with nothing abbreviated, for a model that hasn't seen earlier ones."""
        if not self.alive:
            self.start()
        self.proc.stdin.write(json.dumps(
            {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": text}]}}) + "\n")
        self.proc.stdin.flush()
        self.busy = True

    def interrupt(self):
        """Stop the turn in progress (e.g. the user pressed stop while Claude was acting on their Mac)."""
        if self.alive and self.busy:
            self.proc.stdin.write(json.dumps({"type": "control_request", "request_id": f"stop{id(self)}",
                                              "request": {"subtype": "interrupt"}}) + "\n")
            self.proc.stdin.flush()

    def poll(self):
        out = []
        while not self.events.empty():
            ev = self.events.get()
            if ev[0] in ("done", "exit"):
                self.busy = False
            out.append(ev)
        return out

    def close(self):
        if self.alive:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()


ESCALATE_RULE = (
    "\nYou are the fast first responder (a small model). Handle quick things yourself: chat, short answers, simple "
    "actions on the Mac (open, focus, press, a few steps), Spotify, Todoist, settings. But if the request needs a "
    "stronger model (a task of more than about five steps, writing more than a few sentences, research, maths, "
    "code, careful judgement, or they ask you to think harder or use a better model), reply with exactly ESCALATE "
    "as your whole reply: no HEARD line, no tools, nothing else. A stronger model then takes over with the context."
)


class Router:
    """Haiku first, Opus only when needed: the same interface as ClaudeSession (send, poll, busy, interrupt, close).

    The small model answers most messages itself. For hard ones it replies just "ESCALATE"; that reply is held back
    (never shown or spoken) and the message goes to the big model instead, with the recent conversation, and poll()
    yields ("escalated",) once so the app can say it's thinking harder. The big model starts only when first needed.
    """

    def __init__(self, system, fast=MODELS["haiku"], smart=MODELS["opus"], **kw):
        self.system, self.smart_model, self.kw = system, smart, kw
        self.fast = ClaudeSession(system + ESCALATE_RULE, model=fast, **kw)
        # the big model's process starts now but idles (no tokens used) until a message is escalated to it
        self.smart, self.active, self.phase = ClaudeSession(system, model=smart, **kw), self.fast, "idle"
        self.held, self.message, self.recent = [], "", []  # recent: the last few replies, as context for Opus
        self.pre = None  # what the small model used deciding to hand a message over (added to the big one's info)

    @property
    def busy(self):
        return self.active.busy

    def send(self, text, full=None):
        self.message, self.held, self.phase, self.active, self.pre = full or text, [], "deciding", self.fast, None
        self.fast.send(text)

    def interrupt(self):
        self.active.interrupt()

    def close(self):
        self.fast.close()
        if self.smart is not None:
            self.smart.close()

    def _escalate(self):
        if self.smart is None:
            self.smart = ClaudeSession(self.system, model=self.smart_model, **self.kw)
        context = ("Earlier in this conversation (answered by a faster model):\n" + "\n---\n".join(self.recent) +
                   "\n\n") if self.recent else ""
        self.active, self.phase = self.smart, "smart"
        self.smart.send(context + self.message)

    def poll(self):
        out = []
        for ev in self.active.poll():
            if ev[0] == "done" and isinstance(ev[1], str) and not ev[1].strip().upper().startswith("ESCALATE"):
                self.recent = (self.recent + [ev[1].strip()[:500]])[-6:]
            if self.phase != "deciding":
                if ev[0] == "done" and self.pre and len(ev) > 3:
                    ev = (*ev[:3], {**ev[3], "pre": self.pre})
                    self.pre = None
                out.append(ev)
                continue
            text = "".join(e[1] for e in self.held if e[0] == "text").lstrip().upper()
            maybe_escalate = "ESCALATE".startswith(text[:8]) if len(text) < 8 else text.startswith("ESCALATE")
            if ev[0] == "text":
                self.held.append(ev)
                text = "".join(e[1] for e in self.held).lstrip().upper()
                maybe_escalate = "ESCALATE".startswith(text[:8]) if len(text) < 8 else text.startswith("ESCALATE")
                if not maybe_escalate:  # a normal reply: let it through
                    self.phase = "fast"
                    out += self.held
                    self.held = []
            elif ev[0] in ("done", "exit") and text.startswith("ESCALATE") and not (ev[0] == "done" and ev[2]):
                self.held = []
                self.pre = ev[3] if ev[0] == "done" and len(ev) > 3 else None
                self._escalate()
                out.append(("escalated",))
                return out
            else:  # a tool call, the end of the turn, or an error: it's the small model's turn after all
                self.phase = "fast" if ev[0] not in ("done", "exit") else "idle"
                out += self.held + [ev]
                self.held = []
        return out
