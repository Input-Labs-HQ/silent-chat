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
            elif ev.get("type") == "result":
                self.events.put(("done", ev.get("result") or "", ev.get("is_error", False)))
        self.events.put(("exit", proc.poll()))

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def send(self, text):
        """Start a turn. Its reply arrives through poll() as ("text", delta) events, then ("done", full_text, is_error)."""
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
