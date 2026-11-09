"""Direct connections to your claude.ai connectors (Todoist, Spotify) for silent chat's Claude session.

Silent chat's `claude -p` session normally loads no MCP servers, which keeps it fast. This picks out just the
connectors listed in CONNECTORS from the ones connected on your claude.ai account and passes them to the CLI
explicitly, with only the tools voice control needs. The rest stay out of the context, and the ones that delete
things are never allowed.

  .venv/bin/python connectors.py      # which are connected (also refreshes the cache)

Connect more in claude.ai > Settings > Connectors, then choose Restart in silent chat's menu bar icon.
"""
import json
import subprocess
import tempfile
import time

from learning import DATA

CACHE = DATA / "connectors.json"

CONNECTORS = {  # claude.ai connector name -> server name here, tools Claude may use, what Claude is told
    "Todoist": {
        "server": "todoist",
        "tools": ["find-tasks", "find-tasks-by-date", "find-projects", "get-overview", "add-tasks", "complete-tasks",
                  "uncomplete-tasks", "update-tasks", "reschedule-tasks"],
        "prompt": (
            "TODOIST (direct connection: your todoist tools; use them, not the Todoist app, for anything about the "
            "user's tasks or to-do list): find tasks (today, overdue, upcoming, by project or words), add tasks (with "
            "due dates and priorities), complete, uncomplete, reschedule and update them. You cannot delete tasks or "
            "projects. The lip-reader mishears: to complete or change a task, pick the one that matches by meaning; "
            "if several could match or none fits well, ask which instead of guessing. After a change, say in a few "
            "words what you did (e.g. 'Added \"call mom\" for tomorrow'). When listing tasks, keep it short: how many, "
            "then the few that matter most."
        ),
    },
    "Spotify": {
        "server": "spotify",
        # (not remove_*; get_currently_playing needs a fresh Spotify sign-in, and './macctl spotify now' works)
        "tools": ["search", "save_to_library", "generate_playlist"],
        "prompt": (
            "SPOTIFY (direct connection: your spotify tools): search Spotify's catalogue, save to the user's library and "
            "generate playlists. To PLAY something, find it with the spotify search tool, then run "
            "'./macctl spotify play <its uri, e.g. spotify:track:...>' (not the play_uri). What's playing: './macctl "
            "spotify now'. macctl drives "
            "the Spotify app on this Mac: play, pause, next, volume, shuffle). Don't use WebSearch for Spotify links "
            "while the spotify tools work."
        ),
    },
}


def discover(timeout=25):
    """Ask the claude CLI which claude.ai connectors are connected (no model call). Returns
    {name: {"config": ..., "tools": [...]}} for the ones in CONNECTORS, and caches it."""
    cmd = ["claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--no-session-persistence", "--setting-sources", "project", "--tools", ""]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                         cwd=tempfile.mkdtemp(prefix="claude_connectors_"))
    servers, deadline, n = [], time.monotonic() + timeout, 0
    try:
        while time.monotonic() < deadline:
            time.sleep(1.5)
            n += 1
            p.stdin.write(json.dumps({"type": "control_request", "request_id": f"s{n}",
                                      "request": {"subtype": "mcp_status"}}) + "\n")
            p.stdin.flush()
            for line in p.stdout:  # read until this request's answer
                ev = json.loads(line) if line.startswith("{") else {}
                if ev.get("type") == "control_response" and ev.get("response", {}).get("request_id") == f"s{n}":
                    servers = ev["response"].get("response", {}).get("mcpServers", [])
                    break
            if servers and not any(s.get("status") == "pending" for s in servers):
                break
    finally:
        try:
            p.stdin.close()
            p.wait(5)
        except Exception:
            p.kill()
    found, status = {}, {}
    for s in servers:
        name = s.get("name", "").removeprefix("claude.ai ")
        if name in CONNECTORS:
            status[name] = s.get("status")
            if s.get("status") == "connected" and s.get("config", {}).get("type") == "claudeai-proxy":
                found[name] = {"config": s["config"], "tools": [t["name"] for t in s.get("tools", [])]}
    if servers:  # (an empty list means the CLI didn't answer: keep the old cache)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps({"found": found, "status": status}, indent=1))
    return found


def cached():
    try:
        return json.loads(CACHE.read_text())["found"]
    except (OSError, ValueError, KeyError):
        return {}


def session_options(found):
    """What ClaudeSession needs for these connectors: (mcp config, allowed tools, disallowed tools, prompt)."""
    mcp, allowed, disallowed, prompt = {}, [], [], []
    for name, info in found.items():
        spec = CONNECTORS[name]
        server = spec["server"]
        mcp[server] = info["config"]
        wanted = set(spec["tools"])
        allowed += [f"mcp__{server}__{t}" for t in info["tools"] if t in wanted]
        disallowed += [f"mcp__{server}__{t}" for t in info["tools"] if t not in wanted]  # kept out of the context
        prompt.append(spec["prompt"])
    return {"mcpServers": mcp} if mcp else None, allowed, disallowed, "\n".join(prompt)


def label(tool):
    """mcp__todoist__add-tasks -> ("Todoist", "add tasks")."""
    _, server, name = tool.split("__", 2)
    title = next((n for n, s in CONNECTORS.items() if s["server"] == server), server)
    return title, name.replace("-", " ").replace("_", " ")


if __name__ == "__main__":
    found = discover()
    status = json.loads(CACHE.read_text()).get("status", {}) if CACHE.exists() else {}
    for name, spec in CONNECTORS.items():
        if name in found:
            usable = [t for t in found[name]["tools"] if t in spec["tools"]]
            print(f"{name}: connected; silent chat may use {', '.join(usable)}")
        else:
            print(f"{name}: {status.get(name, 'not connected')} (connect it in claude.ai > Settings > Connectors)")
