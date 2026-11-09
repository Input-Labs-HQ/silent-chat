"""A small model on the Mac that handles hands-free reads, so Claude only sees real requests.

Hands-free lip-reading also fires when you talk to someone, chew or think aloud. For each read that passes the free
word filter (presence.plausible_request), Qwen3 1.7B (4-bit, MLX, on the GPU) decides:

  ignore   not a request for silent chat (it's struck through on the island, nothing happens)
  act      a simple, harmless one it can do itself with one Mac command: pause / play / next / previous song,
           open an app, scroll, search the web (closing or quitting anything goes to Claude: a misread
           shouldn't be able to close your work)
  claude   a real request that needs Claude (play a genre or artist, messages, questions ...), passed on with
           the words fixed ("please write up on spotify" -> "play rap on spotify")

Runs locally: nothing leaves the Mac, no plan usage. Loaded in the background the first time it's needed
(~1.5 GB of memory). Model: mlx-community/Qwen3-1.7B-4bit.
"""
import json
import re
import threading
import time

MODEL = "mlx-community/Qwen3-1.7B-4bit"

SYSTEM = """You turn lip-read text into commands for a Mac assistant. Lip-reading misreads similar mouth shapes
(p/b/m, f/v, t/d/n, s/z, r/w), so fix the words to the most likely request.
Answer with one line of JSON:
- {"kind": "act", "text": ..., "cmd": ...} when one of these commands does it: spotify pause, spotify play, spotify next,
  spotify previous, open <App>, scroll down, scroll up, web <search words>
- {"kind": "claude", "text": ...} for any other request (playing a genre/artist/song, messages, questions, pressing or
  reading things on screen, closing things)
- {"kind": "ignore"} ONLY when it is clearly not a request: someone talking to another person, narration, filler."""

SHOTS = [("PAUSE THE MUSIC", '{"kind": "act", "text": "pause the music", "cmd": "spotify pause"}'),
         ("PLEASE WRITE UP ON SPOTIFY", '{"kind": "claude", "text": "play rap on Spotify"}'),
         ("SO I'M GOING TO TALK A LITTLE BIT ABOUT", '{"kind": "ignore"}'),
         ("OPEN GOOGLE CHROME PLEASE", '{"kind": "act", "text": "open Google Chrome", "cmd": "open Google Chrome"}'),
         ("SUMMARIZE THIS ALSO", '{"kind": "claude", "text": "summarize this"}'),
         ("I HOPE YOU HAD A GOOD DAY", '{"kind": "ignore"}'),
         ("SEARCH FOR NORTH KOREAN NEWS",
          '{"kind": "act", "text": "search North Korean news", "cmd": "web North Korean news"}')]

PLAY_ONLY = {"play", "resume", "the", "music", "spotify", "please", "on", "song", "again", "unpause", "it", "my"}
CMD = re.compile(r"^(spotify (pause|play|next|previous)|open [\w .&'-]{1,40}|"
                 r"scroll (up|down)|web .{1,120})$", re.I)


class Router:
    def __init__(self):
        self.model = self.tok = None
        self.loading, self.error, self.lock = False, None, threading.Lock()
        self.stats = []  # ms per decision
        self.last_used = time.monotonic()

    def load_async(self):
        if self.model is not None or self.loading:
            return
        self.loading = True

        def go():
            try:
                from mlx_lm import load
                self.model, self.tok = load(MODEL)
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
            self.loading = False
        threading.Thread(target=go, daemon=True).start()

    @property
    def ready(self):
        return self.model is not None

    @property
    def usable(self):
        """Can it decide? (Loaded, or loadable: it unloads when idle and reloads in ~2 s.)"""
        return self.error is None

    def _load_now(self):
        with self.lock:
            if self.model is None:
                from mlx_lm import load
                self.model, self.tok = load(MODEL)

    def unload_if_idle(self, seconds=300):
        """Free its ~0.8 GB after a few minutes without hands-free reads."""
        if self.model is not None and time.monotonic() - self.last_used > seconds and not self.lock.locked():
            import gc
            import mlx.core as mx
            with self.lock:
                self.model = self.tok = None
                gc.collect()
                mx.clear_cache()

    def decide(self, hyps, front="", apps=()):
        """-> {"kind": "ignore"|"act"|"claude", "text": str, "cmd": [macctl args] or None, "ms": int}"""
        from mlx_lm import generate
        self._load_now()
        self.last_used = time.monotonic()
        msgs = [{"role": "system", "content": SYSTEM}]
        for q, a in SHOTS:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": hyps[0][0]})
        try:
            text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                                enable_thinking=False)
        except TypeError:
            text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        t0 = time.monotonic()
        with self.lock:
            out = generate(self.model, self.tok, prompt=text, max_tokens=60, verbose=False)
        ms = round((time.monotonic() - t0) * 1000)
        self.stats.append(ms)
        return parse(out, hyps, ms)


def _classify(self, system, text, max_tokens=8, shots=()):
    """A one-word judgement from the same small model (privacy.py uses it). shots: [(input, answer)]. -> answer."""
    from mlx_lm import generate
    self._load_now()
    self.last_used = time.monotonic()
    msgs = [{"role": "system", "content": system}]
    for q, a in shots:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": text})
    try:
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    with self.lock:
        return generate(self.model, self.tok, prompt=prompt, max_tokens=max_tokens, verbose=False)


Router.classify = _classify


def parse(out, hyps, ms=0):
    out = re.sub(r"(?s)<think>.*?</think>", "", out)
    m = re.search(r"\{.*?\}", out, re.S)
    d = {}
    if m:
        try:
            d = json.loads(m.group(0))
        except ValueError:
            d = {}
    kind = d.get("kind") if d.get("kind") in ("ignore", "act", "claude") else "claude"  # (unsure: let Claude decide)
    text = str(d.get("text") or (hyps[0][0] if hyps else "")).strip()[:200]
    cmd = None
    if kind == "act":
        c = str(d.get("cmd") or "").strip()
        c = re.sub(r"^spotify skip$", "spotify next", c, flags=re.I)
        words = set(re.findall(r"[a-z']+", text.lower()))
        if c.lower() == "spotify play" and words - PLAY_ONLY:
            c = ""  # ("play rap", "play Drake": needs a search, so Claude does it)
        if c.lower().startswith("web ") and not re.match(r"(search|look up|google|find)\b", text, re.I):
            c = ""  # (only a web search when they asked for one)
        if CMD.match(c):
            if c.lower().startswith("web "):
                from urllib.parse import quote_plus
                cmd = ["url", "https://www.google.com/search?q=" + quote_plus(c[4:].strip())]
            elif c.lower().startswith("scroll "):
                cmd = ["scroll", c.split()[1].lower()]
            else:
                cmd = c.split(" ", 1) if not c.lower().startswith("spotify") else c.lower().split()
        else:
            kind = "claude"  # (not one it may run itself)
    return {"kind": kind, "text": text, "cmd": cmd, "ms": ms}
