"""Keeps private things on your screen away from Claude.

Everything that reads your screen for Claude goes through here first: the screen summary sent with each message,
what's under your gaze, and Claude's own 'ui' / 'see' / 'context' look-ups. Two layers, cheapest first:

  1. rules (free, microseconds): password managers and banking / tax / health / crypto sites are private
     outright; card numbers, bank and ID numbers, one-time codes, API keys and tokens are blanked anywhere;
  2. the small local model (local_router: Qwen3 1.7B, on your Mac) reads the summary once per window (cached),
     and says whether the window as a whole is private. That runs while you're still mouthing, so it adds no wait;
     if its answer isn't ready in time, only the app's name is sent (it fails closed).

A private window becomes one line for Claude: "Front: Chase (private: financial, not shared)". Claude's 'see'
(screenshot) and 'ui' refuse on a window marked private. The verdicts are kept in data/silent_chat/privacy.json
(window names and a yes/no, nothing else) so macctl, which runs as its own process, can check them.
"""
import hashlib
import json
import re
import threading
import time
from pathlib import Path

STATE = Path(__file__).resolve().parent / "data" / "silent_chat" / "privacy.json"

PRIVATE_APPS = {"1password", "1password 7", "bitwarden", "keychain access", "passwords", "dashlane", "lastpass",
                "keeper", "enpass", "proton pass", "authy", "google authenticator", "yubico authenticator"}
PRIVATE_SITES = re.compile(
    r"\b(bank|banking|chase|wells ?fargo|bank of america|citi(bank)?|capital one|amex|american express|schwab|"
    r"fidelity|vanguard|robinhood|coinbase|binance|kraken|metamask|paypal|venmo|cash ?app|zelle|wise\.com|revolut|"
    r"monzo|turbotax|irs\.gov|hrblock|mychart|patient portal|health records?|lab results|kaiser|"
    r"sign in to|log ?in to|password reset|reset your password|verification code|two-factor|2fa|"
    r"private browsing|incognito)\b", re.I)
SECRETS = [  # (pattern, replacement): blanked in anything sent to Claude
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[number hidden]"),  # card / account numbers
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[ID hidden]"),  # SSN
    (re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}\b"), "[account hidden]"),  # IBAN
    (re.compile(r"\b(sk-[A-Za-z0-9_-]{16,}|sk-ant-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_\w{20,}|"
                r"xox[abpr]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+)"),
     "[key hidden]"),
    (re.compile(r"(?i)\b(code|passcode|otp|pin)\b([^0-9\n]{0,14})\d{4,8}\b"), r"\1\2[code hidden]"),
    (re.compile(r"(?i)(password|passwd|secret|token|api[ _-]?key)(\s*[:=]\s*)\S+"), r"\1\2[hidden]"),
]

JUDGE = """You check what's on someone's Mac screen before it's shared with an AI assistant. Private means: bank or
card accounts and balances, investments, tax or payroll, medical or health records, passwords or secret keys, ID
documents, or legal papers. Ordinary chats, social media, email lists, music, code, documents and websites are NOT
private. Answer with one word: PRIVATE-financial, PRIVATE-medical, PRIVATE-credentials, PRIVATE-identity, or OK."""


SHOTS = [
    ("Front: Discord - '#general'\nFocus: field 'Message #general'\nButtons: Send, Gift, Emoji", "OK"),
    ("Front: Google Chrome - 'Account overview'\nHeadings: Checking balance $2,310.55, Recent transactions, Transfer",
     "PRIVATE-financial"),
    ("Front: Terminal - 'zsh'\nSelected: 'git status'", "OK"),
    ("Front: Preview - 'passport_scan.pdf'", "PRIVATE-identity"),
    ("Front: Notes - 'Groceries'\nSelected: 'milk, eggs, bread'", "OK"),
    ("Front: Google Chrome - 'Visit summary'\nHeadings: Diagnosis, Prescriptions, Blood pressure", "PRIVATE-medical"),
    ("Front: Spotify - 'Spotify'\nButtons: Play, Next, Shuffle", "OK"),
    ("Front: Mail - 'Inbox'\nHeadings: Team lunch Friday, Your order has shipped", "OK"),
]


def redact(text):
    for pat, rep in SECRETS:
        text = pat.sub(rep, text)
    return text


def _key(app, title):
    return f"{app}|{title}"[:200]


def rule_verdict(app, title, url=""):
    """'financial' etc. if the app / window / address is private by rule, else None."""
    if (app or "").lower() in PRIVATE_APPS:
        return "credentials"
    m = PRIVATE_SITES.search(f"{title} {url}")
    if m:
        w = m.group(0).lower()
        if re.search(r"mychart|patient|health|lab results|kaiser", w):
            return "medical"
        if re.search(r"sign in|log ?in|password|verification|two-factor|2fa", w):
            return "credentials"
        if re.search(r"private|incognito", w):
            return "private browsing"
        return "financial"
    return None


def load_state():
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}


def blocked(app, title):
    """For macctl: was this window judged private (by the app's rules or its local model)? -> reason or None."""
    v = rule_verdict(app, title)
    if v:
        return v
    return (load_state().get("verdicts") or {}).get(_key(app, title))


class Guard:
    """The app's side: judges each window once (rules, then the local model) and filters the screen summary."""

    def __init__(self, router):
        self.router, self.cache, self.lock = router, {}, threading.Lock()
        self.stats = {"windows": 0, "private": 0, "model_ms": []}
        st = load_state()
        self.cache = {k: (v or "ok") for k, v in (st.get("verdicts") or {}).items()}

    def _save(self):
        try:
            STATE.parent.mkdir(parents=True, exist_ok=True)
            STATE.write_text(json.dumps({"updated": time.strftime("%Y-%m-%d %H:%M"),
                                         "verdicts": {k: (None if v == "ok" else v) for k, v in
                                                      list(self.cache.items())[-300:]}}))
        except OSError:
            pass

    def verdict(self, app, title, summary, url="", timeout=None):
        """'ok' or a reason ('financial' ...). Rules first; else the local model, once per window (cached)."""
        k = _key(app, title)
        v = rule_verdict(app, title, url)
        if v is None and k in self.cache:
            return self.cache[k]
        if v is None:
            if self.router is None or not self.router.usable:
                v = "ok"  # (no local model: the rules and redaction still apply)
            else:
                t0 = time.monotonic()
                try:
                    out = self.router.classify(JUDGE, summary[:900], shots=SHOTS)
                except Exception:
                    out = "OK"
                self.stats["model_ms"].append(round((time.monotonic() - t0) * 1000))
                m = re.search(r"PRIVATE-?(\w+)", out or "", re.I)
                v = m.group(1).lower() if m else "ok"
        with self.lock:
            self.stats["windows"] += k not in self.cache
            self.stats["private"] += (v != "ok") and k not in self.cache
            self.cache[k] = v
        self._save()
        return v

    def filter(self, summary):
        """The screen summary -> what Claude may see: the whole thing (secrets blanked), or one line."""
        if not summary:
            return summary
        first = summary.splitlines()[0]
        m = re.match(r"Front: (.+?)(?: - ['\"](.*)['\"])?(?: \(no window\))?$", first)
        app, title = (m.group(1), m.group(2) or "") if m else ("", "")
        url = next((l[5:].strip() for l in summary.splitlines() if l.startswith("Page:")), "")
        v = self.verdict(app, title, summary, url)
        if v != "ok":
            return f"Front: {app} (private: {v}, not shared; ask the user if you need something from it)"
        return redact(summary)
