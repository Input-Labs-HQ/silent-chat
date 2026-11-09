"""What silent chat learns from your sessions.

labels.csv     one row per captured clip: the text, and how sure we are of it (status):
                 confirmed   picked or typed by you (--confirm mode)
                 fixed       corrected by you ("no, I said ...", or /fix in Terminal)
                 ok          Claude's reading, and you carried on without correcting it
                 heard       Claude's reading, not yet confirmed (e.g. the last turn of a session)
                 audio       said out loud with the MIC on; labelled by speech recognition (reliable)
                 unlabelled  part of a multi-part message, or Claude couldn't tell
               confirmed / fixed / ok clips are used to fine-tune the lip-reader (train_personal.py).
personal.json  words and names you use, and past misreadings (lip-reader guess -> what you meant). Claude sees
               this every turn, so a correction helps immediately, before any retraining.
"""
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "silent_chat"
FIELDS = ["time", "clip", "text", "status", "source", "lipreader_top1", "options", "nbest"]
TRAINABLE = {"confirmed", "fixed", "ok", "audio"}
STOPWORDS = set("""the a an and or but to of in on at for with from by is are was were be been am i you he she it we
they me my your our their his her its this that these those what which who whom how why when where do does did done
not no yes so if then than just like said say says saying can could would should will shall may might must have has
had get got go going gone want wanted need make made let lets about there here very really also too all any some""".split())


def words(text):
    return re.findall(r"[A-Za-z][A-Za-z'\-]*", text or "")


_COMMON = None


def is_common(word):
    """True if the lip-reader has this as a single whole-word unit, i.e. it can already produce it easily.

    Rare words (e.g. NISSAN -> NI S S AN) are spelled from pieces; those are the ones worth remembering.
    """
    global _COMMON
    if _COMMON is None:
        units = Path(__file__).resolve().parent / "third_party/auto_avsr/spm/unigram/unigram5000_units.txt"
        _COMMON = {u.split()[0][1:] for u in units.read_text(encoding="utf8").splitlines() if u.startswith("▁")}
    return word.upper() in _COMMON


class Labels:
    def __init__(self, path=None):
        self.path = Path(path) if path else DATA / "labels.csv"
        self.rows, self.enabled = [], True  # (enabled False: nothing new is recorded; Settings "Save mouth clips")
        if self.path.exists():
            with open(self.path, newline="") as f:
                self.rows = list(csv.DictReader(f))
            for r in self.rows:  # files from before the status column existed
                if not r.get("status"):
                    if r.get("source", "").startswith(("option", "typed")):
                        r["status"] = "confirmed"
                    else:
                        r["status"] = "heard" if r.get("text") else "unlabelled"

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.rows)
        os.replace(tmp, self.path)

    def add(self, clip, text, status, source, hyps, options=None):
        if not self.enabled:
            return
        self.rows.append({
            "time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "clip": str(Path(clip).relative_to(ROOT)), "text": text,
            "status": status, "source": source, "lipreader_top1": hyps[0][0] if hyps else "",
            "options": json.dumps(options or []), "nbest": json.dumps(hyps or []),
        })
        self.save()

    def update(self, clips, only_status=None, **fields):
        keys = {str(Path(c).relative_to(ROOT)) for c in clips}
        for r in self.rows:
            if r["clip"] in keys and (only_status is None or r["status"] == only_status):
                r.update(fields)
        self.save()

    def trainable(self):
        return [r for r in self.rows
                if r["status"] in TRAINABLE and r["text"].strip() and (ROOT / r["clip"]).exists()]


class PersonalMemory:
    def __init__(self, path=None):
        self.path = Path(path) if path else DATA / "personal.json"
        self.data = {"vocabulary": {}, "corrections": []}
        if self.path.exists():
            self.data.update(json.loads(self.path.read_text()))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))

    def add_words(self, new_words):
        vocab, added = self.data["vocabulary"], []
        for w in new_words:
            w = w.strip(" ,.;:!?\"")
            if len(w) < 2:
                continue
            key = next((k for k in vocab if k.lower() == w.lower()), None)
            if key is None:
                vocab[w] = 1
                added.append(w)
            else:
                vocab[key] += 1
        self.save()
        return added

    def learn_correction(self, lipreader, heard, meant):
        """Remember a misreading. New words in `meant` that are names or rare words join the vocabulary."""
        self.data["corrections"] = (self.data["corrections"] + [{
            "lipreader": lipreader, "heard": heard, "meant": meant, "time": f"{datetime.now():%Y-%m-%d %H:%M}"}])[-200:]
        heard_words = {w.lower() for w in words(heard)}
        new = []
        for i, w in enumerate(words(meant)):
            if w.lower() in heard_words or len(w) < 3 or w.lower().split("'")[0] in STOPWORDS:
                continue  # "what's" counts as "what"
            is_name = w[0].isupper() and i > 0
            if is_name or not is_common(w):
                new.append(w)
        self.add_words(new)
        return new

    def learn_from_speech(self, text):
        """Names and rare words from something you said out loud (reliably transcribed) join the vocabulary."""
        new = []
        for sentence in re.split(r"[.!?]+", text):
            for i, w in enumerate(re.findall(r"[A-Za-z][A-Za-z']*", sentence)):
                if len(w) < 3 or w.lower().split("'")[0] in STOPWORDS:
                    continue
                if not is_common(w) or (w[0].isupper() and i > 0):
                    new.append(w)
        return self.add_words(new)

    def prompt_block(self, max_words=60, max_corrections=15):
        vocab, corrections = self.data["vocabulary"], self.data["corrections"]
        if not vocab and not corrections:
            return ""
        lines = ["What you've learned about this user from past sessions:"]
        if vocab:
            top = sorted(vocab, key=lambda w: -vocab[w])[:max_words]
            lines.append("Words and names they use (prefer these when the lip shapes fit): " + ", ".join(top))
        if corrections:
            lines.append("Past misreadings (lip-reader's top guess -> what they actually said):")
            lines += [f'- "{c["lipreader"]}" -> "{c["meant"]}"' for c in corrections[-max_corrections:]]
        return "\n".join(lines)


def prune_vocabulary(path=None):
    """Drop common words that older versions added to the vocabulary (keeps names and rare words)."""
    memory = PersonalMemory(path)
    vocab = memory.data["vocabulary"]
    removed = [w for w in vocab if is_common(w)]  # the lip-reader produces these easily already
    for w in removed:
        del vocab[w]
    memory.save()
    return removed, list(vocab)
