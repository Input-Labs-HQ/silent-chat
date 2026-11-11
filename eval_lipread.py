"""What the lip-reader has learned from your clips, and whether more can be squeezed out of the same clips.

  survey      how many clips carry which kind of label (data/silent_chat/labels.csv)
  re-ranking  would choosing among the lip-reader's 10 guesses by your own way of putting things help? Each clip's
              logged guesses are re-ranked with a word-pair model of everything you had said *before* that clip.
              The weights are tuned on the earlier half of your clips and scored on the later half.
  --models    word errors of the original model and your fine-tuned one on clips neither has trained on (your
              25 set sentences + the session clips held out by hash + clips newer than the last training), and
              the same for other decoding settings. Loads the models: ~3 minutes, best on mains power.

Measured 2026-11-10, 227 usable clips (207 of them "Claude's reading, and you carried on"; 12 corrected or picked):
  fine-tuning  original -> personal: 80.7 % -> 47.2 % word errors on the 32 held-out session clips (254 words),
               58.6 % -> 46.3 % on the 25 set sentences (162 words). This is the adaptation that works.
  decoding     CTC weight 0 / 0.1 / 0.3: 45.7 / 46.3 / 48.1 % on the set sentences, 47.2 / 47.2 / 47.6 % held out;
               beam 40 reads exactly like beam 10 at 2.5x the time. Left as it is (0.1, beam 10).
  re-ranking   44.6 % -> 45.7-46.5 % on the later half (the right sentence is among the 10 guesses 49 % of the
               time, first 33 %). No gain, so it is not in the app: a few hundred sentences is too little text to
               out-guess the lip-reader's own decoder, and Claude, which sees your vocabulary and corrections,
               already does this job with far more language behind it.

  .venv/bin/python eval_lipread.py [--models]
"""
import argparse
import collections
import itertools
import json
import math
import re

from learning import ROOT, Labels
from run_test import normalize, word_errors


def label_text(text):  # (as in train_personal.py)
    return " ".join(re.sub(r"[^A-Z0-9' ]+", " ", text.upper().replace("’", "'")).split())


class Phrasing:
    """Word and word-pair counts of what you've said so far."""

    def __init__(self):
        self.one, self.two, self.whole, self.n = collections.Counter(), collections.Counter(), collections.Counter(), 0

    def add(self, text):
        w = text.split()
        self.whole[text] += 1
        self.one.update(w)
        self.two.update(zip(["<s>"] + w, w))
        self.n += len(w)

    def fit(self, text, vocab=5000.0):
        """Mean over words of log(P(word | previous word) x vocabulary): above 0 for your own words and pairs."""
        w = text.split()
        if not w or not self.n:
            return 0.0
        total = 0.0
        for a, b in zip(["<s>"] + w, w):
            before = self.one[a] if a != "<s>" else sum(self.whole.values())
            p = 0.5 * (self.two[(a, b)] / before if before else 0.0) + 0.3 * self.one[b] / self.n + 0.2 / vocab
            total += math.log(p * vocab)
        return total / len(w)

    def pick(self, hyps, alpha, beta):
        return max((s + alpha * self.fit(h) + beta * math.log(1 + self.whole[h]), h) for h, s in hyps)[1]


def reranking(data, which, alpha, beta):
    """-> (word errors as it ran, word errors re-ranked, words, exact as it ran, exact re-ranked) on clips `which`."""
    lm, e0, e1, n, x0, x1 = Phrasing(), 0, 0, 0, 0, 0
    for i, (ref, hyps) in enumerate(data):
        if i in which:
            pick = lm.pick(hyps, alpha, beta)
            e0, e1 = e0 + sum(word_errors(ref, hyps[0][0])), e1 + sum(word_errors(ref, pick))
            n, x0, x1 = n + len(ref.split()), x0 + (hyps[0][0] == ref), x1 + (pick == ref)
        lm.add(ref)  # (learned only after it was read, as it would be live)
    return e0, e1, n, x0, x1


def models():
    import train_personal as tp
    import vsr
    items = tp.session_items()
    when = {str(ROOT / r["clip"]): r["time"][:16] for r in Labels().trainable()}
    last = json.loads((tp.MODELS / "personal.json").read_text())["current"]["date"]
    sets = [("25 set sentences", tp.benchmark_items()),
            ("held-out session clips", [it for it in items if tp.is_heldout(it[0])]),
            (f"session clips since the last training ({last})",
             [it for it in items if when[str(it[0])] > last and not tp.is_heldout(it[0])])]
    cache = vsr.LipReader("hybrid", ckpt=vsr.BASE_CKPT, beam_size=10)
    print("\nWord errors on clips the model never trained on:")
    for name, ckpt, ctc, beam in (("original", vsr.BASE_CKPT, 0.1, 10), ("personal", vsr.PERSONAL_CKPT, 0.1, 10),
                                  ("personal", vsr.PERSONAL_CKPT, 0.0, 10), ("personal", vsr.PERSONAL_CKPT, 0.3, 10),
                                  ("personal", vsr.PERSONAL_CKPT, 0.1, 40)):
        reader = vsr.LipReader("hybrid", ckpt=ckpt, beam_size=beam, ctc_weight=ctc)
        scores = [tp.fmt(*tp.evaluate(reader, its, cache)) for _, its in sets]
        print(f"  {name}, CTC weight {ctc}, beam {beam}:")
        for (label, its), sc in zip(sets, scores):
            print(f"      {label:52s} {sc}  ({len(its)} clips)")
        reader.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", action="store_true", help="also score the original and personal models (~3 min)")
    args = ap.parse_args()
    labels = Labels()
    usable = sorted(labels.trainable(), key=lambda r: r["time"])
    print(f"{len(labels.rows)} clips logged, {len(usable)} usable for training:")
    for k, v in collections.Counter(r["status"] for r in labels.rows).most_common():
        print(f"  {k:11s} {v:4d}")
    data = []
    for r in usable:
        hyps = [(normalize(h), float(s)) for h, s in json.loads(r["nbest"] or "[]")]
        if hyps:
            data.append((normalize(label_text(r["text"])), hyps))
    first = sum(h[0][0] == ref for ref, h in data)
    among = sum(any(x == ref for x, _ in h) for ref, h in data)
    print(f"\nThe right sentence is the lip-reader's first guess in {first / len(data):.0%} of them, and among its 10 in "
          f"{among / len(data):.0%}.")
    cut = int(len(data) * 0.55)
    tune, test = set(range(20, cut)), set(range(cut, len(data)))
    grid = list(itertools.product((0, 0.1, 0.2, 0.35, 0.5, 1.0), (0, 0.5, 1.0, 2.0)))
    alpha, beta = min(grid, key=lambda ab: reranking(data, tune, *ab)[1])
    print(f"\nRe-ranking the 10 guesses by your own phrasing (weights tuned on clips 20-{cut}: word pairs {alpha}, "
          f"whole sentences {beta}):")
    for label, (a, b) in (("as tuned", (alpha, beta)), ("word pairs only", (0.5, 0)), ("whole sentences only", (0, 2.0))):
        e0, e1, n, x0, x1 = reranking(data, test, a, b)
        print(f"  {label:21s} the later {len(test)} clips: {e0 / n:.1%} -> {e1 / n:.1%} word errors, "
              f"{x0} -> {x1} sentences exactly right")
    if args.models:
        models()


if __name__ == "__main__":
    main()
