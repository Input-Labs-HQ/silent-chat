#!/usr/bin/env python
"""Practice drills: mouth prompted phrases so the lip-reader can learn your weak spots.

Every clip is saved with its exact text (status "confirmed"), which makes it perfect training data for
train_personal.py. Numbers and spelling are where lip-reading is weakest ('-teen' vs '-ty' looks the same,
'twenty-three' collapses to 'three', single letters are unreadable), so the drills focus there:

  numbers   numbers in everyday phrases, said naturally ("twenty three gauge wire") and digit by digit
            ("two two zero volts"), plus teen/ty contrasts ("thirteen not thirty")
  spelling  spelling with the NATO alphabet ("Nissan: November India Sierra Sierra Alpha November"), including
            the names and rare words in your personal vocabulary
  mixed     some of each (default)

  .venv/bin/python practice.py               # 20 mixed prompts
  .venv/bin/python practice.py numbers -n 30

In the camera window: hold SPACE while you mouth the phrase (or tap to start / tap to stop), ESC = retake,
Q = quit. Run it from the macOS Terminal app, with the silent chat closed. At the end it retrains the lip-reader if
there's enough new data (Ctrl+C skips).
"""
import argparse
import random
from datetime import datetime

import vsr
from learning import DATA, Labels, PersonalMemory, is_common
from run_test import number_words

NATO = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", """Alpha Bravo Charlie Delta Echo Foxtrot Golf Hotel India Juliett Kilo
Lima Mike November Oscar Papa Quebec Romeo Sierra Tango Uniform Victor Whiskey X-ray Yankee Zulu""".split()))
DIGITS = "zero one two three four five six seven eight nine".split()
SPELL_WORDS = ["Nissan", "Canberra", "Sarah", "Tesla", "gauge", "Ohm", "Zoom", "Quinn", "Javier", "Bluetooth",
               "AirPods", "Kyoto", "Felix", "Wyoming", "Hawaii", "Pixel", "Ubuntu", "Xbox"]


def digits(n):
    return " ".join(DIGITS[int(d)] for d in str(n))


def number_prompts(rng):
    n2, n3 = rng.randint(10, 99), rng.randint(100, 999)
    teen = rng.randint(13, 19)
    hour, minute = rng.randint(1, 12), rng.choice([15, 20, 30, 40, 45, 50])
    return [
        f"{number_words(rng.choice([10, 12, 14, 16, 18, 20, 22, 23, 24, 26, 28]))} gauge wire",
        f"{digits(rng.choice([12, 14, 16, 18, 22, 23, 24]))} gauge wire",
        f"set a timer for {number_words(rng.randint(2, 59))} minutes",
        f"it costs {number_words(n2)} dollars",
        f"{number_words(teen)} not {number_words((teen - 10) * 10)}",
        f"{number_words((teen - 10) * 10)} not {number_words(teen)}",
        f"call me at {number_words(hour)} {number_words(minute)}",
        f"{digits(rng.choice([110, 120, 220, 230, 240]))} volts",
        f"room {digits(n3)}",
        f"my code is {digits(rng.randint(1000, 9999))}",
        f"turn it up to {number_words(rng.randint(1, 100))} percent",
        f"page {number_words(n3)}",
        f"about {number_words(rng.choice([2, 3, 5, 8, 9, 10, 11]))} hours ago",
    ]


def spelling_prompts(rng, vocabulary):
    words = [w for w in vocabulary if w.isalpha() and not is_common(w)] + SPELL_WORDS
    out = []
    for w in rng.sample(words, min(len(words), 8)):
        out.append(f"{w}: " + " ".join(NATO[c] for c in w.upper() if c in NATO))
    out.append(" ".join(NATO[c] for c in rng.sample(sorted(NATO), 4)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drill", nargs="?", default="mixed", choices=["numbers", "spelling", "mixed"])
    ap.add_argument("-n", type=int, default=20, help="number of prompts")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-train", action="store_true")
    args = ap.parse_args()

    rng = random.Random()
    vocab = PersonalMemory().data["vocabulary"]
    pool = {"numbers": number_prompts(rng) + number_prompts(rng),
            "spelling": spelling_prompts(rng, vocab) + spelling_prompts(rng, vocab)}
    if args.drill == "mixed":
        prompts = rng.sample(pool["numbers"], min(len(pool["numbers"]), args.n * 2 // 3))
        prompts += rng.sample(pool["spelling"], min(len(pool["spelling"]), args.n - len(prompts)))
        rng.shuffle(prompts)
    else:
        prompts = rng.sample(pool[args.drill], min(len(pool[args.drill]), args.n))
    prompts = list(dict.fromkeys(prompts))[:args.n]

    labels = Labels()
    rec, checker, saved = vsr.Recorder(args.camera), vsr.ClipChecker(), 0
    try:
        tips = {"numbers": "Mouth numbers clearly; say them exactly as written (digit by digit where shown).",
                "spelling": "Spell with the NATO words as written, one clear word at a time.",
                "mixed": "Say each phrase exactly as written, clearly, at a normal pace."}[args.drill]
        if not rec.intro([f"Practice: {args.drill} ({len(prompts)} phrases)", "Silently mouth each phrase.", tips,
                          "Hold SPACE while you mouth it, release when done."]):
            return
        for i, text in enumerate(prompts, 1):
            clip = DATA / "clips" / f"practice_{datetime.now():%Y%m%d_%H%M%S_%f}.mp4"
            if rec.take(clip, [f"Practice {i} of {len(prompts)}", tips], text, max_seconds=20, check=checker) is None:
                break
            labels.add(clip, text, "confirmed", f"practice_{args.drill}", [])
            saved += 1
    finally:
        rec.close()
        checker.close()
    print(f"Saved {saved} practice clip(s); {len(Labels().trainable())} clips are now usable for training.")
    if saved and not args.no_train:
        import train_personal
        try:
            train_personal.maybe_train()
        except KeyboardInterrupt:
            print("\nTraining skipped.")


if __name__ == "__main__":
    main()
