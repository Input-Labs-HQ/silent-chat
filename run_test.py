#!/usr/bin/env python
"""Lip-reading test: record the sentences in sentences.txt, transcribe them, and score WER.

  python run_test.py                            # all three conditions in one session (recommended)
  python run_test.py --condition normal_muted   # speak out loud normally (only video is recorded)
  python run_test.py --condition mouthed        # silently mouth the words with clear lip movement
  python run_test.py --condition minimal        # your natural silent style, lips barely moving

All clips are recorded first, each checked on the spot (face tracked, mouth croppable) and retaken if not;
then everything is transcribed and scored. Clips go to recordings/<condition>/NN.mp4 and are reused, so
you can quit (Q) and resume later.
  --redo 3,7      re-record just those sentences
  --fresh         re-record every sentence
  --no-record     only score the clips already recorded
Results: a table on screen, results/<condition>.csv, and one line per condition appended to results/summary.csv.
"""
import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import vsr

ROOT = Path(__file__).resolve().parent
CONDITIONS = {  # name: (title, instruction)
    "normal_muted": ("NORMAL SPEECH", "Say each sentence OUT LOUD, normally, at your usual pace."),
    "mouthed": ("SILENT, CLEAR LIPS", "Make NO sound. Mouth each sentence with clear, normal lip movement."),
    "minimal": ("SILENT, MINIMAL MOVEMENT", "Make NO sound. Your natural silent-talking style: lips barely moving."),
}


ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n):
    """Spell out 0-9999. 3-4 digit clock-like numbers (730, 1015) read as times: "seven thirty"."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    hours, minutes = divmod(n, 100)
    if 1 <= hours <= 12 and 10 <= minutes < 60:  # the model writes "7:30" as "730"
        return f"{number_words(hours)} {number_words(minutes)}"
    if n < 1000:
        return f"{ONES[n // 100]} hundred" + ("" if n % 100 == 0 else " " + number_words(n % 100))
    if n < 10000:
        return f"{number_words(n // 1000)} thousand" + ("" if n % 1000 == 0 else " " + number_words(n % 1000))
    return str(n)


def normalize(text):
    """Lower-case, strip punctuation, and spell out numbers so "730"/"seven thirty" score the same.

    The model writes "$50" as "<unk> 50" ($ is not in its vocabulary), so "unk <number>" becomes
    "<number> dollars".
    """
    text = re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("’", "'"))
    words = [w.strip("'") for w in text.split() if w.strip("'")]
    out, i = [], 0
    while i < len(words):
        w = words[i]
        if w == "unk" and i + 1 < len(words) and words[i + 1].isdigit():
            out += [number_words(int(words[i + 1])), "dollars"]
            i += 2
            continue
        out.append(number_words(int(w)) if w.isdigit() else w)
        i += 1
    return " ".join(out)


def word_errors(ref, hyp):
    """Word-level Levenshtein alignment -> (substitutions, deletions, insertions)."""
    r, h = ref.split(), hyp.split()
    # d[i][j] = (cost, subs, dels, ins) for r[:i] vs h[:j]
    d = [[(j, 0, 0, j) for j in range(len(h) + 1)]]
    for i in range(1, len(r) + 1):
        row = [(i, 0, i, 0)]
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                best = d[i - 1][j - 1]
            else:
                s, dl, ins = d[i - 1][j - 1], d[i - 1][j], row[j - 1]
                best = min(
                    (s[0] + 1, s[1] + 1, s[2], s[3]),
                    (dl[0] + 1, dl[1], dl[2] + 1, dl[3]),
                    (ins[0] + 1, ins[1], ins[2], ins[3] + 1),
                )
            row.append(best)
        d.append(row)
    _, subs, dels, ins = d[-1][-1]
    return subs, dels, ins


def parse_list(spec):
    return sorted({int(x) for x in spec.split(",") if x.strip()}) if spec else []


def record(conditions, sentences, clips, todo, args):
    """Record every clip in todo[cond]; returns False if the user quit."""
    total = sum(len(t) for t in todo.values())
    print(f"Recording {total} clip(s). In the camera window: hold SPACE while you say each sentence "
          "(or tap to start / tap to stop), ESC = retake, Q = quit (finished clips are kept).", flush=True)
    rec, checker = vsr.Recorder(args.camera), vsr.ClipChecker()
    try:
        for k, cond in enumerate(conditions, 1):
            if not todo[cond]:
                continue
            title, instruction = CONDITIONS[cond]
            part = f"Part {k} of {len(conditions)}: " if len(conditions) > 1 else ""
            if not rec.intro([f"{part}{title}", instruction, "",
                              f"{len(todo[cond])} sentences. Each one appears here in yellow.",
                              "Hold SPACE while you say it, release when done.",
                              "Keep your face inside the box, close to the camera."]):
                return False
            for i in todo[cond]:
                header = [f"{part}{title}  -  sentence {i} of {len(sentences)}", instruction]
                if rec.take(clips[cond][i - 1], header, sentences[i - 1], args.max_seconds, check=checker) is None:
                    return False
        return True
    finally:
        rec.close()
        checker.close()


def score(cond, sentences, clips, reader, args):
    available = [(i, s, c) for i, (s, c) in enumerate(zip(sentences, clips), 1) if c.exists()]
    if not available:
        print(f"\n[{cond}] no clips recorded yet.")
        return None
    print(f"\n=== {cond}: transcribing {len(available)} clip(s) on {args.device} (beam {args.beam_size}) ===", flush=True)
    rows, fallbacks = [], set()
    for i, sentence, clip in available:
        error = ""
        try:
            r = reader.transcribe(clip)
            raw, clip_s, model_s = r["text"], r["clip_seconds"], r["model_seconds"]
            fallbacks.update(r["mps_fallbacks"])
        except Exception as e:  # e.g. no face found; score as all words missed
            raw, clip_s, model_s, error = "", 0.0, 0.0, f"{type(e).__name__}: {e}"
        ref, hyp = normalize(sentence), normalize(raw)
        subs, dels, ins = word_errors(ref, hyp)
        n = len(ref.split())
        rows.append({
            "n": i, "reference": ref, "prediction": hyp, "ref_words": n,
            "substitutions": subs, "deletions": dels, "insertions": ins,
            "wer": round((subs + dels + ins) / n, 4),
            "clip_seconds": round(clip_s, 2), "model_seconds": round(model_s, 2),
            "clip": str(clip.relative_to(ROOT)), "error": error,
        })
        print(f"{i:2d}. WER {rows[-1]['wer']:6.1%}  REF: {ref}")
        print(f"                 HYP: {hyp or '(nothing)'}{'   [' + error + ']' if error else ''}", flush=True)

    words = sum(r["ref_words"] for r in rows)
    errors = sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in rows)
    correct = sum(r["ref_words"] - r["substitutions"] - r["deletions"] for r in rows)
    wer = errors / words
    exact = sum(r["wer"] == 0 for r in rows)
    print(f"--- {cond}: {len(rows)} sentences, {words} words ---")
    print(f"WER:             {wer:.1%}  ({errors} errors = substitutions + deletions + insertions)")
    print(f"Word accuracy:   {1 - wer:.1%}  (1 - WER)")
    print(f"Words correct:   {correct}/{words} ({correct / words:.1%})")
    print(f"Exact sentences: {exact}/{len(rows)}")
    for w in sorted(fallbacks):
        print(f"MPS fallback to CPU: {w}")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"{cond}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = out_dir / "summary.csv"
    new = not summary.exists()
    with open(summary, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["time", "condition", "device", "beam", "sentences", "words", "errors", "wer",
                             "word_accuracy", "words_correct", "exact_sentences"])
        writer.writerow([f"{datetime.now():%Y-%m-%d %H:%M}", cond, args.device, args.beam_size,
                         len(rows), words, errors, round(wer, 4), round(1 - wer, 4), correct, exact])
    return {"sentences": len(rows), "words": words, "wer": wer, "correct": correct, "exact": exact}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", default="all", choices=["all", *CONDITIONS])
    ap.add_argument("--sentences", type=Path, default=ROOT / "sentences.txt")
    ap.add_argument("--redo", type=str, default="", help="comma-separated sentence numbers to re-record")
    ap.add_argument("--fresh", action="store_true", help="re-record all sentences")
    ap.add_argument("--no-record", action="store_true", help="don't open the camera; score existing clips")
    ap.add_argument("--device", choices=vsr.DEVICES, default=vsr.DEFAULT_DEVICE)
    ap.add_argument("--beam-size", type=int, default=40)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=15.0)
    args = ap.parse_args()

    lines = [l.strip() for l in args.sentences.read_text(encoding="utf8").splitlines()]
    sentences = [l for l in lines if l and not l.startswith("#")]
    conditions = list(CONDITIONS) if args.condition == "all" else [args.condition]
    clips = {c: [ROOT / "recordings" / c / f"{i:02d}.mp4" for i in range(1, len(sentences) + 1)]
             for c in conditions}

    todo = {}
    for c in conditions:
        if args.no_record:
            todo[c] = []
        elif args.fresh:
            todo[c] = list(range(1, len(sentences) + 1))
        else:
            missing = {i for i, clip in enumerate(clips[c], 1) if not clip.exists()}
            todo[c] = sorted(i for i in set(parse_list(args.redo)) | missing if 1 <= i <= len(sentences))
    if any(todo.values()) and not record(conditions, sentences, clips, todo, args):
        print("Quit recording. Finished clips are kept; run the same command again to continue.")

    reader = vsr.LipReader(args.device, beam_size=args.beam_size)
    results = {c: score(c, sentences, clips[c], reader, args) for c in conditions}
    reader.close()

    done = {c: r for c, r in results.items() if r}
    if len(done) > 1:
        print("\n=== Summary ===")
        print(f"{'condition':14s} {'sentences':>9s} {'WER':>7s} {'word acc':>9s} {'words right':>12s} {'exact':>6s}")
        for c, r in done.items():
            print(f"{c:14s} {r['sentences']:9d} {r['wer']:7.1%} {1 - r['wer']:9.1%} "
                  f"{r['correct']:>5d}/{r['words']:<6d} {r['exact']:>3d}/{r['sentences']}")
    if done:
        print("\nSaved results/<condition>.csv and appended to results/summary.csv")


if __name__ == "__main__":
    main()
