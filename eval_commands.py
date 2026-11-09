"""Measure closed-set lip-reading (instant on-screen commands) on your own labelled clips.

For each clip with a known text, the lip-reader scores a candidate list against the video (no search):
  in-set:      the true text + 39 other texts from your clips -> did it pick the true one?
  out-of-set:  the same 39 others without the true text       -> did it correctly refuse?
A candidate is accepted only if its score is within MARGIN of the free (open-vocabulary) reading's score; this is
what makes "none of these" possible. Compares with the open-vocabulary top-1 and times both.

  .venv/bin/python eval_commands.py [--limit N] [--size 40]
"""
import argparse
import random
import re
import time
from pathlib import Path

import numpy as np

import vsr
from learning import ROOT, TRAINABLE, Labels


def norm(t):
    return " ".join(re.sub(r"[^A-Z0-9' ]+", " ", t.upper()).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--size", type=int, default=40)
    ap.add_argument("--model", choices=["base", "personal"], default="personal",
                    help="personal: only clips it never trained on (held out, or newer than its last training)")
    args = ap.parse_args()
    rows = [r for r in Labels().rows if r["status"] in TRAINABLE and r["text"].strip() and (ROOT / r["clip"]).exists()]
    texts = sorted({norm(r["text"]) for r in rows})  # distractors come from all your clips
    ckpt = vsr.BASE_CKPT if args.model == "base" else vsr.PERSONAL_CKPT
    if args.model == "personal":
        import json
        import train_personal
        kept = [h["date"] for h in json.loads((ROOT / "models" / "personal.json").read_text())["history"] if h["kept"]]
        rows = [r for r in rows if train_personal.is_heldout(r["clip"]) or r["time"][:16] > kept[-1]]
        print(f"personal model (last kept training {kept[-1]}): {len(rows)} clips it never trained on")
    if args.limit:
        rows = rows[:args.limit]
    rng = random.Random(0)
    reader = vsr.LipReader(vsr.DEFAULT_DEVICE, ckpt=ckpt, beam_size=10)
    out = []
    for i, r in enumerate(rows):
        truth = norm(r["text"])
        try:
            crops, _ = reader.mouth_crops(vsr.load_video(ROOT / r["clip"]))
        except Exception as e:
            print(f"skip {r['clip']}: {e}")
            continue
        t0 = time.perf_counter()
        enc = reader.encode(crops)
        t_enc = time.perf_counter() - t0
        t0 = time.perf_counter()
        free_text, free_score = reader.decode_enc(enc, nbest=1)[0]
        t_dec = time.perf_counter() - t0
        others = rng.sample([t for t in texts if t != truth], args.size - 1)
        t0 = time.perf_counter()
        in_set = reader.score(enc, [truth] + others)
        t_score = time.perf_counter() - t0
        out_set = reader.score(enc, others)
        out.append({"truth": truth, "words": len(truth.split()), "free": norm(free_text), "free_score": free_score,
                    "in_best": in_set[0][0], "in_score": in_set[0][1], "in_second": in_set[1][1],
                    "out_score": out_set[0][1], "t_enc": t_enc, "t_dec": t_dec, "t_score": t_score})
        if i % 20 == 0:
            print(f"{i + 1}/{len(rows)}  enc {t_enc * 1000:.0f} ms, decode {t_dec * 1000:.0f} ms, "
                  f"score {args.size} candidates {t_score * 1000:.0f} ms", flush=True)

    def report(name, sel):
        if not sel:
            return
        n = len(sel)
        free_ok = np.mean([s["free"] == s["truth"] for s in sel])
        top1 = np.mean([s["in_best"] == s["truth"] for s in sel])
        print(f"\n== {name}: {n} clips")
        print(f"open-vocabulary top-1 exactly right: {free_ok:.0%}")
        print(f"closed-set top-1 right (out of {args.size} candidates): {top1:.0%}")
        print("margin  in-set: accepted & right | accepted & wrong | refused    out-of-set: wrongly accepted")
        for margin in (0, 2, 4, 6, 8, 12):
            acc = [s for s in sel if s["in_score"] >= s["free_score"] - margin]
            right = sum(s["in_best"] == s["truth"] for s in acc)
            false_acc = np.mean([s["out_score"] >= s["free_score"] - margin for s in sel])
            print(f"  {margin:>4}            {right / n:>5.0%}         |      {(len(acc) - right) / n:>5.0%}      | "
                  f"{1 - len(acc) / n:>5.0%}                     {false_acc:>5.0%}")

    report("all clips", out)
    report("short clips (<= 4 words, command length)", [s for s in out if s["words"] <= 4])
    print(f"\ntiming (median): encoder {np.median([s['t_enc'] for s in out]) * 1000:.0f} ms, open-vocabulary "
          f"decode {np.median([s['t_dec'] for s in out]) * 1000:.0f} ms, closed-set scoring of {args.size} "
          f"candidates {np.median([s['t_score'] for s in out]) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
