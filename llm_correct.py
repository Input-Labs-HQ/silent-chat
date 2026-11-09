#!/usr/bin/env python
"""LLM post-correction experiment on the recorded lip-reading test clips.

For every clip, the lip-reader's top-10 hypotheses go to Claude, which answers with the sentence the user most
likely said. Claude never sees the reference sentences; they are only used for scoring afterwards.

Three scores per condition:
  top-1    the lip-reader's best guess (the baseline from run_test.py)
  LLM      Claude's corrected sentence
  oracle   the best of the top-10 hypotheses, picked using the answer key: an upper bound on what
           choosing from the list could achieve (not achievable in practice)

  python llm_correct.py                    # all three conditions
  python llm_correct.py --condition mouthed

Claude is called through the locally logged-in Claude Code CLI (`claude -p`) because no API key is set up;
MCP servers, skills and settings are switched off so each call is only ~1K tokens of context.
Caches: results/nbest_<condition>.json and results/llm_<condition>.json (delete to recompute).
"""
import argparse
import concurrent.futures as cf
import csv
import json
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from run_test import CONDITIONS, normalize, word_errors

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODEL = "claude-opus-5-5"
SYSTEM = (
    "You correct the output of a lip-reading (visual speech recognition) model. The user said a short request or "
    "question to an AI assistant, either out loud or silently, and a camera-only model transcribed it. "
    "The lip-reader only sees the lips, so it confuses words that look alike (p/b/m, f/v, t/d/n/s/z/th, k/g, "
    "ch/j/sh, similar vowels) and often gets word boundaries wrong. Its decoder was trained on TED talks, so when "
    "unsure it drifts into fluent but unrelated TED-style phrases. You get its top hypotheses, best first, with "
    "log-probability scores (closer to zero = more confident). Work out what the user most likely said: use the "
    "words and lip shapes the hypotheses agree on, undo look-alike confusions, and prefer a natural thing to say to "
    "an assistant. Reply with only that sentence, no quotes or explanation."
)


def get_nbest(cond, n_sentences, k, device, reader_box):
    cache = RESULTS / f"nbest_{cond}.json"
    if cache.exists():
        return {int(i): h for i, h in json.loads(cache.read_text()).items()}
    if reader_box[0] is None:
        import vsr
        print(f"Loading lip-reader on {device} ...", flush=True)
        reader_box[0] = vsr.LipReader(device)
    import vsr
    reader, out = reader_box[0], {}
    for i in range(1, n_sentences + 1):
        clip = ROOT / "recordings" / cond / f"{i:02d}.mp4"
        if not clip.exists():
            continue
        crops, _ = reader.mouth_crops(vsr.load_video(clip))
        out[i], _ = reader.decode(crops, nbest=k)
        print(f"  {cond} {i:2d}: {out[i][0][0]}", flush=True)
    cache.write_text(json.dumps(out, indent=1))
    return out


# MCP servers, skills/plugins and user settings switched off: ~1K tokens of context per call instead of ~400K.
CLI_FLAGS = ["--output-format", "json", "--no-session-persistence", "--tools", "", "--strict-mcp-config",
             "--disable-slash-commands", "--setting-sources", "project"]
_WORKDIR = []


def run_claude(prompt, system, model=MODEL, retries=3):
    """One Claude call through the locally logged-in `claude` CLI. Returns {"text", "seconds", "cost_usd"}."""
    if not _WORKDIR:
        _WORKDIR.append(tempfile.mkdtemp(prefix="claude_cli_"))  # neutral cwd: no project files to pick up
    cmd = ["claude", "-p", prompt, "--model", model, *CLI_FLAGS, "--system-prompt", system]
    err = None
    for attempt in range(retries):
        t = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=_WORKDIR[0], timeout=300)
        try:
            d = json.loads(r.stdout)
            if d.get("is_error") or not d.get("result"):
                raise ValueError(d.get("result") or r.stderr[:200])
            return {"text": d["result"].strip(), "seconds": time.perf_counter() - t,
                    "cost_usd": d.get("total_cost_usd", 0.0)}
        except (json.JSONDecodeError, ValueError) as e:
            err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"claude CLI failed: {err}")


def format_nbest(hyps):
    return "\n".join(f"{j}. [{score:.1f}] {text}" for j, (text, score) in enumerate(hyps, 1))


def ask_claude(hyps):
    prompt = "Lip-reader hypotheses, best first:\n" + format_nbest(hyps) + "\n\nWhat did the user most likely say?"
    r = run_claude(prompt, SYSTEM)
    r["text"] = r["text"].splitlines()[0].strip().strip('"')
    return r


def get_llm(cond, nbest, workers):
    cache = RESULTS / f"llm_{cond}.json"
    done = {int(i): v for i, v in json.loads(cache.read_text()).items()} if cache.exists() else {}
    todo = [i for i in nbest if i not in done]
    if todo:
        print(f"Asking {MODEL} about {len(todo)} {cond} clip(s) ...", flush=True)
        with cf.ThreadPoolExecutor(workers) as pool:
            futures = {pool.submit(ask_claude, nbest[i]): i for i in todo}
            for f in cf.as_completed(futures):
                done[futures[f]] = f.result()
                cache.write_text(json.dumps(done, indent=1))  # save as we go
    return done


def wer_parts(ref, hyp):
    s, d, i = word_errors(ref, hyp)
    return s + d + i, len(ref.split()) - s - d  # errors, correct words


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", default="all", choices=["all", *CONDITIONS])
    ap.add_argument("--sentences", type=Path, default=ROOT / "sentences.txt")
    ap.add_argument("--nbest", type=int, default=10)
    ap.add_argument("--device", default="hybrid")
    ap.add_argument("--workers", type=int, default=6, help="parallel Claude calls")
    args = ap.parse_args()

    lines = [l.strip() for l in args.sentences.read_text(encoding="utf8").splitlines()]
    sentences = [l for l in lines if l and not l.startswith("#")]
    conditions = list(CONDITIONS) if args.condition == "all" else [args.condition]
    RESULTS.mkdir(exist_ok=True)

    reader_box, summary = [None], {}
    for cond in conditions:
        nbest = get_nbest(cond, len(sentences), args.nbest, args.device, reader_box)
        if not nbest:
            print(f"[{cond}] no clips.")
            continue
        llm = get_llm(cond, nbest, args.workers)
        rows, tot = [], {"words": 0, "top1": [0, 0], "llm": [0, 0], "oracle": [0, 0]}
        print(f"\n=== {cond} ===")
        for i in sorted(nbest):
            ref = normalize(sentences[i - 1])
            top1 = normalize(nbest[i][0][0])
            fixed = normalize(llm[i]["text"])
            oracle = min((normalize(t) for t, _ in nbest[i]), key=lambda h: word_errors(ref, h))
            n = len(ref.split())
            scores = {"top1": wer_parts(ref, top1), "llm": wer_parts(ref, fixed), "oracle": wer_parts(ref, oracle)}
            tot["words"] += n
            for key, (err, ok) in scores.items():
                tot[key][0] += err
                tot[key][1] += ok
            mark = "+" if scores["llm"][0] < scores["top1"][0] else "-" if scores["llm"][0] > scores["top1"][0] else " "
            print(f"{i:2d}. REF : {ref}")
            print(f"    TOP1: {top1}  ({scores['top1'][0] / n:.0%})")
            print(f"  {mark} LLM : {fixed}  ({scores['llm'][0] / n:.0%})", flush=True)
            rows.append({"n": i, "reference": ref, "top1": top1, "llm": fixed, "oracle_pick": oracle,
                         "wer_top1": round(scores["top1"][0] / n, 4), "wer_llm": round(scores["llm"][0] / n, 4),
                         "wer_oracle": round(scores["oracle"][0] / n, 4),
                         "llm_seconds": round(llm[i]["seconds"], 1)})
        with open(RESULTS / f"llm_{cond}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        summary[cond] = tot | {"sentences": len(rows),
                               "seconds": sum(v["seconds"] for v in llm.values()) / len(llm),
                               "cost": sum(v["cost_usd"] for v in llm.values())}

    if reader_box[0] is not None:
        reader_box[0].close()
    if not summary:
        return
    print("\n=== Summary: word accuracy (1 - WER) / words correct ===")
    print(f"{'condition':14s} {'lip-reader top-1':>22s} {'+ LLM correction':>22s} {'oracle (best of 10)':>22s}")
    with open(RESULTS / "llm_summary.csv", "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["time", "condition", "model", "words", "acc_top1", "acc_llm", "acc_oracle",
                        "correct_top1", "correct_llm", "correct_oracle", "mean_llm_seconds", "llm_cost_usd"])
        for cond, t in summary.items():
            n = t["words"]
            cell = lambda k: f"{1 - t[k][0] / n:6.1%}  {t[k][1]:3d}/{n}"
            print(f"{cond:14s} {cell('top1'):>22s} {cell('llm'):>22s} {cell('oracle'):>22s}")
            w.writerow([f"{datetime.now():%Y-%m-%d %H:%M}", cond, MODEL, n,
                        *(round(1 - t[k][0] / n, 4) for k in ("top1", "llm", "oracle")),
                        *(t[k][1] for k in ("top1", "llm", "oracle")), round(t["seconds"], 1), round(t["cost"], 3)])
    secs = sum(t["seconds"] for t in summary.values()) / len(summary)
    cost = sum(t["cost"] for t in summary.values())
    print(f"\nClaude ({MODEL}): {secs:.1f} s per correction on average, ~${cost:.2f} total at API prices.")
    print("Saved results/llm_<condition>.csv and appended to results/llm_summary.csv")


if __name__ == "__main__":
    main()
