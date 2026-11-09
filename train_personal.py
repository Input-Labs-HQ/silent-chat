#!/usr/bin/env python
"""Self-improvement: fine-tune the lip-reader on your own silent-chat clips.

Training data: clips in data/silent_chat/labels.csv whose text you confirmed (picked or typed in --confirm mode),
corrected ("no, I said ..." or /fix), or didn't object to in the next turn.

How it stays safe:
  * Always starts from the original Auto-AVSR checkpoint and trains on everything collected so far, so models
    don't drift over many rounds.
  * Front-end (ResNet) frozen, BatchNorm statistics frozen, small learning rate, a few epochs.
  * The new model is kept only if it makes fewer word errors than the current one on clips it never trained on:
    your 25 'mouthed' test recordings (recordings/mouthed, the benchmark) plus ~15 % of your session clips (held
    out by a fixed hash, so they stay held out across rounds).
Saves models/personal.pth (+ models/personal.json); vsr.py and silent_chat.py then use it automatically.

  .venv/bin/python train_personal.py           # train if there's enough new data (also runs when the chat closes)
  .venv/bin/python train_personal.py --force   # train now
"""
import argparse
import hashlib
import os
import json
import random
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sentencepiece
import torch

import vsr
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from learning import ROOT, Labels
from run_test import normalize, word_errors

MODELS = ROOT / "models"
CACHE = ROOT / "data" / "cache" / "crops"
BENCH_DIR = ROOT / "recordings" / "mouthed"
SENTENCES = ROOT / "sentences.txt"
MIN_CLIPS, MIN_NEW = 30, 15  # train once there are 30 usable clips, then again after every 15 new ones
EVAL_BEAM = 10


# ----------------------------------------------------------------------------- data

def label_text(text):
    text = re.sub(r"[^A-Z0-9' ]+", " ", text.upper().replace("’", "'"))
    return " ".join(text.split())


def is_heldout(clip):
    return int(hashlib.sha1(Path(clip).name.encode()).hexdigest(), 16) % 100 < 15


def session_items():
    return [(ROOT / r["clip"], label_text(r["text"])) for r in Labels().trainable() if label_text(r["text"])]


def benchmark_items():
    lines = [l.strip() for l in SENTENCES.read_text().splitlines()]
    sentences = [l for l in lines if l and not l.startswith("#")]
    return [(BENCH_DIR / f"{i:02d}.mp4", label_text(s)) for i, s in enumerate(sentences, 1)
            if (BENCH_DIR / f"{i:02d}.mp4").exists()]


def crops_for(reader, clip):
    """96x96 mouth crops for a clip, cached on disk."""
    st = clip.stat()
    key = hashlib.sha1(f"{clip.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()
    path = CACHE / f"{key}.npy"
    if path.exists():
        return np.load(path)
    crops, _ = reader.mouth_crops(vsr.load_video(clip))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, crops.astype(np.uint8))
    return crops


class Tokenizer:
    def __init__(self):
        self.sp = sentencepiece.SentencePieceProcessor(model_file=str(vsr.AVSR_DIR / "spm/unigram/unigram5000.model"))
        units = vsr.UNITS_FILE.read_text(encoding="utf8").splitlines()
        self.ids = {u.split()[0]: int(u.split()[-1]) for u in units}

    def __call__(self, text):
        return torch.tensor([self.ids.get(p, self.ids["<unk>"]) for p in self.sp.EncodeAsPieces(text)])


def augment(crops):
    """Training-time version of the Auto-AVSR video transform: random 88x88 crop + time masking."""
    x = torch.from_numpy(crops).permute(0, 3, 1, 2).float() / 255.0  # T,3,96,96
    i, j = random.randint(0, 8), random.randint(0, 8)
    x = x[:, :, i:i + 88, j:j + 88]
    x = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)  # grayscale
    x = (x - 0.421) / 0.165
    for _ in range(max(1, len(x) // 50)):  # mask up to 0.4 s per 2 s of video
        n = random.randint(0, min(10, len(x) // 5))
        s = random.randint(0, len(x) - n)
        x[s:s + n] = 0
    return x


# ----------------------------------------------------------------------------- evaluation

def evaluate(reader, items, cache_reader):
    """Top-1 word errors / words (no LLM), with the reader's beam size."""
    errors = words = 0
    for clip, text in items:
        hyp, _ = reader.decode(crops_for(cache_reader, clip))
        s, d, i = word_errors(normalize(text), normalize(hyp))
        errors += s + d + i
        words += len(normalize(text).split())
    return errors, words


def fmt(e, w):
    return f"{e / w:.1%} WER ({e}/{w} words wrong)" if w else "n/a"


# ----------------------------------------------------------------------------- training

def finetune(items, crops, device, epochs, lr, accumulate=4, seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    tok = Tokenizer()
    units = vsr.UNITS_FILE.read_text(encoding="utf8").splitlines()
    model = E2E(len(dict.fromkeys(u.split()[0] for u in units)) + 2, "video")  # <blank> + units + <eos>, as in vsr
    model.load_state_dict(torch.load(vsr.BASE_CKPT, map_location="cpu", weights_only=True))
    model.to(device).train()
    for p in model.frontend.parameters():
        p.requires_grad = False
    for m in model.modules():  # batch size 1: keep the pretrained BatchNorm statistics
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    total = epochs * len(items) // accumulate
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, total // 10)))
    labels = {clip: tok(text) for clip, text in items}
    for epoch in range(1, epochs + 1):
        order, losses, t0 = list(items), [], time.perf_counter()
        random.shuffle(order)
        opt.zero_grad()
        for step, (clip, _) in enumerate(order, 1):
            x = augment(crops[clip]).unsqueeze(0).to(device)
            y = labels[clip].unsqueeze(0).to(device)
            loss, _, _, _ = model(x, torch.tensor([x.shape[1]], device=device), y)
            if torch.isfinite(loss):
                (loss / accumulate).backward()
                losses.append(loss.item())
            if step % accumulate == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
        print(f"  epoch {epoch}/{epochs}: loss {np.mean(losses):.2f}  ({time.perf_counter() - t0:.0f} s)", flush=True)
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def train(epochs=8, lr=2e-5, device=None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    items, bench = session_items(), benchmark_items()
    train_items = [it for it in items if not is_heldout(it[0])]
    val_items = [it for it in items if is_heldout(it[0])]
    print(f"\n=== Self-improvement: fine-tuning on {len(train_items)} of your clips "
          f"({len(val_items)} held out, {len(bench)} benchmark clips) ===", flush=True)
    if not train_items:
        print("Nothing to train on yet.")
        return None

    base_reader = vsr.LipReader("hybrid", ckpt=vsr.BASE_CKPT, beam_size=EVAL_BEAM)
    print("Preparing mouth crops ...", flush=True)
    crops = {}
    for clip, _ in train_items + val_items + bench:
        try:
            crops[clip] = crops_for(base_reader, clip)
        except Exception as e:
            print(f"  skipping {clip.name}: {e}")
    train_items = [it for it in train_items if it[0] in crops]
    val_items = [it for it in val_items if it[0] in crops]
    bench = [it for it in bench if it[0] in crops]

    meta_path = MODELS / "personal.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"history": []}
    current = vsr.PERSONAL_CKPT if vsr.PERSONAL_CKPT.exists() else vsr.BASE_CKPT
    print(f"Scoring the current model ({current.name}) ...", flush=True)
    cur_reader = base_reader if current == vsr.BASE_CKPT else vsr.LipReader("hybrid", ckpt=current, beam_size=EVAL_BEAM)
    cur_bench, cur_val = evaluate(cur_reader, bench, base_reader), evaluate(cur_reader, val_items, base_reader)
    print(f"  current: benchmark {fmt(*cur_bench)}, held-out {fmt(*cur_val)}", flush=True)

    print(f"Training on {device} ({epochs} epochs, lr {lr}) ...", flush=True)
    t0 = time.perf_counter()
    state = finetune(train_items, crops, device, epochs, lr)
    candidate = MODELS / "personal_candidate.pth"
    torch.save(state, candidate)
    print(f"  trained in {time.perf_counter() - t0:.0f} s. Scoring the new model ...", flush=True)
    cand_reader = vsr.LipReader("hybrid", ckpt=candidate, beam_size=EVAL_BEAM)
    new_bench, new_val = evaluate(cand_reader, bench, base_reader), evaluate(cand_reader, val_items, base_reader)
    print(f"  new:     benchmark {fmt(*new_bench)}, held-out {fmt(*new_val)}", flush=True)

    cur_err, new_err = cur_bench[0] + cur_val[0], new_bench[0] + new_val[0]
    words = cur_bench[1] + cur_val[1]
    keep = new_err < cur_err and new_bench[0] <= cur_bench[0] + max(1, cur_bench[1] // 50)
    record = {"date": f"{datetime.now():%Y-%m-%d %H:%M}", "n_train": len(train_items), "n_heldout": len(val_items),
              "benchmark_wer": round(new_bench[0] / max(1, new_bench[1]), 4),
              "heldout_wer": round(new_val[0] / new_val[1], 4) if new_val[1] else None,
              "previous_benchmark_wer": round(cur_bench[0] / max(1, cur_bench[1]), 4),
              "previous_heldout_wer": round(cur_val[0] / cur_val[1], 4) if cur_val[1] else None,
              "kept": keep, "epochs": epochs, "lr": lr}
    meta["history"] = meta.get("history", []) + [record]
    meta["n_train_last_attempt"] = len(train_items) + len(val_items)
    if keep:
        if vsr.PERSONAL_CKPT.exists():
            shutil.copy(vsr.PERSONAL_CKPT, MODELS / "personal_previous.pth")
        candidate.replace(vsr.PERSONAL_CKPT)
        meta["current"] = record
        print(f"Kept: {cur_err}/{words} -> {new_err}/{words} word errors on clips it never saw. "
              "Silent chat will use it from the next start.", flush=True)
    else:
        candidate.unlink(missing_ok=True)
        print(f"Not kept: {cur_err}/{words} -> {new_err}/{words} word errors on clips it never saw. "
              "Staying with the current model; more clips usually help.", flush=True)
    meta_path.write_text(json.dumps(meta, indent=1))
    for r in (base_reader, cur_reader, cand_reader):
        r.close()
    return record


def should_train():
    """(True/False, explanation): is there enough new data for another round?"""
    usable = len(session_items())
    meta_path = MODELS / "personal.json"
    last = json.loads(meta_path.read_text()).get("n_train_last_attempt", 0) if meta_path.exists() else 0
    if usable < MIN_CLIPS:
        return False, (f"Self-improvement: {usable}/{MIN_CLIPS} usable clips so far; the lip-reader retrains on you "
                       f"automatically from {MIN_CLIPS}. (Corrections already help right away via your personal memory.)")
    if usable - last < MIN_NEW:
        return False, f"Self-improvement: {usable - last} new usable clips since the last training; retraining after {MIN_NEW}."
    return True, f"Self-improvement: {usable - last} new usable clips; retraining."


LOCK = MODELS / "training.lock"


def _lock():
    """Only one retrain at a time (e.g. the chat quitting while a practice drill is training)."""
    if LOCK.exists():
        try:
            os.kill(int(LOCK.read_text()), 0)
            return False  # another retrain is running
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    LOCK.write_text(str(os.getpid()))
    return True


def maybe_train(force=False, **kw):
    ok, why = should_train()
    if not force and not ok:
        print(why)
        return None
    if not _lock():
        print("A retrain is already running; skipping.")
        return None
    try:
        print("Retraining the lip-reader on your clips (a few minutes; Ctrl+C to skip) ...", flush=True)
        return train(**kw)
    finally:
        LOCK.unlink(missing_ok=True)


def train_in_background(log_path):
    """Start a retrain as its own low-priority process (survives the caller quitting). Returns its message."""
    import subprocess
    import sys
    ok, why = should_train()
    if not ok:
        return why
    if LOCK.exists():
        try:
            os.kill(int(LOCK.read_text()), 0)
            return "A retrain is already running in the background."
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    log = open(log_path, "a")
    log.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M} retrain started =====\n")
    log.flush()
    subprocess.Popen(["nice", "-n", "10", sys.executable, str(ROOT / "train_personal.py")], cwd=ROOT,
                     stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                     env={**os.environ, "PYTHONUNBUFFERED": "1"})
    return f"{why} Running in the background at low priority; progress in {log_path}."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="train now, even with few clips")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--device", choices=["mps", "cpu"], default=None)
    args = ap.parse_args()
    maybe_train(force=args.force, epochs=args.epochs, lr=args.lr, device=args.device)


if __name__ == "__main__":
    main()
