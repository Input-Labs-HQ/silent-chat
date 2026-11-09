"""Out-loud speech for silent chat: the MIC toggle, the "Hey Claude" wake word and Option dictation while hidden.
Microphone capture + local Whisper transcription.

The mic is open while MIC or "Hey Claude" is on, or while you hold Option (macOS shows its orange mic dot then). It records from the MacBook's built-in
microphone by default, because using the AirPods mic switches them to low-quality "phone call" audio.
Transcription runs locally on the GPU with Whisper large-v3-turbo (MLX); nothing leaves the Mac. It's biased
towards your personal vocabulary, so names like Todoist come out spelled right.
"""
import collections
import queue
import re
import threading
import time

import numpy as np
import sounddevice as sd

RATE = 16000
WHISPER = "mlx-community/whisper-large-v3-turbo"  # what you say: loaded when needed, unloaded after 2 idle minutes
FAST_WHISPER = "mlx-community/whisper-base.en-mlx"  # just "is this 'Hey Claude'?": ~150 MB, ~30 ms, always loaded
_WHISPER_LOCK = threading.Lock()  # (mlx_whisper keeps one model in a global; calls take turns)


class Mic:
    """Microphone ring buffer with monotonic timestamps, so audio can be matched to captured video frames."""

    def __init__(self, prefer="MacBook", keep_seconds=45):
        self.prefer, self.keep = prefer, keep_seconds
        self.stream, self.rate, self.name = None, RATE, None
        self.blocks, self.lock = collections.deque(), threading.Lock()

    def _device(self):
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and self.prefer.lower() in d["name"].lower():
                return i, d["name"]
        i = sd.default.device[0]
        return i, sd.query_devices(i)["name"]

    def _callback(self, indata, frames, time_info, status):
        now = time.monotonic()
        with self.lock:
            self.blocks.append((now, indata[:, 0].copy()))
            while self.blocks and now - self.blocks[0][0] > self.keep:
                self.blocks.popleft()

    @property
    def on(self):
        return self.stream is not None

    def start(self):
        if self.stream is None:
            device, self.name = self._device()
            for rate in (RATE, int(sd.query_devices(device)["default_samplerate"])):
                try:
                    self.stream = sd.InputStream(device=device, channels=1, samplerate=rate, dtype="float32",
                                                 blocksize=rate // 10, callback=self._callback)
                    self.rate = rate
                    break
                except sd.PortAudioError:
                    continue
            self.stream.start()
        return self.name

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        with self.lock:
            self.blocks.clear()

    def segment(self, t0, t1):
        """Audio between two time.monotonic() timestamps, as 16 kHz float32."""
        with self.lock:
            blocks = list(self.blocks)
        out = []
        for t_end, block in blocks:
            t_start = t_end - len(block) / self.rate
            if t_end < t0 or t_start > t1:
                continue
            a, z = max(0, int((t0 - t_start) * self.rate)), min(len(block), int((t1 - t_start) * self.rate))
            out.append(block[a:z])
        audio = np.concatenate(out) if out else np.zeros(0, np.float32)
        if self.rate != RATE and len(audio):
            from scipy.signal import resample_poly
            audio = resample_poly(audio, RATE, self.rate).astype(np.float32)
        return audio


class Transcriber:
    """Whisper, loaded on first use. idle_unload: seconds unused after which the model is dropped from memory
    (it reloads from disk, ~1 s, the next time it's needed)."""

    def __init__(self, model=WHISPER, idle_unload=None):
        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder
        self.mw, self.path, self.holder, self.lock = mlx_whisper, model, ModelHolder, _WHISPER_LOCK
        self.loaded, self.last_used, self.idle_unload = None, time.monotonic(), idle_unload
        if idle_unload:
            threading.Thread(target=self._reaper, daemon=True).start()

    def _reaper(self):
        while True:
            time.sleep(15)
            if self.loaded is not None and time.monotonic() - self.last_used > self.idle_unload:
                self.unload()

    def unload(self):
        import gc
        import mlx.core as mx
        with self.lock:
            if self.holder.model is self.loaded:
                self.holder.model, self.holder.model_path = None, None
            self.loaded = None
            gc.collect()
            mx.clear_cache()

    def _ensure(self):  # (under self.lock) make this the model mlx_whisper uses
        if self.loaded is None:
            import mlx.core as mx
            from mlx_whisper.load_models import load_model
            self.loaded = load_model(self.path, dtype=mx.float16)
        self.holder.model, self.holder.model_path = self.loaded, self.path

    def warm(self):
        """Load the model (a second or two) so the first real transcription is fast."""
        self(np.random.default_rng(0).normal(0, 0.01, RATE).astype(np.float32))

    def __call__(self, audio, vocabulary=()):
        """Text spoken in `audio` (16 kHz float32), or "" if there's no clear speech."""
        if len(audio) < RATE * 0.3 or float(np.sqrt(np.mean(audio ** 2))) < 0.003:
            return ""  # too short or too quiet: Whisper would only hallucinate ("Thank you.")
        prompt = ("Vocabulary: " + ", ".join(vocabulary) + ".") if vocabulary else None
        with self.lock:
            self._ensure()
            self.last_used = time.monotonic()
            result = self.mw.transcribe(audio, path_or_hf_repo=self.path, language="en", initial_prompt=prompt,
                                        condition_on_previous_text=False, temperature=0.0)
        segments = [s for s in result.get("segments", [])
                    if not (s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -1.0)]
        text = " ".join(s["text"].strip() for s in segments).strip()
        return "" if HALLUCINATION.match(text) else text


# what Whisper writes for silence, music or noise (and echoes of its own vocabulary prompt)
HALLUCINATION = re.compile(r"^\W*(closed captioning\b.*|captions? by\b.*|subtitles? by\b.*|thank you\W*|thanks for "
                           r"watching\W*|so,? let's go\W*|um+\W*|uh+\W*|yeah\W*|but\W*|bye\W*|you\W*|"
                           r"vocabulary\b.*|.*\bvoiceover\b.*)$", re.I)


class SpeechDetector:
    """Is there a person speaking in this audio? Silero VAD (a 2 MB model, ~5 ms per second of audio): music,
    video soundtracks, typing and room noise score ~0, so Whisper only runs on real speech."""

    def __init__(self):
        self.model, self.lock = None, threading.Lock()

    def fraction(self, audio):
        """Share of 32 ms frames with speech (0..1)."""
        import torch
        with self.lock:
            if self.model is None:
                from silero_vad import load_silero_vad
                self.model = load_silero_vad()
            a = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
            n = len(a) // 512
            if n == 0:
                return 0.0
            with torch.no_grad():
                p = [float(self.model(a[i * 512:(i + 1) * 512], RATE)) for i in range(n)]
            self.model.reset_states()
        return float(np.mean(np.array(p) > 0.5))


def voiced(audio, level=None):
    """Fraction of 100 ms windows clearly louder than the quiet parts of the same clip (speech out loud, not just
    mouthing). The threshold adapts to the room and the mic's gain."""
    n = len(audio) // (RATE // 10)
    if n == 0:
        return 0.0
    rms = np.sqrt(np.mean(audio[:n * (RATE // 10)].reshape(n, -1) ** 2, axis=1))
    if level is None:
        level = max(0.003, 3.0 * float(np.percentile(rms, 10)))
    return float(np.mean(rms > level))


# "Hey Claude" as Whisper tends to write it (with your vocabulary prompt it's usually spelled right)
WAKE_PHRASE = re.compile(r"^\W*(?:hey|hi|hello|okay|ok)\W+(?:claude|claud|clod|cloud|clawed|klaud|klod)\b\W*", re.I)


class WakeWord:
    """Always-on "Hey Claude". Finds stretches of speech in the mic stream (by loudness), transcribes the first ~2 s
    of each locally to check for the wake phrase, and only then transcribes the whole utterance. Speech without the
    wake phrase is dropped without being transcribed in full. Everything stays on this Mac.

    poll() returns events: ("wake",) as soon as the phrase is heard (you may still be talking), ("said", text) for
    the request after it, and ("idle",) when it heard "Hey Claude" but nothing followed. After "Hey Claude" on its
    own, or follow_up(seconds), the next utterance counts without the wake phrase.
    """
    BLOCK = 0.1

    def __init__(self, mic, transcriber, vocabulary=lambda: (), muted=lambda: False):
        self.mic, self.transcriber, self.vocabulary, self.muted = mic, transcriber, vocabulary, muted
        self.enabled, self.follow_until, self.events = False, 0.0, queue.Queue()
        self.levels = collections.deque(maxlen=150)  # the last 15 s of mic loudness (RMS per 100 ms)
        self.vad, self.stats = SpeechDetector(), {"checked": 0, "no_speech": 0}
        self.fast = None  # the small Whisper for the wake check (made on first use)
        self.log = lambda text: print(f"[wake] {text}", flush=True)
        threading.Thread(target=self._run, daemon=True).start()

    def follow_up(self, seconds):
        self.follow_until = time.monotonic() + seconds

    @property
    def following(self):
        return time.monotonic() < self.follow_until

    def level_report(self):
        if not self.levels:
            return "no mic audio yet"
        st = self.stats
        skipped = f"; {st['no_speech']} of {st['checked']} sounds weren't speech (not transcribed)"
        lv = np.array(self.levels)
        return (f"mic noise {np.percentile(lv, 20):.4f}, loudest {lv.max():.4f} in the last {len(lv) / 10:.0f} s, "
                f"speech threshold {self.threshold():.4f}" + skipped)

    def threshold(self):
        """Louder than this counts as speech: 2.5x the room's background level (the quiet 20% of the last 15 s)."""
        noise = float(np.percentile(np.array(self.levels), 20)) if len(self.levels) >= 10 else 0.002
        return max(0.003, 2.5 * noise)

    def poll(self):
        out = []
        while not self.events.empty():
            out.append(self.events.get())
        return out

    def _text(self, t0, t1, fast=False):
        if fast:
            if self.fast is None:
                self.fast = Transcriber(FAST_WHISPER)
            return self.fast(self.mic.segment(t0, t1), ["Claude"])
        tr = self.transcriber()
        if tr is None:
            return ""
        return tr(self.mic.segment(t0, t1), ["Claude", *self.vocabulary()])

    def _run(self):
        last, seg, waiting = time.monotonic(), None, False
        while True:
            time.sleep(self.BLOCK)
            now = time.monotonic()
            if waiting and not self.following and seg is None:
                waiting = False
                self.events.put(("idle",))  # "Hey Claude" ... and then nothing
            if not self.enabled or not self.mic.on or self.muted() or self.transcriber() is None:
                last, seg = now, None
                continue
            with self.mic.lock:
                blocks = [(t, b) for t, b in self.mic.blocks if t > last]
            for t_end, block in blocks:
                last = t_end
                rms = float(np.sqrt(np.mean(block ** 2)))
                loud = rms > self.threshold()
                self.levels.append(rms)
                if seg is None:
                    if not loud:
                        continue
                    seg = {"start": t_end - len(block) / self.mic.rate - 0.3, "voiced": 0, "quiet": 0,
                           "follow": self.following, "checked": False, "woke": False, "skip": False}
                seg["voiced"] += loud
                seg["quiet"] = 0 if loud else seg["quiet"] + 1
                length = t_end - seg["start"]
                if not seg["checked"] and not seg["follow"] and length >= 2.2:  # early look for "Hey Claude"
                    seg["checked"] = True
                    self.stats["checked"] += 1
                    if self.vad.fraction(self.mic.segment(seg["start"], seg["start"] + 2.2)) < 0.25:
                        self.stats["no_speech"] += 1  # music, a video, noise: not worth transcribing
                        seg["skip"] = True
                        continue
                    head = self._text(seg["start"], seg["start"] + 2.2, fast=True)
                    seg["woke"] = bool(WAKE_PHRASE.search(head))
                    self.log(f"speech: {'WAKE' if seg['woke'] else 'no wake phrase'} ({head[:28]!r})")
                    seg["skip"] = not seg["woke"]
                    if seg["woke"]:
                        self.events.put(("wake",))
                if seg["quiet"] >= 8 or length > 25:  # 0.8 s of quiet (or very long): the utterance is over
                    done, seg = seg, None
                    if done["skip"] or done["voiced"] < 3:
                        continue
                    if not done["checked"]:  # (a short utterance: the speech check happens here instead)
                        self.stats["checked"] += 1
                        if self.vad.fraction(self.mic.segment(done["start"], t_end - 0.6)) < 0.25:
                            self.stats["no_speech"] += 1
                            continue
                    if not done["woke"] and not done["follow"]:  # a short sound: the small model checks for the
                        quick = self._text(done["start"], t_end - 0.6, fast=True)  # wake phrase first
                        if not WAKE_PHRASE.search(quick):
                            self.log(f"short speech: no wake phrase ({quick[:28]!r})")
                            continue
                    text = self._text(done["start"], t_end - 0.6)
                    m = WAKE_PHRASE.search(text)
                    if not done["woke"] and not done["follow"]:
                        self.log(f"short speech: {'WAKE' if m else 'no wake phrase'} ({text[:28]!r})")
                    if m or done["woke"]:
                        rest = text[m.end():].strip() if m else text
                        if not done["woke"]:
                            self.events.put(("wake",))
                        if re.search(r"[A-Za-z0-9]{2}", rest):
                            self.follow_until, waiting = 0.0, False
                            self.events.put(("said", rest))
                        else:  # "Hey Claude" ... (pause) ... the request
                            self.follow_up(7)
                            waiting = True
                    elif done["follow"] and re.search(r"[A-Za-z0-9]{2}", text):
                        self.follow_until, waiting = 0.0, False
                        self.events.put(("said", text))
