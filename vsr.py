#!/usr/bin/env python
"""Visual-only speech recognition (lip reading) with Auto-AVSR on Apple Silicon.

Examples:
  python vsr.py --record                    # record from the webcam (hold SPACE while speaking), then transcribe
  python vsr.py --video clip.mp4            # transcribe an existing video
  python vsr.py --video clip.mp4 --device mps --save-crop
"""
import argparse
import contextlib
import json
import os
import subprocess
import sys
import textwrap
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

# Let ops that MPS lacks run on the CPU; we capture and report every fallback (see LipReader.transcribe).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import mediapipe as mp
import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parent
AVSR_DIR = ROOT / "third_party" / "auto_avsr"
sys.path.append(str(AVSR_DIR))

from espnet.nets.batch_beam_search import BatchBeamSearch  # noqa: E402
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E  # noqa: E402
from espnet.nets.scorers.length_bonus import LengthBonus  # noqa: E402
from preparation.detectors.mediapipe.video_process import VideoProcess  # noqa: E402

BASE_CKPT = ROOT / "models" / "vsr_trlrs2lrs3vox2avsp_base.pth"
PERSONAL_CKPT = ROOT / "models" / "personal.pth"  # written by train_personal.py when it beats the base model
DEFAULT_CKPT = BASE_CKPT


def default_ckpt():
    return PERSONAL_CKPT if PERSONAL_CKPT.exists() else BASE_CKPT
DETECTOR_MODEL = ROOT / "models" / "blaze_face_short_range.tflite"
LANDMARKER_MODEL = ROOT / "models" / "face_landmarker.task"  # lip landmarks + blendshapes
UNITS_FILE = AVSR_DIR / "spm" / "unigram" / "unigram5000_units.txt"
SPM_MODEL = AVSR_DIR / "spm" / "unigram" / "unigram5000.model"  # text -> the model's word pieces
FPS = 25  # the model was trained on 25 fps video
MAX_SIDE = 1280  # downscale bigger frames to save memory; the mouth crop is 96x96 anyway


# ----------------------------------------------------------------------------- video I/O

def _probe_size(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:stream_side_data=rotation", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    w, h = int(stream["width"]), int(stream["height"])
    rot = next((int(float(sd["rotation"])) for sd in stream.get("side_data_list", []) if "rotation" in sd), 0)
    if abs(rot) % 180 == 90:  # ffmpeg auto-rotates phone videos, so swap
        w, h = h, w
    return w, h


def load_video(path):
    """Decode any video to RGB uint8 frames (T, H, W, 3) resampled to 25 fps."""
    w, h = _probe_size(path)
    vf = f"fps={FPS}"
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        w, h = int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2
        vf += f",scale={w}:{h}"
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def write_video(frames, path, fps=FPS, pix_fmt="rgb24"):
    """Write uint8 frames (T, H, W, 3) to an H.264 mp4 with ffmpeg."""
    t, h, w = frames.shape[:3]
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{w}x{h}", "-r", str(fps),
         "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE,
    )
    proc.communicate(np.ascontiguousarray(frames).tobytes())
    if proc.returncode:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def resample_to_fps(frames, stamps, fps=FPS):
    """Frames captured at a variable camera rate (with monotonic timestamps) -> exactly `fps` by nearest timestamp."""
    stamps = np.asarray(stamps)
    grid = np.arange(stamps[0], stamps[-1], 1.0 / fps)
    idx = np.abs(stamps[None, :] - grid[:, None]).argmin(axis=1)
    return np.stack([frames[i] for i in idx])


# ----------------------------------------------------------------------------- webcam recorder

class Recorder:
    """Webcam recorder with a live (mirrored) preview window.

    Hold SPACE while speaking (or tap to start / tap to stop), ESC retakes, Q quits.
    """

    WINDOW = "Lip-reading recorder"
    REPEAT_GAP = 0.25  # macOS key-repeat events arrive faster than this while a key is held down

    def __init__(self, camera=0, width=1280, height=720, wait_seconds=30, window=True, start=True):
        self.camera, self.size, self.cap = camera, (width, height), None
        if start:  # start=False: the camera (and its light) stays off until resume()
            self._open(wait_seconds)
        if window:
            cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)

    def _open(self, wait_seconds=30):
        # On first use macOS asks for camera permission asynchronously and OpenCV fails straight away,
        # so keep retrying while the user answers the prompt.
        deadline = time.monotonic() + wait_seconds
        while True:
            self.cap = cv2.VideoCapture(self.camera, cv2.CAP_AVFOUNDATION)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                if self.cap.read()[0]:
                    break
            self.cap.release()
            self.cap = None
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Could not open the camera. Allow camera access for the app running this script "
                    "(System Settings > Privacy & Security > Camera), then try again."
                )
            print("Waiting for camera access (click Allow if macOS asks) ...", flush=True)
            time.sleep(2)

    def pause(self):
        """Release the camera (its light goes off)."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def resume(self, wait_seconds=10):
        deadline = time.monotonic() + 3
        while getattr(self, "_opening", False) and time.monotonic() < deadline:  # open_async() still starting it
            time.sleep(0.05)
        self._cancel = False
        if self.cap is None:
            self._open(wait_seconds)

    def open_async(self):
        """Start the camera in the background (it takes ~0.5-1 s); `cap` is set once it delivers frames."""
        import threading
        if self.cap is not None or getattr(self, "_opening", False):
            return
        self._opening, self._cancel = True, False

        def run():
            cap = cv2.VideoCapture(self.camera, cv2.CAP_AVFOUNDATION)
            ok = cap.isOpened()
            if ok:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
                cap.set(cv2.CAP_PROP_FPS, 30)
                ok = cap.read()[0]
            if ok and not self._cancel and self.cap is None:
                self.cap = cap
            else:
                cap.release()
            self._opening = False
        threading.Thread(target=run, daemon=True).start()

    def close_async(self):
        """Release the camera, including one still being opened by open_async()."""
        self._cancel = True
        self.pause()

    def _read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera stopped delivering frames.")
        if max(frame.shape[:2]) > MAX_SIDE:
            s = MAX_SIDE / max(frame.shape[:2])
            frame = cv2.resize(frame, (int(frame.shape[1] * s) // 2 * 2, int(frame.shape[0] * s) // 2 * 2))
        return frame

    @staticmethod
    def _text(img, text, org, scale, color, thick):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def _show(self, frame, header, sentence, status, recording=False, notice=None):
        disp = cv2.flip(frame, 1)  # mirror the preview only; the saved video is not mirrored
        h, w = disp.shape[:2]
        # face guide: keep your face roughly inside this box, close to the camera
        cv2.rectangle(disp, (int(w * 0.3), int(h * 0.25)), (int(w * 0.7), int(h * 0.95)), (200, 200, 200), 1)
        small = [p for line in header for p in (textwrap.wrap(line, 60) or [""])]
        lines_big = [sentence] if isinstance(sentence, str) else list(sentence or [])
        big = [p for s in lines_big if s for p in textwrap.wrap(s, 32)]
        banner = 20 + 34 * len(small) + 56 * len(big) + 10
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, disp, 0.45, 0, disp)
        y = 36
        for part in small:
            self._text(disp, part, (20, y), 0.8, (220, 220, 220), 2)
            y += 34
        for part in big:
            y += 18
            self._text(disp, part, (20, y + 20), 1.5, (80, 230, 255), 3)  # the sentence to say, in yellow
            y += 38
        if notice and not recording:
            self._text(disp, notice, (20, h - 65), 0.9, (60, 60, 255), 2)
        if recording:
            cv2.circle(disp, (w - 40, h - 35), 14, (0, 0, 255), -1)
        self._text(disp, status, (20, h - 25), 0.9, (0, 0, 255) if recording else (0, 220, 0), 2)
        cv2.imshow(self.WINDOW, disp)

    def intro(self, header, sentence=""):
        """Show an instruction screen until ENTER (returns True) or Q (returns False)."""
        while True:
            self._show(self._read(), header, sentence, "Press ENTER to begin   Q = quit")
            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10):
                return True
            if key in (ord("q"), ord("Q")):
                return False

    def take(self, out_path, header, sentence="", max_seconds=20.0, check=None):
        """Record one clip. Returns the saved path, or None if the user pressed Q to quit.

        SPACE works both ways: hold it while speaking (push-to-talk), or tap to start and tap to stop.
        OpenCV gives no key-up events, so holding is detected from the key-repeat stream, and a
        second "tap" only stops recording if no key-repeat follows it within REPEAT_GAP.
        check(frames_rgb) may return an error message; the clip is then discarded and retaken.
        """
        notice = None
        while True:  # loop so ESC / too-short / unusable clips get retaken
            frames, stamps = [], []
            start = last_space = pending_stop = end = None
            held = retake = False
            while True:
                frame = self._read()
                now = time.monotonic()
                if start is not None:
                    frames.append(frame)
                    stamps.append(now)
                    hint = "release SPACE to stop" if held else "tap SPACE to stop"
                    status = f"REC {now - start:4.1f}s   {hint}   ESC = retake"
                else:
                    status = "Hold SPACE while you say it   Q = quit"
                self._show(frame, header, sentence, status, start is not None, notice)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    return None
                if key == 27 and start is not None:  # ESC: discard and retake
                    retake = True
                    break
                if key == 32:  # SPACE
                    if start is None:
                        start = now
                    elif now - last_space < self.REPEAT_GAP:
                        held, pending_stop = True, None  # key-repeat: SPACE is being held
                    elif not held:
                        pending_stop = now  # a fresh tap: stop, unless key-repeat follows
                    last_space = now
                if start is not None:
                    if held and now - last_space > 1.5 * self.REPEAT_GAP:  # released after holding
                        end = last_space + 0.1
                        break
                    if pending_stop is not None and now - pending_stop > self.REPEAT_GAP:  # tap confirmed
                        end = pending_stop
                        break
                    if now - start > max_seconds:
                        break
            if end is not None:  # drop the frames recorded after the stop gesture
                frames = [f for f, s in zip(frames, stamps) if s <= end]
                stamps = [s for s in stamps if s <= end]
            if retake:
                notice = None
                continue
            if len(frames) < 15:
                notice = "Too short - hold SPACE for the whole sentence. Please say it again."
                print(notice, flush=True)
                continue
            clip = resample_to_fps(frames, stamps)
            if check is not None:
                self._show(frame, header, sentence, "Checking the clip ...")
                cv2.waitKey(1)
                problem = check(np.ascontiguousarray(clip[..., ::-1]))  # BGR -> RGB
                if problem:
                    notice = f"{problem} - please say it again."
                    print(notice, flush=True)
                    continue
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            write_video(clip, out_path, pix_fmt="bgr24")
            return Path(out_path)

    def close(self):
        self.pause()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


# ----------------------------------------------------------------------------- mouth cropping

@contextlib.contextmanager
def _quiet_stderr():
    """Hide MediaPipe/LiteRT's native INFO logging (it writes straight to fd 2)."""
    sys.stderr.flush()
    saved = os.dup(2)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)


class FaceKeypoints:
    """MediaPipe Tasks face detector -> 4 keypoints per frame (right eye, left eye, nose tip, mouth centre).

    Same keypoints as the legacy mp.solutions.face_detection used by Auto-AVSR's mediapipe detector,
    which no longer exists in current MediaPipe releases.
    """

    def __init__(self, model_path=DETECTOR_MODEL, min_confidence=0.5):
        opts = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )
        with _quiet_stderr():
            self.detector = mp.tasks.vision.FaceDetector.create_from_options(opts)

    def close(self):
        if self.detector is not None:
            with _quiet_stderr():
                self.detector.close()
            self.detector = None

    def __call__(self, frames):
        with _quiet_stderr():
            return self._detect(frames)

    def _detect(self, frames):
        landmarks = []
        for frame in frames:
            result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)))
            if not result.detections:
                landmarks.append(None)
                continue
            face = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
            h, w = frame.shape[:2]
            landmarks.append(np.array([[k.x * w, k.y * h] for k in face.keypoints[:4]], dtype=np.float64))
        return landmarks


LIP_POINTS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]  # outer lips


class MouthActivity:
    """How much the mouth is moving right now, from MediaPipe FaceLandmarker blendshapes.

    Each frame gives 27 mouth/jaw blendshape scores (jawOpen, mouthClose, mouthPucker, ...). Activity is the
    frame-to-frame change summed over them, smoothed with an exponential moving average. A still mouth gives
    ~0.01-0.03, silent mouthing ~0.1-0.5. Returns None when no face is found.
    """

    def __init__(self, model_path=LANDMARKER_MODEL, smoothing=0.35):
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,  # head pose
        )
        with _quiet_stderr():
            self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
        self.smoothing, self.prev, self.level, self.last_ms = smoothing, None, 0.0, -1
        self.mouth = None  # (centre x, centre y, mouth width) in pixels, from the lip landmarks
        self.landmarks = None  # (478, 3) face landmarks in pixels (z scaled like x)
        self.blend = {}  # all 52 blendshape scores, e.g. mouthSmileLeft, jawOpen
        self.pose = None  # head (yaw, pitch, roll) in degrees; (0, 0, 0) = facing the camera
        self.matrix = None  # the 4x4 face transform (rotation + position), for gaze

    def __call__(self, frame_rgb, t):
        ms = max(int(t * 1000), self.last_ms + 1)  # VIDEO mode needs strictly increasing timestamps
        self.last_ms = ms
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb)), ms)
        if not result.face_blendshapes:
            self.prev, self.mouth, self.landmarks, self.blend, self.pose = None, None, None, {}, None
            self.matrix = None
            return None
        h, w = frame_rgb.shape[:2]
        lips = np.array([[result.face_landmarks[0][i].x * w, result.face_landmarks[0][i].y * h] for i in LIP_POINTS])
        self.mouth = (*lips.mean(axis=0), float(np.ptp(lips[:, 0])))
        self.landmarks = np.array([[p.x * w, p.y * h, p.z * w] for p in result.face_landmarks[0]])
        self.blend = {c.category_name: c.score for c in result.face_blendshapes[0]}
        self.matrix = None
        if result.facial_transformation_matrixes:
            self.matrix = np.asarray(result.facial_transformation_matrixes[0])  # head rotation + position (gaze)
            r = self.matrix[:3, :3]
            self.pose = (float(np.degrees(np.arctan2(r[0, 2], r[2, 2]))),  # yaw: turned left/right
                         float(np.degrees(np.arcsin(np.clip(-r[1, 2], -1, 1)))),  # pitch: up/down
                         float(np.degrees(np.arctan2(r[1, 0], r[1, 1]))))  # roll: tilted
        v = np.array([c.score for c in result.face_blendshapes[0] if c.category_name.startswith(("mouth", "jaw"))])
        change = 0.0 if self.prev is None else float(np.abs(v - self.prev).sum())
        self.prev = v
        self.level = self.smoothing * change + (1 - self.smoothing) * self.level
        return self.level

    def close(self):
        with _quiet_stderr():
            self.landmarker.close()


class ClipChecker:
    """Quick usability check right after a take: is the face tracked, and can the mouth be cropped?"""

    def __init__(self, min_face_fraction=0.8):
        self.keypoints = FaceKeypoints()
        self.video_process = VideoProcess(convert_gray=False)
        self.min_face_fraction = min_face_fraction

    def __call__(self, frames_rgb):
        landmarks = self.keypoints(frames_rgb)
        found = sum(lm is not None for lm in landmarks)
        if found < self.min_face_fraction * len(landmarks):
            return f"Face found in only {found}/{len(landmarks)} frames (keep your face in the box)"
        try:
            self.video_process(frames_rgb, landmarks)
        except Exception as e:  # e.g. mouth too close to the frame edge
            return f"Mouth crop failed ({e})"
        return None

    def close(self):
        self.keypoints.close()


VIDEO_TRANSFORM = torch.nn.Sequential(  # Auto-AVSR test-time transform
    torchvision.transforms.CenterCrop(88),
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Normalize(0.421, 0.165),
)


# ----------------------------------------------------------------------------- model

def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


DEVICES = ["hybrid", "cpu", "mps"]
DEFAULT_DEVICE = "hybrid" if torch.backends.mps.is_available() else "cpu"


class LipReader:
    """device: "cpu", "mps", or "hybrid" (encoder on MPS, beam search on CPU).

    Measured on an M5 Pro for a 5 s clip, beam 40: CPU = 0.75 s encoder + 2.6 s beam search;
    MPS = 0.04 s encoder + 5.9 s beam search (lots of tiny sequential ops). Hybrid takes the best of both.
    """

    def __init__(self, device=DEFAULT_DEVICE, ckpt=None, beam_size=40, ctc_weight=0.1):
        ckpt = Path(ckpt) if ckpt else default_ckpt()
        self.ckpt, self.loaded_mtime = ckpt, ckpt.stat().st_mtime
        if device in ("mps", "hybrid") and not torch.backends.mps.is_available():
            raise RuntimeError(f"--device {device} needs MPS, but torch.backends.mps.is_available() is False.")
        self.device = torch.device("mps" if device in ("mps", "hybrid") else "cpu")  # encoder device
        self.search_device = torch.device("cpu" if device in ("cpu", "hybrid") else "mps")  # beam search device
        units = UNITS_FILE.read_text(encoding="utf8").splitlines()
        self.token_list = ["<blank>"] + list(dict.fromkeys(u.split()[0] for u in units)) + ["<eos>"]

        self.model = E2E(len(self.token_list), "video")
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        for part in (self.model.frontend, self.model.proj_encoder, self.model.encoder):
            part.to(self.device)
        for part in (self.model.decoder, self.model.ctc):
            part.to(self.search_device)

        scorers = self.model.scorers()
        scorers["lm"] = None
        scorers["length_bonus"] = LengthBonus(len(self.token_list))
        self.beam_search = BatchBeamSearch(
            beam_size=beam_size,
            vocab_size=len(self.token_list),
            weights={"decoder": 1.0 - ctc_weight, "ctc": ctc_weight, "lm": 0.0, "length_bonus": 0.0},
            scorers=scorers,
            sos=self.model.odim - 1,
            eos=self.model.odim - 1,
            token_list=self.token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
        )
        self.keypoints = FaceKeypoints()
        self.video_process = VideoProcess(convert_gray=False)

    def close(self):
        self.keypoints.close()

    def reload_if_newer(self):
        """Switch to models/personal.pth if it was retrained since this reader loaded. Returns True if it did."""
        ckpt = default_ckpt()
        mtime = ckpt.stat().st_mtime
        if ckpt == self.ckpt and mtime <= self.loaded_mtime:
            return False
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)  # copies into the existing modules, on whatever device they're on
        self.ckpt, self.loaded_mtime = ckpt, mtime
        return True

    def mouth_crops(self, frames):
        """RGB frames (T,H,W,3) -> 96x96 mouth crops (T,96,96,3), as in Auto-AVSR preprocessing."""
        landmarks = self.keypoints(frames)
        found = sum(lm is not None for lm in landmarks)
        if found == 0:
            raise RuntimeError("No face detected in any frame. Is your face visible and well lit?")
        crops = self.video_process(frames, landmarks)
        return crops, found

    def _hyp_text(self, hyp):
        ids = [int(t) for t in hyp.yseq[1:]]
        return "".join(self.token_list[i] for i in ids).replace("<eos>", "").replace("▁", " ").strip()

    def decode(self, crops, nbest=1):
        """Mouth crops -> transcript. Returns (text, list of MPS->CPU fallback warnings).

        With nbest > 1, returns ([(text, log_prob), ...] best first with duplicate texts removed, fallbacks).
        """
        x = torch.from_numpy(crops).permute(0, 3, 1, 2).float() / 255.0  # T,3,96,96
        x = VIDEO_TRANSFORM(x).unsqueeze(0).to(self.device)  # 1,T,1,88,88
        with warnings.catch_warnings(record=True) as caught, torch.no_grad():
            warnings.simplefilter("always")
            feats = self.model.frontend(x)
            feats = self.model.proj_encoder(feats)
            enc, _ = self.model.encoder(feats, None)
            hyps = self.beam_search(enc.squeeze(0).to(self.search_device))
        fallbacks = sorted({str(w.message) for w in caught if "MPS" in str(w.message) or "fall back" in str(w.message)})
        if nbest == 1:
            return self._hyp_text(hyps[0]), fallbacks
        ranked = {}
        for h in hyps:
            ranked.setdefault(self._hyp_text(h), float(h.score))
            if len(ranked) == nbest:
                break
        return list(ranked.items()), fallbacks

    def encode(self, crops):
        """Mouth crops -> encoder output (1, T, D), on the decoder's device. Shared by decode_enc() and score()."""
        x = torch.from_numpy(crops).permute(0, 3, 1, 2).float() / 255.0
        x = VIDEO_TRANSFORM(x).unsqueeze(0).to(self.device)
        with torch.no_grad():
            enc, _ = self.model.encoder(self.model.proj_encoder(self.model.frontend(x)), None)
        return enc.to(self.search_device)

    def decode_enc(self, enc, nbest=10):
        """Open-vocabulary n-best from an encoder output: [(text, log_prob), ...]."""
        with torch.no_grad():
            hyps = self.beam_search(enc.squeeze(0))
        ranked = {}
        for h in hyps:
            ranked.setdefault(self._hyp_text(h), float(h.score))
            if len(ranked) == nbest:
                break
        return list(ranked.items())

    def tokens(self, text):
        if not hasattr(self, "_sp"):
            import sentencepiece
            self._sp = sentencepiece.SentencePieceProcessor(model_file=str(SPM_MODEL))
            self._index = {t: i for i, t in enumerate(self.token_list)}
        clean = re.sub(r"[^A-Z0-9' ]+", " ", text.upper()).split()
        return [self._index.get(p, self._index.get("<unk>", 1)) for p in self._sp.encode(" ".join(clean), out_type=str)]

    def score(self, enc, candidates, ctc_weight=None):
        """Closed-set lip-reading: how well each candidate phrase fits the clip, in one batched pass (no search).

        Returns [(candidate, log_prob), ...] best first, using the same decoder + CTC mix as decoding, so the
        scores are comparable with decode_enc()'s.
        """
        import torch.nn.functional as F
        from espnet.nets.pytorch_backend.transformer.mask import target_mask
        w = self.beam_search.weights["ctc"] if ctc_weight is None else ctc_weight
        ids = [self.tokens(c) or [0] for c in candidates]
        k, longest, eos = len(ids), max(len(i) for i in ids), self.model.eos
        dev = enc.device
        ys_in = torch.full((k, longest + 1), eos, dtype=torch.long, device=dev)
        ys_out = torch.full((k, longest + 1), -1, dtype=torch.long, device=dev)
        for j, seq in enumerate(ids):
            ys_in[j, 0] = self.model.sos
            ys_in[j, 1:len(seq) + 1] = torch.tensor(seq)
            ys_out[j, :len(seq)] = torch.tensor(seq)
            ys_out[j, len(seq)] = eos
        with torch.no_grad():
            mem = enc.expand(k, -1, -1)
            pred, _ = self.model.decoder(ys_in, target_mask(ys_in, -1), mem, None)
            logp = F.log_softmax(pred, dim=-1)
            valid = ys_out >= 0
            att = (logp.gather(-1, ys_out.clamp(min=0).unsqueeze(-1)).squeeze(-1) * valid).sum(-1)
            lp = self.model.ctc.log_softmax(enc).transpose(0, 1).expand(-1, k, -1)  # (T, k, V)
            targets = torch.tensor([t for seq in ids for t in seq], dtype=torch.long)
            lengths = torch.tensor([len(seq) for seq in ids])
            ctc = -F.ctc_loss(lp.cpu(), targets, torch.full((k,), lp.shape[0], dtype=torch.long), lengths,
                              blank=0, reduction="none", zero_infinity=True)
            total = (1 - w) * att.cpu() + w * ctc
        return sorted(zip(candidates, total.tolist()), key=lambda c: -c[1])

    def transcribe(self, video_path, save_crop=None):
        t0 = time.perf_counter()
        frames = load_video(video_path)
        crops, found = self.mouth_crops(frames)
        t1 = time.perf_counter()
        if save_crop:
            model_view = (VIDEO_TRANSFORM[:2](torch.from_numpy(crops).permute(0, 3, 1, 2)))  # what the model sees
            model_view = model_view.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).numpy()
            model_view = np.repeat(np.repeat(model_view, 4, axis=1), 4, axis=2)  # 88 -> 352 px for viewing
            write_video(model_view, save_crop)
        _sync(self.device)
        t2 = time.perf_counter()
        text, fallbacks = self.decode(crops)
        _sync(self.device)
        t3 = time.perf_counter()
        return {
            "text": text,
            "frames": len(frames),
            "face_frames": found,
            "clip_seconds": len(frames) / FPS,
            "crop_seconds": t1 - t0,
            "model_seconds": t3 - t2,
            "mps_fallbacks": fallbacks,
        }


# ----------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="existing video file to transcribe")
    src.add_argument("--record", action="store_true", help="record a clip from the webcam first")
    ap.add_argument("--device", choices=DEVICES, default=DEFAULT_DEVICE,
                    help="hybrid = encoder on Apple GPU (MPS) + beam search on CPU; fastest on Apple Silicon")
    ap.add_argument("--camera", type=int, default=0, help="camera index (0 = built-in FaceTime camera)")
    ap.add_argument("--max-seconds", type=float, default=20.0, help="recording stops automatically after this")
    ap.add_argument("--beam-size", type=int, default=40)
    ap.add_argument("--ckpt", type=Path, default=None, help="default: models/personal.pth if present, else base")
    ap.add_argument("--save-crop", action="store_true", help="also write <video>_mouth.mp4 showing what the model sees")
    args = ap.parse_args()

    video = args.video
    if args.record:
        video = ROOT / "recordings" / f"clip_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        rec, checker = Recorder(args.camera), ClipChecker()
        try:
            video = rec.take(video, ["Say (or mouth) anything. Look at the camera, face inside the box."],
                             max_seconds=args.max_seconds, check=checker)
        finally:
            rec.close()
            checker.close()
        if video is None:
            print("Cancelled.")
            return
        print(f"Saved {video}")

    print(f"Loading model on {args.device} ...", flush=True)
    t = time.perf_counter()
    reader = LipReader(args.device, args.ckpt, args.beam_size)
    print(f"  loaded in {time.perf_counter() - t:.1f}s")

    crop_path = video.with_name(video.stem + "_mouth.mp4") if args.save_crop else None
    try:
        r = reader.transcribe(video, save_crop=crop_path)
    except (RuntimeError, OverflowError) as e:
        sys.exit(f"Could not transcribe {video}: {e}")
    finally:
        reader.close()
    print(f"\nTranscript: {r['text'] or '(empty)'}\n")
    print(f"clip {r['clip_seconds']:.1f}s, face found in {r['face_frames']}/{r['frames']} frames | "
          f"crop {r['crop_seconds']:.1f}s, model {r['model_seconds']:.1f}s on {args.device}")
    if crop_path:
        print(f"Mouth crop preview: {crop_path}")
    for w in r["mps_fallbacks"]:
        print(f"MPS fallback to CPU: {w}")


if __name__ == "__main__":
    main()
