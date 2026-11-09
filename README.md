# Silent Chat

Talk to Claude on your Mac without making a sound. Mouth words at your camera, and a local lip-reading model plus
Claude turn them into a conversation, typed text, or actions on your Mac. You can also say "Hey Claude" out loud,
or hold Option to dictate into any app.

> **Prototype.** On its own the lip-reader gets roughly 40–60% of words right (see Results below). Claude's correction
> brings that to about 60–70%, and it improves as it fine-tunes on clips of your own mouth.

## Requirements

* An Apple Silicon Mac with macOS 14 or later (built on macOS 26).
* [Claude Code](https://claude.com/claude-code), installed and signed in with a claude.ai plan (no API key).
  Silent Chat runs `claude -p` in the background, so its use counts toward your plan's limits.
* [uv](https://docs.astral.sh/uv/) and ffmpeg (`brew install uv ffmpeg`), and the Xcode Command Line Tools
  (`xcode-select --install`) to build the menu bar app.

## Install

```bash
git clone <this repo> silent-chat && cd silent-chat && ./setup.command
```

`setup.command` creates the Python environment and downloads the models (about 2.5 GB in total). It then builds
**Silent Chat.app** into `~/Applications`, adds it to Login Items and opens it. macOS asks for permissions for
"Silent Chat":

* **Camera**, **Accessibility** and **Input Monitoring**, on first start.
* **Microphone**, for "Hey Claude", MIC and dictation.
* **Screen Recording**, only if Claude needs a screenshot.
* **Automation of Spotify**.

The app is built and signed on your own Mac, so Gatekeeper doesn't block it. It runs the Python code from this
folder, so if you move the folder, run `.venv/bin/python make_app.py --force` again.

## Privacy

* Lip-reading, face tracking, the "Hey Claude" wake word and speech recognition all run **on your Mac**.
* Clips of your mouth and their transcripts are saved locally in `data/`, to fine-tune the lip-reader on you. They
  are never uploaded, and git ignores them, as it does `models/` (which includes your personal fine-tuned model).
* What does go to Claude, through your Claude Code session:
  * the lip-reader's text guesses, and what you typed or said;
  * what Claude reads on screen when you ask it to act (accessibility text, occasionally a screenshot);
  * if you've connected them in claude.ai, your Todoist and Spotify connectors.
* App tips Claude learns are saved in `skills/`. Check them before you commit, in case they mention people.

See [THIRD_PARTY.md](THIRD_PARTY.md) for the third-party code and models. The lip-reading weights are for
research and non-commercial use only.

## Using it


The goal is to talk to Claude with mouth movements only. It runs as a **menu bar app, "Silent Chat"**
(`~/Applications/Silent Chat.app`), and **opens at login**, so you can tap `\` anytime.
* **Menu bar icon** (a mouth, filled while the overlay shows): Show/Hide, the TYPE / VOICE / FACE / MIC switches,
  Open at Login, Open Console, Restart and Quit.
* **Open Console** (or double-click `silent_chat.command`) opens a Terminal window with the live log, where you can
  type a message or a command (`/fix`, `/learn`, `/status`, `/restart`, `/quit` …). Closing it doesn't stop silent
  chat. Commands reach it through a pipe (`data/silent_chat/commands.fifo`).
* **At login it starts hidden**, with the camera off. Opening the app again (Spotlight, Finder) shows the overlay.
* **Build or update the app:** `.venv/bin/python make_app.py --login`. The app is a tiny native launcher
  (`app/launcher.swift`) that runs `silent_chat.py` from this folder, so **Python changes need no rebuild: use
  Restart in the menu**. If silent chat crashes, the app restarts it (hidden), at most 3 times in 10 minutes.
* **Permissions belong to "Silent Chat"** (not Terminal): Camera, Accessibility and Input Monitoring (asked on
  first start; `\` starts working by itself once they're allowed), and on first use Microphone (MIC), Screen
  Recording (`look`), Automation of Spotify and the Downloads folder (where this project lives). The app is signed
  ad hoc, so **rebuilding it makes macOS ask again**; `make_app.py` only rebuilds when the launcher changes.
* Everything it prints goes to `data/silent_chat/session.log`.

**Say "Hey Claude" (out loud), anytime.** No camera needed, and it works while the overlay is hidden: it pops
up (camera off), Claude answers out loud, and it hides again a few seconds after the conversation goes quiet.
* Say it in one breath ("Hey Claude, what's on my to-do list?") or pause after "Hey Claude" (it chimes) and then
  say your request. After Claude answers, you have 5 s to reply without saying "Hey Claude" again.
* It all runs on this Mac: the mic level shows when someone's speaking, and local Whisper checks only the first ~2 s
  of each utterance for "Hey Claude". Speech without it is dropped right away and never sent anywhere.
* The mic stays on while it's enabled, so macOS shows its orange mic dot. Turn it off with the menu bar item
  "HEY CLAUDE", `/wake`, or by asking Claude ("turn off the wake word"). The setting is kept across restarts
  (`data/silent_chat/settings.json`).
* It doesn't listen while Claude is speaking, so it can't trigger itself through the speakers.

**Hold Option to dictate into any app, with the overlay hidden.** While you hold Option, the menu bar icon becomes a
waveform; let go and the text is typed where your cursor is.
* **Speak out loud** and local Whisper types exactly what you said (about 0.3 s after you let go).
* **Or just mouth it**: the camera turns on only while you hold Option (start mouthing ~0.5 s after pressing,
  while it starts), then the lip-reader plus Claude's clean-up types it (a couple of seconds).
* Taps and Option shortcuts (Option+arrow, Option+letter …) are ignored: a hold only counts if it's longer than
  0.5 s, no other key was pressed, and it heard speech or saw mouthing.
* With the overlay showing, Option works as before (chat, or TYPE mode). `--no-hold-dictation` turns this off.

**Todoist and Spotify, directly.** Silent chat uses the Todoist and Spotify connectors from your claude.ai account
(claude.ai > Settings > Connectors), so Claude reads and changes your tasks and searches Spotify's catalogue
directly instead of opening the apps. `connectors.py` checks which are connected (run it after connecting one,
then Restart). Only the tools voice control needs are loaded, which keeps replies fast.
* Todoist: find tasks (today, overdue, by project), add, complete, uncomplete, reschedule and update. Claude can't
  delete tasks or projects, and if a lip-read task name could match more than one task, it asks first.
* Spotify: search, save to library, generate playlists. Playing goes through `macctl spotify play <uri>` on the
  Spotify app, and "what's playing" through `macctl spotify now`. (The connector's own "currently playing" needs a
  fresh Spotify sign-in, so it isn't used.)
* `--no-connectors` turns this off.

**Just ask Claude to change modes.** "Turn on typing mode", "stop talking out loud", "turn the mic on", "face
gestures off", "hide". Claude switches the setting and confirms. This works in TYPE mode too: mouthing "turn off
typing mode" switches back to chat instead of typing those words. Claude is told the current settings with every
message, including changes you made yourself with the buttons, so it knows what's on.

**Control your Mac: nearly anything you could do.** Ask for things like "open my DMs with Alex", "summarise my
Discord notifications", "put Todoist and Calendar side by side" or multi-step jobs in any app. Claude works through
`macctl` (`mac.py` + `control.py`), cheapest way first, so it's fast and light on tokens:

1. **Direct commands, shortcuts and menus**: `open`, `focus`, `url`, `place`/`arrange`, `spotify …`,
   `menu <App> "File > Export…"` (`menu <App> <Menu>` lists items *with their keyboard shortcuts*), `key cmd+k`,
   `type <text>`. Many tasks are a shortcut plus typing ("open my DMs with Alex" = Cmd+K, "alex", Return).
2. **The accessibility tree**: `ui [App] [--find words]` lists the window's controls as `#ids` plus its visible
   text. It's exact and cheap: all of Discord is ~1,100 tokens, and with `--find` ~100. Read in under 0.1 s. Then
   `press #id` (doesn't move your mouse or take focus), `set #id <text>` (writes into the field directly),
   `click #id`. Chrome and Electron apps (Discord, Slack, VS Code …) are switched into accessibility mode
   automatically.
3. **Screenshots only as a last resort**, for canvases, images and games: `see [App]` saves a ~900-token JPEG and
   `click x y` uses its pixel coordinates (`see --zoom` looks closer).

* **Batching**: `macctl do "focus Discord" "key cmd+k" "type alex" "wait 0.5" "key return"` runs several steps
  in one call (one model round trip instead of five) and stops at the first failure.
* **Cheap checks**: every action prints which app, window and field have focus afterwards, so Claude rarely needs
  to look again. `wait-for <text>` waits for something to appear.
* **App playbooks** in `skills/` (Discord, Chrome, Slack, Finder, Notes, general macOS): `macctl skill <App>`.
  When Claude works out something new about an app it saves a one-line tip (`macctl learn <App> "<tip>"`), so the
  next time is faster. You can edit these files.
* **Safety**:
  * Claude **asks first** (and waits for your yes) before anything that sends, posts, deletes, buys, submits,
    starts a machine, changes settings or can't be undone. Typing a draft is fine; pressing Send isn't until you
    say yes.
  * Password fields are refused, and it never signs in for you.
  * Keys and typing only go to the app it brought to the front. If another app is in front, it sends nothing.
  * **Press `\` while it's working to stop it** (or say "Hey Claude, stop"). Every further action is then refused
    until your next message.
* **Permissions**: Accessibility, which Silent Chat already has. Screenshots (`see`) need **Screen Recording**;
  macOS asks the first time.
* **Testing from the console**: `/macctl <command>` runs macctl with Silent Chat's permissions and prints the
  result to the log. `ui <App> --stats` shows only sizes and counts, no content.

**Show / hide anytime: tap `\`** in any app. Hidden means the overlay is gone, **the camera is off** (its light
goes out) and nothing listens. The model and Claude stay loaded, so it comes back in about a second.
* **Double-tap `\`** to type an actual backslash: the key is swallowed so it can work as a toggle.
* Shift+`\` (`|`) and shortcuts with `\` pass through as normal.
* The overlay's **x** and the menu bar icon also hide it. In the console, `/hide` and `/show` work too.
* `--hidden` starts hidden. `--toggle-key '`'` (or `§`, or `none`) changes the key.
* It needs **Accessibility** and **Input Monitoring** for Silent Chat (System Settings > Privacy & Security); macOS
  asks on first start, and the key starts working as soon as they're allowed.

**Talking**
* **Hold Option** (in any app) while you silently mouth a message, then let go. It keeps 0.4 s from before the
  press. Option only counts when held on its own, so Cmd+Option+… shortcuts don't trigger it. `--hotkey
  right_option` or `--hotkey control` changes the key. If holding Option does nothing, allow Silent Chat under System
  Settings > Privacy & Security > Input Monitoring.
* **Or use face gestures, hands-free** (the **FACE** button turns them on or off; turn them off on a video call):

  | Gesture | Does |
  |---|---|
  | **Long blink** (~0.5 s) | Starts listening, with a soft chime if voice is on. Open your eyes and mouth; it stops after about 1 s without lip movement. Natural blinks (0.1–0.3 s) and drooping eyelids from looking down are ignored. |
  | **Smile**, closed lips (TYPE mode) | Deletes dictated words while you hold it: the first after 0.7 s, then one per second. |
  | **Eyebrow raise** (TYPE mode) | Presses **Enter** to send. A quick "eyebrow flash" looks like a normal friendly expression in public. `--send wink` makes it a wink instead. |

  Gestures don't fire while you're mouthing, or for 1 s afterwards.
* **Interrupt** Claude by holding Option or blinking long while it talks. It stops, and Claude is told how far it got.

**Typing anywhere (TYPE mode).** Click **TYPE** in the overlay (or start with `--type`, or type `/type`). What you
mouth is typed into whatever app is in front (messages, email, documents, search boxes), as if from the keyboard.
* Hold Option (or blink long) and mouth a phrase. Claude turns the lip-reading into just the text: punctuation, and
  capitals that continue what you've already typed there. Spoken "new line", "comma", "period" and so on become the
  symbols.
* The text is **typed word by word as it streams in**: first words about 1.5 s after you let go.
* **Smile to delete** dictated words, one per second while you hold it. **Raise your eyebrows to send** (Enter).
  Deleting only removes words it typed into the current app, never text that was already there. Enter only fires
  when there's something you dictated there that hasn't been sent yet.
* It needs **Accessibility** permission for Silent Chat, so it can send key presses (System Settings > Privacy &
  Security > Accessibility). Click into the app you want to type into first.
* Phrases you send or keep unedited are marked "ok" and become training data. Ones you edit by smiling stay
  unconfirmed.

**MIC: speak out loud to correct it.** Click **MIC** in the overlay (or `/mic`, or start with `--mic`) and speak
normally while holding Option. Speech recognition is far more reliable than lip-reading for names and numbers, so
this is the quick fix when it gets something wrong ("No, I meant Todoist").
* **Transcription is local**: Whisper large-v3-turbo on your GPU via MLX, about 0.25 s per sentence. Nothing leaves
  the Mac. It's biased towards your personal vocabulary.
* It uses the **MacBook's built-in microphone** (`--mic-device` to change it). Using the AirPods mic would switch them
  to low-quality call audio. The mic is only open while MIC is on, and it's closed whenever the overlay is hidden.
  macOS shows its orange mic dot while it's open.
* Claude takes what you said as given. If it was a correction, the earlier message is relabelled.
* **Names and rare words you say out loud join your vocabulary** (e.g. Todoist), so silent lip-reading gets them
  next time.
* Clips recorded with the mic on are **labelled from your voice** (status `audio`), which makes them reliable
  training data for the lip-reader.
* Mouthing silently still works while MIC is on: with no speech in the audio, it falls back to lip-reading.
* The first time, macOS asks to allow Silent Chat to use the **Microphone**.

**Numbers and spelling** are the weakest spot for any lip-reader. "-teen" and "-ty" look the same
(thirteen/thirty), "twenty-three" tends to collapse to "three", and single letters (B/P/M, D/T/N) are
indistinguishable. What helps:
* Say numbers **digit by digit** when they matter: "two three gauge", "two two zero volts".
* Spell with the **NATO alphabet**: "November India Sierra Sierra Alpha November".
* Claude knows these failure modes and says the number it's assuming ("Assuming 23-gauge: ..."), so a wrong number
  is easy to catch and correct.
* For exact values, type them in the console.
* Train it with the practice drill (below).

**The floating overlay**
* A small panel that stays **above every app, including full-screen ones** (like Zoom's floating controls). It
  never takes keyboard focus. **Drag it anywhere; it snaps to the nearest screen edge** and remembers where you
  left it.
* It's built from **Liquid Glass** cards (macOS 26's glass material, frosted glass on older systems): a face card,
  a row of glass pills, and a transcript card.
* The face card shows **your face as glowing line art**, drawn from face landmarks, not video: eyebrows, eyes with
  irises, nose, lips, and topographic contour rings. Because eyes and brows are inputs, they visibly react: a blink
  closes the drawn eyes, and raised brows lift and brighten. The lips follow your mouthing. "NO SIGNAL" means the
  camera can't see you. The pill in the bottom right shows the state: LIVE, LISTENING, BLINK SEEN, THINKING,
  SPEAKING.
* Buttons: **MIC** (speak out loud, below), **TYPE** (chat with Claude, or type into the app in front), **VOICE** (spoken replies or text only),
  **FACE** (face gestures on/off) and **x** (hide; tap `\` to bring it back). The conversation is
  shown below.
* Spoken replies go to your **AirPods whenever they're connected to this Mac**, otherwise to the default output.
  The log prints which device it's using. `--audio Speakers` forces the built-in speakers.
* In the console (menu bar > Open Console) you can also **type** a message, `/fix <what you actually said>`,
  `/learn <word, word>`, `/voice`, `/face`, `/type`, `/mic`, `/hide`, `/show`, `/restart` or `/quit`.

**Speed.** One `claude` process stays running for the whole chat, so there's no start-up per turn and it keeps the
conversation context. It streams the reply, and speech starts as soon as the first sentence is complete. Claude runs
at **low effort** (`--effort`): about twice as fast, and as accurate on the test sentences. The lip-reader uses
**beam 10** (`--beam`): about three times faster than 40, with the same accuracy. Clips are saved in the
background. Measured: **first words about 1 s after the lip-reading finishes** (previously about 7 s), and
lip-reading takes about 0.15 s per second of mouthing.

**Usage.** The Claude calls go through the `claude` CLI signed in to your claude.ai Pro plan. They count toward
your plan's usage limits; there are no per-call charges unless you've turned on paid extra usage.

### Self-improvement

It learns from your sessions in two ways:

1. **Right away, through your personal memory** (`data/silent_chat/personal.json`). When you correct it, either
   naturally ("no, I said Nissan", which Claude detects) or with `/fix …`, the misreading and any new words go into
   a memory that Claude sees every turn. The next time the lip-reader produces something similar, Claude knows what
   you mean. `/learn Nissan, Sarah` adds names or words directly.
2. **Retraining the lip-reader on your mouth** (`train_personal.py`). Every clip is saved with its text and a status
   (`data/silent_chat/labels.csv`):
   * `confirmed`: you picked or typed it
   * `fixed`: you corrected it
   * `ok`: you carried on without correcting it
   * `heard`: not confirmed yet

   Clips marked `confirmed`, `fixed`, `ok` or `audio` are training data. Once there are at least 30, plus 15 new
   ones since the last round, it retrains **in the background** whenever you hide silent chat with `\\` (you're not
   using it then) or quit it. The retrain runs as its own low-priority process, logged to
   `data/silent_chat/training.log`. If it keeps a better model, silent chat switches to it the next time you show it,
   with no restart needed. It always starts from the original checkpoint, freezes the visual front-end, and **keeps the new model only if it
   makes fewer word errors** on clips it never trained on: your 25 `mouthed` test recordings plus a fixed ~15 % of
   your session clips. The kept model is `models/personal.pth`; scores and history are in `models/personal.json`. To
   run it by hand:

   ```bash
   .venv/bin/python train_personal.py --force
   ```

3. **Practice drills** (`practice.py`) target the weak spots. It shows number and spelling phrases ("nineteen not
   ninety", "two four gauge wire", "Canberra: Charlie Alpha November ..."), plus the rare words in your vocabulary.
   You mouth them in a camera window (hold SPACE), and every clip is saved with its exact text, which makes it
   perfect training data. It retrains at the end if there's enough new data. Close the chat first, then run one of:

   ```bash
   .venv/bin/python practice.py
   ```

   ```bash
   .venv/bin/python practice.py numbers -n 30
   ```

Retraining needs volume: expect real gains after a few hundred varied clips, not a few dozen. Corrections matter
most, because they are exactly the cases the model gets wrong.

**`--confirm` mode** is the earlier flow: hold SPACE in the camera window, then pick 1/2/3 (or T to type). Its
labels are all confirmed, so it collects the cleanest training data.

Limits of this prototype:
* Claude can talk but has no tools here, so it can't set reminders or send messages.
* Silent mouthing only. The minimal-movement style is not readable by a camera (see results).

## The lip-reading feasibility test (phase 1)

The project started as a feasibility test of camera-only lip-reading on this Mac. To run it:

All commands are run from this folder (`silent-speech/`).

Quick check (records one clip, transcribes it, and saves a video of the mouth crop the model sees):

```bash
.venv/bin/python vsr.py --record --save-crop
```

The full test, with all three conditions in one session (25 sentences each, from `sentences.txt`).
Double-click `run_test.command`, or run this in Terminal:

```bash
.venv/bin/python run_test.py
```

Each part starts with an instruction screen (press ENTER), then shows each sentence in large yellow text.
Every take is checked as soon as you stop: if your face wasn't tracked or the mouth couldn't be cropped,
it asks you to say that sentence again. After all 75 clips are recorded, it transcribes them (a few minutes)
and prints a comparison table. You can also run a single condition:

```bash
.venv/bin/python run_test.py --condition mouthed
```

| condition | what to do |
|---|---|
| `normal_muted` | speak out loud normally (baseline; only video is recorded, so audio is "muted" by design) |
| `mouthed` | silently mouth the words with clear lip movement |
| `minimal` | your natural silent style, lips barely moving, mostly tongue |

**Camera window controls:** **hold `SPACE` while speaking** (push-to-talk), or tap `SPACE` to start and tap again to stop.
`ESC` discards the take and lets you retake, and `Q` quits.
The preview is mirrored and the saved clip is not. Keep your face inside the grey box, close to the camera, with even lighting.

**Run the recording commands from the macOS Terminal app.** The first time, macOS asks to allow Terminal to use the camera.
Claude's built-in shell and terminal panel can't show that permission prompt, so camera access fails there.
`record_test.command` is a double-clickable shortcut for the quick check.

**Resuming and redoing:** clips are saved to `recordings/<condition>/NN.mp4`. Re-running the same command
records only the missing clips, so you can quit with `Q` and continue later. Other options:

* `--redo 3,7` re-records sentences 3 and 7
* `--fresh` re-records everything
* `--no-record` only re-scores existing clips (for example with `--beam-size 10` or `--device cpu`)

**Output:** a per-sentence table (reference vs prediction, WER) and overall **WER**, **word accuracy (1 − WER)**,
words correct, and exact sentences. `results/<condition>.csv` holds the per-sentence detail, and each run
appends one line to `results/summary.csv` so you can compare conditions.

Text is lower-cased and stripped of punctuation before scoring. The model outputs words, not digits,
so numbers in `sentences.txt` are spelled out.

## Speed (measured on this Mac)

5-second clip, beam size 40:

| `--device` | encoder | beam search | notes |
|---|---|---|---|
| `cpu` | 0.75 s | 2.6 s | |
| `mps` | 0.04 s | 5.9 s | every op runs on MPS (no CPU fallbacks), but beam search is many tiny ops and is slow on the GPU |
| `hybrid` (default) | 0.04 s | 2.6 s | encoder on MPS, beam search on CPU; faster than real time |

`--beam-size 10` cuts the beam search to about 1 s, usually with a small accuracy cost.
Face detection and cropping add about 0.1 s per second of video.

## How it works

1. The video is decoded with ffmpeg and resampled to 25 fps, the rate the model was trained on.
2. The MediaPipe face detector finds 4 keypoints per frame (eyes, nose tip, mouth centre).
   Missed frames are interpolated.
3. Auto-AVSR's own `VideoProcess` aligns each frame to a mean face and cuts a 96×96 mouth crop.
4. The crop is center-cropped to 88×88, converted to grayscale, and normalised.
5. The model runs a ResNet front-end and a Conformer encoder, then hybrid CTC/attention beam search
   (beam 40, CTC weight 0.1, no external language model, the same as Auto-AVSR's inference tutorial).

## Deviations from the upstream repo

* **MediaPipe API ported.** Auto-AVSR's MediaPipe detector uses the legacy `mp.solutions.face_detection`. That API is
  gone from current MediaPipe, and the old versions that had it no longer ship Mac wheels on PyPI. `vsr.py` uses the
  MediaPipe Tasks `FaceDetector`, which returns the same keypoints, with the short-range BlazeFace model
  (`models/blaze_face_short_range.tflite`). That model is meant for faces within about 2 m, which suits a laptop camera.
  The legacy code also tried a full-range model first, but that one isn't published for the Tasks API.
* **One-line patch** in `third_party/auto_avsr/espnet/nets/ctc_prefix_score.py`: the CTC scorer chose `cuda` or `cpu`
  only. It now uses the input tensor's device, so MPS works.
* No `pytorch_lightning` or `torchaudio`: the small pieces needed for inference are reimplemented in `vsr.py`.
* **Checkpoint source.** The Imperial College download links (`doc.ic.ac.uk/~pm4115/autoAVSR/...`, including the
  demo video) all return 404 now. The checkpoint came from the Google Drive link in the Auto-AVSR README
  (file id `1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_`). Its MD5 is `49f770f2c0d8b8d769347ee47ed1648f`, which matches the
  README's `49f77` prefix.

## Recreating the setup

`setup.command` does all of this. No sudo is needed.

* `uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -r requirements.txt`
* Clones [Auto-AVSR](https://github.com/mpc001/auto_avsr) into `third_party/auto_avsr` at commit `182b628`
  and applies `patches/auto_avsr-ctc-device.patch`.
* Downloads the lip-reading weights with `gdown 1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_`, plus MediaPipe's
  `blaze_face_short_range.tflite` and `face_landmarker.task`, into `models/`.
* `make_app.py --login` builds the menu bar app.

## Licensing

* Auto-AVSR **code** is Apache-2.0.
* The **pretrained weights** were trained on LRS3, LRS2, VoxCeleb2 and AVSpeech. Those datasets are licensed for
  research and non-commercial use (LRS2/LRS3 come from BBC/TED material under academic agreements, and
  VoxCeleb2/AVSpeech are YouTube-derived). Treat the model as **research-only**. A commercial product would need
  different training data or a licensing arrangement.

## Results: first run (2026-11-07)

These come from 25 sentences per condition, with the built-in camera in a dim room, the hybrid device and beam 40.
The full per-sentence detail is in `results/*.csv`. The last three rows of `results/summary.csv` are the current scoring,
which spells out numbers. The earlier rows counted "730" as wrong for "seven thirty".

| condition | WER | word accuracy | words correct | exact sentences | sentences ≤ 20 % WER |
|---|---|---|---|---|---|
| normal_muted | 48.1 % | 51.9 % | 102/162 (63 %) | 7/25 | 11/25 |
| mouthed | 59.9 % | 40.1 % | 76/162 (47 %) | 4/25 | 6/25 |
| minimal | 92.0 % | 8.0 % | 23/162 (14 %) | 0/25 | 0/25 |

A 20-second free-form spoken monologue recorded earlier came out word for word.
Short, unrelated commands are much harder for the model than long, flowing speech.

### With LLM post-correction (`llm_correct.py`)

The lip-reader's top-10 hypotheses for each clip go to Claude Opus 5.5, which replies with the sentence the user most
likely said. Claude is told only that this is lip-reader output for a request to an AI assistant. It never sees the
reference sentences, which are used only for scoring.

| condition | lip-reader top-1 | + Claude correction | best of top-10 (oracle, uses the answer key) |
|---|---|---|---|
| normal_muted | 51.9 % (102/162 words) | **69.1 %** (123/162), 14/25 exact | 70.4 % |
| mouthed | 40.1 % (76/162) | **59.3 %** (115/162), 9/25 exact | 58.6 % |
| minimal | 8.0 % (23/162) | 8.6 % (28/162) | 15.4 % |

* The correction adds about 17–19 points of word accuracy wherever the lip-reader has something to work with.
  For example, "stay on top of it for ten months" became "set a timer for ten minutes", and "some crises are uniformly"
  became "summarize this article for me". For mouthed speech it beats the oracle, because Claude can write words
  that none of the hypotheses contain.
* It can't rescue a hypothesis list that has drifted into an unrelated sentence, and it can't recover names
  ("Sarah" became "the restaurant"). Minimal-movement speech stays unreadable.
* **Caveat:** these test sentences are prototypical assistant commands, which is exactly what an LLM expects. Real
  requests with names, unusual content or longer dictation will likely gain less. On the other hand, a real assistant
  would also have the conversation so far as extra context.
* Speed: 6.2 s per correction through the `claude` CLI (Opus 5.5, default effort), including CLI start-up.
  A direct API call at low effort would be much faster. Cost was about $1.20 at API prices for all 75 clips.

To re-run it, use the command below. It needs the `claude` CLI to be logged in, and the outputs are cached in `results/`.

```bash
.venv/bin/python llm_correct.py
```

## What to expect

* `normal_muted` should land roughly at benchmark level or somewhat worse (benchmark is 20 % WER on TED talks).
  The sentences here are short commands, which gives less context per sentence.
* `mouthed` is usually worse: silent articulation is smaller and lacks the timing cues of voiced speech.
* `minimal` will likely fail badly, because a camera can't see the tongue. Confirming that is the point of the condition.
* Random or unclear input tends to decode to fluent but wrong TED-style phrases (for example
  "and that's what we're going to do"). That is the decoder's language prior, so watch for it in the results.
