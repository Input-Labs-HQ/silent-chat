# Third-party code and models

None of these are included in this repository; `setup.command` downloads them.

| What | From | License / terms |
|---|---|---|
| Auto-AVSR code | [mpc001/auto_avsr](https://github.com/mpc001/auto_avsr), commit `182b628` | Apache-2.0. Modified by `patches/auto_avsr-ctc-device.patch` (CTC scoring uses the input's device, so it runs on Apple Silicon/MPS). |
| Lip-reading weights `vsr_trlrs2lrs3vox2avsp_base.pth` | Auto-AVSR authors | Trained on LRS2, LRS3, VoxCeleb2 and AVSpeech, which are licensed for **research and non-commercial use**. Treat the model as research-only. |
| Face detector (BlazeFace) and Face Landmarker models | [Google MediaPipe](https://ai.google.dev/edge/mediapipe) | Apache-2.0 |
| Whisper large-v3-turbo (MLX conversion) | OpenAI Whisper, via [mlx-community](https://huggingface.co/mlx-community/whisper-large-v3-turbo) and mlx-whisper | MIT. Downloaded on first use of the microphone. |
| Python packages | `requirements.txt` | Each under its own license. |

Claude is used through your own Claude Code installation and account, under Anthropic's terms.
