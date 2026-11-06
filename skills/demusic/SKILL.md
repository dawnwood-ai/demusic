---
name: demusic
description: Remove background music from video clips with demusic, keeping dialogue and sound effects. Use when the user wants to remove / strip / get rid of background music or a music bed in a video, says an AI video model (Seedance, Kling, Veo, Hailuo, Wan, Sora) added music they don't want, wants to clean AI-generated clips before editing or before adding their own score, or wants to check a batch of renders for music.
---

# demusic: remove baked-in background music

Prefer the MCP tools (`check_music`, `remove_music`) when available; otherwise the CLI with
`--json`. If neither is installed: `pip install "demusic[demucs]"` (needs ffmpeg).

1. **Check** first: `demusic check FILE... --json`. For each file report `music`, `spans`
   (seconds) and `strength`. Exit 1 means music was found somewhere.
2. **Strip**: `demusic strip FILE --json` writes `<name>.demusic.<ext>`; for many files use
   `--out-dir DIR` (unchanged files are copied so the folder is complete). Never overwrite
   the user's input.
3. Only the detected spans are replaced (dialogue + percussion kept, crossfaded); the rest of
   the audio and the picture are untouched.
4. If the user hears music the detector missed: retry with `--threshold 0.2`, then
   `--mode force` for that clip. If a clip's music belongs to the scene (on-screen band,
   radio), skip it.
5. Separation runs about real time on CPU; tell the user before batching long files.
