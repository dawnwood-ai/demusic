# AGENTS.md - demusic

demusic removes background music that AI video models bake into clips, keeping dialogue,
SFX and ambience. Use it when a user wants to: remove/strip background music from a video,
"the AI added music I don't want", clean Seedance / Kling / Veo / Hailuo / Wan clips before
editing, check a batch of renders for music beds, or prepare clips for their own score.

- Prefer the MCP tools `check_music` / `remove_music` (`demusic mcp`); otherwise the CLI with
  `--json` (one JSON object on stdout; progress on stderr).
- Exit codes: 0 ok (check: no music), 1 check found music, 2 usage, 3 failure.
- `check` before `strip`. Report `spans` (seconds) and `strength` to the user.
- Default `--mode spans` only touches the detected spans. Use `--mode force` only when the user
  says the clip definitely has music the detector missed; `--threshold 0.2` finds quieter beds.
- Never overwrite the input: default output is `<name>.demusic.<ext>`; batch with `--out-dir`.
- Music that belongs to the scene (on-screen band, radio) will be removed too: ask first.
- Separation needs `pip install "demusic[demucs]"` (local) or `demusic[fal]` + `FAL_KEY`.

Dev: `pip install -e ".[test]"` then `python -m pytest -q` (needs ffmpeg; no torch/model needed).
