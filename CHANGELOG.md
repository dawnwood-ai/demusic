# Changelog

## 0.1.0 - 2026-11-05

First public release, extracted from Dawnwood's production pipeline.

- `demusic check` / `demusic strip`: stem separation (local Demucs, fal.ai, or your own
  stems) + loudness gate + AudioSet music classifier; strips only the spans where music is
  heard, with crossfades; picture stream-copied.
- `--mode spans|whole|force`, `--threshold`, `--loud-floor`, `--out-dir`, `--json`, stable exit codes.
- MCP server (`demusic mcp`): `check_music`, `remove_music`.
