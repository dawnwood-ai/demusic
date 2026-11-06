# demusic

[![tests](https://github.com/dawnwood-ai/demusic/actions/workflows/test.yml/badge.svg)](https://github.com/dawnwood-ai/demusic/actions/workflows/test.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![MCP server](https://img.shields.io/badge/MCP-server-6f42c1.svg)](#ai-agents-mcp)

**demusic removes the background music that AI video models (Seedance, Kling, Veo, Hailuo,
Wan, ...) bake into generated clips, and keeps the dialogue, sound effects and ambience.
It finds *where* the music is and only replaces the audio there, so the rest of the clip
is untouched. Local, free, CLI + MCP server.**

[Read in Chinese](README.zh-CN.md) · [How detection works](docs/how-it-works.md)

<!-- mcp-name: io.github.dawnwood-ai/demusic -->

## Why

Native-audio video models often add a music bed nobody asked for: a few seconds of score
at the start of a shot, a pad under the dialogue. Cut ten such clips together and you get
ten pieces of music fighting each other, and you can't lay your own score over it.

The usual fix - run a vocal remover and keep only the vocals - throws away footsteps,
doors, rain and room tone along with the music, and it is wrong on every clip that had
no music to begin with. demusic is what we run on every clip of our AI dramas at
[Dawnwood](https://dawnwood.ai):

1. **Separate** the audio into stems (Demucs `htdemucs_6s`, locally or on your fal key).
2. **Is it loud enough to be a bed?** The melodic stems must be audible (> -50 dBFS).
3. **Is it actually music?** An AudioSet classifier must hear music on the accompaniment
   for 2+ seconds. Engine hum, gunfire and room tone pass step 2 but fail this one.
4. **Strip only those spans**: there the audio becomes dialogue + percussion/impacts, with
   a 0.25 s crossfade; everywhere else the original audio stays. The picture is
   stream-copied (no re-encode).

It is deliberately conservative. On 164 clips with no music it wrongly flagged 2; on 87
clips a person had flagged by ear it found 33. The misses are very quiet beds - lower
`--threshold` or use `--mode force` when you know a clip has music.
[Calibration details](docs/how-it-works.md).

## Install

Requires Python >= 3.10 and **ffmpeg** on `PATH`.

```bash
pip install "demusic[demucs]"     # local separation (pulls in PyTorch; CPU is fine)
# or, no torch, separation on fal.ai with your own key:
pip install "demusic[fal]" && export FAL_KEY=...
```

The music classifier (~90 MB ONNX, BSD-3) downloads once to `~/.cache/demusic`.

## Quickstart

```bash
# where is the music?
demusic check shot_012.mp4
# shot_012.mp4: MUSIC  spans 0-2.8s  melodic -42.1 dBFS  p=0.34

# write shot_012.demusic.mp4 with the music removed
demusic strip shot_012.mp4

# a whole folder; clean copies of every file (unchanged ones copied) into clean/
demusic strip renders/*.mp4 --out-dir clean/
```

| option | meaning |
|---|---|
| `--separator demucs\|fal\|DIR` | local Demucs (default), fal.ai with `FAL_KEY`, or a folder of stems you already separated (`vocals.wav`, `drums.wav`, `bass/guitar/piano/other.wav` or `music.wav`) |
| `--mode spans\|whole\|force` | replace only where music was heard (default) / the whole track if any music was heard / always, without detection |
| `--threshold 0.3` | classifier music probability; `0.2` finds quieter beds at a higher false-positive rate |
| `--loud-floor -50` | melodic-stem loudness gate (dBFS) |
| `-o FILE` / `--out-dir DIR` | output path for one file / for a batch |
| `--json` | one JSON object on stdout (`files[]` with `music`, `spans`, `melodic_dbfs`, `strength`, `action`, `output`) |

Exit codes: `0` ok (for `check`: no music), `1` `check` found music, `2` usage, `3` failure.
`demusic check` in CI or a render pipeline = fail the build when a clip has a music bed.

Speed: separation is roughly real time on a laptop CPU (a 10 s clip ~10-20 s), much faster
on a GPU. Detection itself is a few hundred ms.

## Python

```python
from demusic.core import process
r = process("shot.mp4", "shot.clean.mp4")          # strip
r = process("shot.mp4", strip=False)                # check only
print(r["music"], r["spans"])
```

## AI agents (MCP)

```bash
claude mcp add demusic -- demusic mcp            # Claude Code
# or any MCP client: command "uvx", args ["--from", "demusic[demucs]", "demusic", "mcp"]
```

Tools: `check_music` (read-only) and `remove_music` (writes a new file, never overwrites the
input). Set `DEMUSIC_SEPARATOR=fal` in the server env to use fal. There is also a
[Claude skill](skills/demusic/SKILL.md) and [AGENTS.md](AGENTS.md).

## Limits

- Music that is *meant* to be in the scene (a band on screen, a radio) is removed too when
  it is detected. Skip those clips.
- Inside a stripped span, non-percussive effects (a running tap, wind) go with the music;
  Demucs has no separate SFX stem. Outside the spans nothing changes.
- Sung vocals land in the vocals stem and are kept.

## Credit

Apache-2.0. demusic writes `cleaned with demusic by Dawnwood` into each output file's
`comment` tag. If it saves you time, please keep that and mention
"demusic by [Dawnwood](https://dawnwood.ai)" - a request, not a condition (see [NOTICE](NOTICE)).

Made by [Dawnwood](https://dawnwood.ai), an AI drama studio. More tools we use in
production: [dramapack](https://github.com/dawnwood-ai/dramapack) (short-drama delivery
packs and QC) · [all our open source](https://dawnwood.ai/open-source/).
