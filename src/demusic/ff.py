"""ffmpeg helpers.

The binary comes from ``$DEMUSIC_FFMPEG`` when set, otherwise from ``PATH``. Every
subprocess has a timeout: a truncated mp4 can make ffmpeg read forever, and a batch job
that hangs silently is worse than one that fails.
"""
import os
import re
import shutil
import subprocess
import sys

import numpy as np

TIMEOUT = 1800


def log(*args) -> None:
    """Progress goes to stderr so stdout stays clean for ``--json`` and the MCP protocol."""
    print(*args, file=sys.stderr, flush=True)


def exe() -> str:
    return os.environ.get("DEMUSIC_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"


def available() -> bool:
    e = exe()
    return bool(shutil.which(e) or os.path.isfile(e))


def run(args: list, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([exe(), "-hide_banner", "-v", "error", "-y", *args],
                          capture_output=True, check=True, timeout=TIMEOUT, **kw)


def _info(path: str) -> str:
    return subprocess.run([exe(), "-hide_banner", "-i", path], capture_output=True,
                          text=True, timeout=120).stderr


def has_audio(path: str) -> bool:
    return bool(re.search(r"Stream #.*Audio:", _info(path)))


def has_video(path: str) -> bool:
    return bool(re.search(r"Stream #.*Video:", _info(path)))


def duration(path: str) -> float:
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", _info(path))
    if not m:
        return 0.0
    h, mm, s = m.groups()
    return int(h) * 3600 + int(mm) * 60 + float(s)


def mean_volume(path: str):
    """mean_volume (dBFS) via ffmpeg volumedetect. None on silence/failure."""
    try:
        r = subprocess.run([exe(), "-hide_banner", "-i", path, "-af", "volumedetect", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=600)
    except subprocess.SubprocessError:
        return None
    m = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", r.stderr or "")
    return float(m.group(1)) if m else None


def pcm(path: str, sr: int, channels: int) -> np.ndarray:
    """Decode any audio/video file to float32 PCM, shape (n,) for mono or (n, channels)."""
    raw = subprocess.run([exe(), "-v", "error", "-i", path, "-vn", "-ac", str(channels), "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True, check=True, timeout=TIMEOUT).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    return x if channels == 1 else x.reshape(-1, channels)


def write_wav(x: np.ndarray, sr: int, out: str) -> str:
    """Write float32 PCM (n,) or (n, ch) to a 16-bit wav."""
    ch = 1 if x.ndim == 1 else x.shape[1]
    subprocess.run([exe(), "-y", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", str(ch), "-i", "-",
                    "-c:a", "pcm_s16le", out], input=np.ascontiguousarray(x, dtype=np.float32).tobytes(),
                   check=True, capture_output=True, timeout=TIMEOUT)
    return out


def mix(paths: list, out: str, sr: int = 48000) -> str:
    """Sum audio files (no normalisation: stems add back up to the original level)."""
    if len(paths) == 1:
        run(["-i", paths[0], "-ar", str(sr), "-ac", "2", out])
        return out
    args = []
    for p in paths:
        args += ["-i", p]
    args += ["-filter_complex", f"amix=inputs={len(paths)}:duration=longest:normalize=0[a]",
             "-map", "[a]", "-ar", str(sr), "-ac", "2", out]
    run(args)
    return out
