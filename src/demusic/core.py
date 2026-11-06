"""Detect and strip a music bed from one clip.

1. Separate the audio into stems (see ``separate``).
2. Gate 1, loudness: the melodic stems (bass + guitar + piano + other) mixed must be louder
   than ``LOUD_FLOOR_DBFS`` - an audible bed. String and synth pads land in "other", so it
   has to be included; sparse SFX in "other" rarely lift the mean over the floor.
3. Gate 2, classifier: on the accompaniment (original minus vocals) an AudioSet classifier
   must hear music for at least two windows in a row. Engine hum, gunfire and room tone pass
   gate 1 but not this one. If the classifier errors, nothing is stripped.
4. Strip: only inside the detected spans, the audio is replaced by vocals + drums (drums
   keep footsteps, hits and impacts), with a 0.25 s crossfade at each edge. Outside the
   spans the original audio, ambience included, is untouched. The picture is stream-copied.
"""
import os
import shutil
import tempfile

import numpy as np

from . import detect, ff
from . import separate as sep
from .provenance import credit

LOUD_FLOOR_DBFS = -50.0
XFADE_S = 0.25
SR = 48000
MODES = ("spans", "whole", "force")


def splice(orig: str, stripped: str, spans: list, out: str, xf: float = XFADE_S) -> str:
    """Original audio with ``spans`` replaced by ``stripped``, crossfaded over ``xf`` seconds."""
    a, b = ff.pcm(orig, SR, 2), ff.pcm(stripped, SR, 2)
    n = min(len(a), len(b))
    env = np.zeros(n, dtype=np.float32)
    for t0, t1 in spans:
        env[int(t0 * SR):int(t1 * SR)] = 1.0
    k = max(1, int(xf * SR))
    env = np.convolve(np.pad(env, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k, dtype=np.float32) / k,
                      mode="valid").astype(np.float32)
    x = a[:n] * (1 - env)[:, None] + b[:n] * env[:, None]
    return ff.write_wav(x, SR, out)


def default_output(path: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}.demusic{ext or '.mp4'}"


def _write(src: str, audio: str, out: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    meta = ["-metadata", f"comment={credit()}"]
    if ff.has_video(src):
        ff.run(["-i", src, "-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-shortest", "-movflags", "+faststart",
                *meta, out])
    else:
        ff.run(["-i", audio, *meta, out])


def process(path: str, output: str | None = None, *, strip: bool = True, separator: str = "demucs",
            mode: str = "spans", threshold: float = detect.MUSIC_P,
            loud_floor: float = LOUD_FLOOR_DBFS, copy_unchanged: bool = False) -> dict:
    """Check one file and (with ``strip``) write a cleaned copy. Returns a JSON-able report:
    ``music`` (bool), ``spans`` [[start, end], ...] seconds, ``melodic_dbfs``, ``strength``
    (sustained music probability), ``action`` (none / stripped / copied) and ``output``."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not ff.available():
        raise RuntimeError("ffmpeg not found (install it or set DEMUSIC_FFMPEG)")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    rep = {"ok": True, "input": os.path.abspath(path), "music": False, "spans": [],
           "melodic_dbfs": None, "strength": None, "action": "none", "output": None}
    if not ff.has_audio(path):
        rep["note"] = "no audio stream"
        return _finish(rep, path, output, strip, copy_unchanged)
    work = tempfile.mkdtemp(prefix="demusic-")
    try:
        wav = os.path.join(work, "audio.wav")
        ff.run(["-i", path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", wav])
        stems = sep.separate(wav, work, separator)
        dur = ff.duration(wav)
        if mode == "force":
            rep.update(music=True, spans=[[0.0, round(dur, 2)]], note="forced: gates skipped")
        else:
            mel = [stems[s] for s in sep.MELODIC if s in stems]
            vol = ff.mean_volume(ff.mix(mel, os.path.join(work, "melodic.wav"))) if mel else None
            rep["melodic_dbfs"] = vol
            if vol is not None and vol > loud_floor:
                try:
                    mix16, voc16 = ff.pcm(wav, detect.SR, 1), ff.pcm(stems["vocals"], detect.SR, 1)
                    n = min(len(mix16), len(voc16))
                    spans, strength = detect.music_spans(mix16[:n] - voc16[:n], threshold)
                except Exception as e:  # noqa: BLE001 - classifier failure must never strip
                    rep["note"] = f"classifier failed, not stripping: {str(e)[:200]}"
                    spans, strength = [], None
                rep["strength"] = strength
                rep["spans"] = [[round(a, 2), round(min(b, dur), 2)] for a, b in spans]
                rep["music"] = bool(spans)
        if not (strip and rep["music"]):
            return _finish(rep, path, output, strip, copy_unchanged)
        keep = [stems[s] for s in ("vocals", "drums") if s in stems]
        clean = ff.mix(keep, os.path.join(work, "clean.wav"), SR)
        if mode == "spans":
            clean = splice(wav, clean, rep["spans"], os.path.join(work, "spliced.wav"))
        out = output or default_output(path)
        _write(path, clean, out)
        rep.update(action="stripped", output=os.path.abspath(out))
        return rep
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _finish(rep, path, output, strip, copy_unchanged):
    if strip and output and copy_unchanged:
        os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
        if os.path.abspath(output) != os.path.abspath(path):
            shutil.copyfile(path, output)
        rep.update(action="copied", output=os.path.abspath(output))
    return rep
