"""Stem separation backends. Every backend returns {stem: wav_path} with Demucs 6-stem names:
vocals, drums, bass, guitar, piano, other.

- ``demucs``: local Demucs ``htdemucs_6s`` (``pip install "demusic[demucs]"``; CPU works,
  roughly real time; a GPU is used automatically when torch sees one).
- ``fal``: Demucs on fal.ai with your own ``FAL_KEY`` (``pip install "demusic[fal]"``). No
  torch locally; the audio is uploaded to fal.
- a directory: stems you already separated (UVR, Demucs, anything) as ``vocals.wav``,
  ``drums.wav`` and any of ``bass/guitar/piano/other.wav`` - or ``music.wav`` /
  ``instrumental.wav`` for a two-stem split.
"""
import os
import subprocess
import sys
import urllib.request

STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")
MELODIC = ("bass", "guitar", "piano", "other")
DEMUCS_MODEL = "htdemucs_6s"
FAL_MODEL = "fal-ai/demucs"
_EXT = (".wav", ".flac", ".mp3")


def from_dir(d: str) -> dict:
    out = {}
    for f in sorted(os.listdir(d)):
        stem, ext = os.path.splitext(f)
        stem = stem.lower()
        if ext.lower() not in _EXT:
            continue
        if stem in ("music", "instrumental", "no_vocals", "accompaniment"):
            stem = "other"
        if stem in STEMS and stem not in out:
            out[stem] = os.path.join(d, f)
    if "vocals" not in out:
        raise ValueError(f"{d}: needs at least vocals.wav plus the music stems")
    return out


def demucs(wav: str, work: str) -> dict:
    try:
        import demucs  # noqa: F401
    except ImportError as e:
        raise RuntimeError('local separation needs Demucs: pip install "demusic[demucs]" '
                           "(or use --separator fal, or pass a folder of stems)") from e
    r = subprocess.run([sys.executable, "-m", "demucs.separate", "-n", DEMUCS_MODEL, "-o", work, wav],
                       capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(f"demucs failed: {(r.stderr or '')[-400:]}")
    d = os.path.join(work, DEMUCS_MODEL, os.path.splitext(os.path.basename(wav))[0])
    return from_dir(d)


def fal(wav: str, work: str) -> dict:
    if not os.environ.get("FAL_KEY"):
        raise RuntimeError("--separator fal needs FAL_KEY in the environment")
    try:
        import fal_client
    except ImportError as e:
        raise RuntimeError('--separator fal needs: pip install "demusic[fal]"') from e
    url = fal_client.upload_file(wav)
    res = fal_client.subscribe(FAL_MODEL, arguments={"audio_url": url, "model": DEMUCS_MODEL,
                                                     "stems": list(STEMS), "output_format": "wav"})
    urls = {}
    for container in (res, res.get("stems") or {}):
        if isinstance(container, dict):
            for s in STEMS:
                f = container.get(s)
                u = f.get("url") if isinstance(f, dict) else (f if isinstance(f, str) else None)
                if u and s not in urls:
                    urls[s] = u
    if "vocals" not in urls:
        raise RuntimeError(f"fal demucs returned no stems: {str(res)[:300]}")
    out = {}
    for s, u in urls.items():
        p = os.path.join(work, f"{s}.wav")
        with urllib.request.urlopen(u, timeout=300) as r, open(p, "wb") as f:
            f.write(r.read())
        out[s] = p
    return out


def separate(wav: str, work: str, separator: str = "demucs") -> dict:
    if os.path.isdir(separator):
        return from_dir(separator)
    if separator == "demucs":
        return demucs(wav, work)
    if separator == "fal":
        return fal(wav, work)
    raise ValueError(f"unknown separator {separator!r}: use demucs, fal, or a folder of stems")
