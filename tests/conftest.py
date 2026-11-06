import subprocess

import numpy as np
import pytest

from demusic import ff


def tone(path, freq, secs=4.0, vol=0.3, sr=44100):
    subprocess.run([ff.exe(), "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={secs}",
                    "-af", f"volume={vol}", "-ac", "2", "-ar", str(sr), str(path)], check=True)
    return str(path)


@pytest.fixture
def clip(tmp_path):
    """4 s black video whose audio = 'dialogue' (300 Hz) + 'music' (440 Hz), plus a stems folder."""
    d = tmp_path / "stems"
    d.mkdir()
    tone(d / "vocals.wav", 300, vol=0.2)
    tone(d / "drums.wav", 60, vol=0.0)
    tone(d / "other.wav", 440, vol=0.3)
    mp4 = tmp_path / "clip.mp4"
    subprocess.run([ff.exe(), "-v", "error", "-y", "-f", "lavfi", "-i", "color=black:s=64x64:d=4",
                    "-i", str(d / "vocals.wav"), "-i", str(d / "other.wav"),
                    "-filter_complex", "[1:a][2:a]amix=inputs=2:normalize=0[a]", "-map", "0:v", "-map", "[a]",
                    "-shortest", "-c:v", "libx264", "-c:a", "aac", str(mp4)], check=True)
    return {"mp4": str(mp4), "stems": str(d), "tmp": tmp_path}


def band_db(x, sr, f):
    """Level (dB) of the frequency bin nearest f."""
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return 20 * np.log10(spec[int(round(f * len(x) / sr))] + 1e-9)
