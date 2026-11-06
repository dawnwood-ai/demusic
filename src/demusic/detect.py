"""Is this music? An AudioSet AST classifier run window by window.

Why a classifier and not just loudness: after stem separation, a music bed lands in the
bass / guitar / piano / other stems - but so do engine hum, gunshots and room tone. In our
production calibration (87 clips a human had de-musiced by ear vs 164 signed-off clips with
no music) a loudness-only gate wrongly flagged 8/164 clean clips; loudness AND this
classifier flagged 2/164. Recall was 33/87: the misses are mostly very quiet beds the
classifier cannot hear. The tool is built to miss rather than damage a clean clip.

Model: MIT's AST fine-tuned on AudioSet (BSD-3-Clause), the int8 ONNX export from
onnx-community (~90 MB), downloaded once to ``$DEMUSIC_CACHE`` (default
``~/.cache/demusic``; honours ``$HF_ENDPOINT`` mirrors) or taken from ``$DEMUSIC_MODEL``. Features are Kaldi-compatible
log-mel fbanks normalised like ``ASTFeatureExtractor``, in plain numpy (max abs error
vs torchaudio < 2e-5), so no torch is needed for detection.
"""
import os
import threading
import urllib.request

import numpy as np

SR = 16000
MODEL_REPO = "onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX"
MODEL_REV = "2fe67046c486b8b8bc90a7b469239b2dec822f72"
MODEL_FILE = "onnx/model_int8.onnx"
MODEL_SIZE = 90592065

#: AudioSet classes that mean "a music bed": Music, Musical instrument, Piano, Keyboard,
#: Synthesizer, Pad sound, Bowed string, Violin, Cello, String section, Orchestra,
#: Classical, Ambient, Background music, Soundtrack music, Theme music.
MUSIC_IDX = (137, 138, 152, 153, 158, 184, 189, 190, 191, 193, 237, 246, 267, 268, 270)
WIN_S, HOP_S = 1.0, 0.5
SUSTAIN = 2        # windows in a row that must pass (a single-window spike is usually an SFX)
MUSIC_P = 0.3      # calibrated: 0.3 -> 2/164 false, 33/87 recall; 0.2 -> 4/164, 41/87

_lock = threading.Lock()
_sess: dict = {"v": None}


def _melbank(n_mels=128, nfft=512, lo=20.0):
    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)
    nb = nfft // 2
    mlo, mhi = mel(lo), mel(SR / 2)
    d = (mhi - mlo) / (n_mels + 1)
    freqs = mel(SR / nfft * np.arange(nb))
    bank = np.zeros((n_mels, nb + 1))
    for m in range(n_mels):
        left, c, r = mlo + m * d, mlo + (m + 1) * d, mlo + (m + 2) * d
        bank[m, :nb] = np.maximum(0, np.minimum((freqs - left) / (c - left), (r - freqs) / (r - c)))
    return bank


_MB = _melbank()


def fbank(x) -> np.ndarray:
    """Kaldi-compatible log-mel (the torchaudio.compliance.kaldi.fbank settings used by
    ASTFeatureExtractor: 128 mel, 25 ms / 10 ms, hanning, pre-emphasis 0.97, DC removal)."""
    x = np.asarray(x, dtype=np.float64)
    fl, fs = 400, 160
    n = 1 + (len(x) - fl) // fs if len(x) >= fl else 0
    if n <= 0:
        return np.zeros((0, 128))
    fr = x[np.arange(fl)[None, :] + fs * np.arange(n)[:, None]]
    fr = fr - fr.mean(1, keepdims=True)
    fr = np.concatenate([fr[:, :1], fr[:, 1:] - 0.97 * fr[:, :-1]], 1)
    fr = fr * (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(fl) / (fl - 1)))
    m = (np.abs(np.fft.rfft(fr, 512)) ** 2) @ _MB.T
    return np.log(np.maximum(m, np.finfo(np.float32).eps))


def ast_input(x, max_len=1024) -> np.ndarray:
    f = fbank(x)
    f = np.pad(f, ((0, max_len - len(f)), (0, 0))) if len(f) < max_len else f[:max_len]
    return ((f - (-4.2677393)) / (4.5689974 * 2)).astype(np.float32)


def model_path() -> str:
    """Local path of the ONNX model, downloading it on first use."""
    if os.environ.get("DEMUSIC_MODEL"):
        return os.environ["DEMUSIC_MODEL"]
    cache = os.environ.get("DEMUSIC_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "demusic")
    p = os.path.join(cache, f"ast-audioset-int8-{MODEL_REV[:8]}.onnx")
    if os.path.exists(p) and os.path.getsize(p) == MODEL_SIZE:
        return p
    os.makedirs(cache, exist_ok=True)
    from .ff import log
    url = f"{os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')}/{MODEL_REPO}/resolve/{MODEL_REV}/{MODEL_FILE}"
    log(f"demusic: downloading the music classifier (~{MODEL_SIZE // 1_000_000} MB, once) to {p}")
    tmp = p + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if os.path.getsize(tmp) != MODEL_SIZE:
        os.remove(tmp)
        raise RuntimeError(f"model download incomplete from {url}")
    os.replace(tmp, p)
    return p


def _session():
    with _lock:
        if _sess["v"] is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = max(1, os.cpu_count() or 2)
            so.log_severity_level = 3
            _sess["v"] = ort.InferenceSession(model_path(), so, providers=["CPUExecutionProvider"])
        return _sess["v"]


def music_probs(x) -> np.ndarray:
    """16 kHz mono float waveform -> per-window (1 s, 0.5 s hop) music probability
    (max over the music classes)."""
    x = np.asarray(x, dtype=np.float32)
    n, h = int(WIN_S * SR), int(HOP_S * SR)
    chunks = [x[i:i + n] for i in range(0, max(1, len(x) - n + 1), h)]
    s = _session()
    name = s.get_inputs()[0].name
    out = []
    for i in range(0, len(chunks), 8):
        feats = np.stack([ast_input(c) for c in chunks[i:i + 8]])
        lg = s.run(None, {name: feats})[0]
        out.append((1 / (1 + np.exp(-lg)))[:, list(MUSIC_IDX)].max(1))
    return np.concatenate(out) if out else np.zeros(0)


def sustained(p, k: int = SUSTAIN) -> float:
    """Max over positions of the min of k consecutive windows (= strength held for k windows)."""
    p = np.asarray(p)
    if len(p) == 0:
        return 0.0
    if len(p) < k:
        return float(p.min())
    return float(max(p[i:i + k].min() for i in range(len(p) - k + 1)))


def spans_from_probs(p, threshold: float = MUSIC_P, pad: float = 0.3, merge: float = 1.0) -> list:
    """Windows where SUSTAIN consecutive windows pass ``threshold`` -> [(start_s, end_s)],
    each widened by ``pad`` and joined when the gap is under ``merge`` seconds."""
    p = list(p)
    hot = [False] * len(p)
    for i in range(len(p) - SUSTAIN + 1):
        if min(p[i:i + SUSTAIN]) >= threshold:
            for j in range(i, i + SUSTAIN):
                hot[j] = True
    spans, i = [], 0
    while i < len(hot):
        if not hot[i]:
            i += 1
            continue
        j = i
        while j < len(hot) and hot[j]:
            j += 1
        a, b = max(0.0, i * HOP_S - pad), (j - 1) * HOP_S + WIN_S + pad
        if spans and a - spans[-1][1] < merge:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
        i = j
    return spans


def music_spans(accompaniment_16k, threshold: float = MUSIC_P) -> tuple:
    """Where is the music? Pass the accompaniment (original minus the separated vocals -
    dialogue pulls the classifier towards Speech). Returns (spans, sustained_strength)."""
    p = music_probs(accompaniment_16k)
    return spans_from_probs(p, threshold), round(sustained(p), 3)
