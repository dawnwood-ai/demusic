import numpy as np
import pytest

from demusic import detect


def test_sustained_needs_consecutive_windows():
    assert detect.sustained([0.1, 0.9, 0.1, 0.9], k=2) == 0.1      # single-window spikes don't count
    assert detect.sustained([0.1, 0.5, 0.6, 0.1], k=2) == 0.5
    assert detect.sustained([], k=2) == 0.0


def test_spans_from_probs_pads_and_merges():
    p = [0, 0, 0.5, 0.6, 0, 0, 0, 0, 0, 0, 0.7, 0.8, 0]
    spans = detect.spans_from_probs(p)
    assert len(spans) == 2
    (a0, b0), (a1, b1) = spans
    assert a0 == pytest.approx(1.0 - 0.3) and b0 == pytest.approx(1.5 + 1.0 + 0.3)
    assert a1 == pytest.approx(5.0 - 0.3)
    # a short gap between two runs is merged
    assert len(detect.spans_from_probs([0.5, 0.5, 0, 0.5, 0.5])) == 1


def test_single_spike_is_not_a_span():
    assert detect.spans_from_probs([0, 0.9, 0, 0.9, 0]) == []


def test_ast_input_shape_and_padding():
    f = detect.ast_input(np.zeros(16000, dtype=np.float32))
    assert f.shape == (1024, 128) and f.dtype == np.float32
    assert np.allclose(f[200], (0 - (-4.2677393)) / (4.5689974 * 2))


def test_fbank_matches_torchaudio_kaldi():
    torch = pytest.importorskip("torch")
    kaldi = pytest.importorskip("torchaudio.compliance.kaldi")
    x = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    ref = kaldi.fbank(torch.from_numpy(x)[None], htk_compat=True, sample_frequency=16000, use_energy=False,
                      window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10).numpy()
    assert np.abs(detect.fbank(x) - ref).max() < 1e-3


def test_model_env_override(monkeypatch, tmp_path):
    p = tmp_path / "m.onnx"
    p.write_bytes(b"x")
    monkeypatch.setenv("DEMUSIC_MODEL", str(p))
    assert detect.model_path() == str(p)
