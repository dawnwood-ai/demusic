import json
import os

import numpy as np

from demusic import cli, core, detect, ff
from conftest import band_db


def _fake_classifier(monkeypatch, spans):
    monkeypatch.setattr(detect, "music_spans", lambda x, threshold=detect.MUSIC_P: (spans, 0.6 if spans else 0.05))


def test_check_reports_music_and_exits_1(clip, monkeypatch, capsys):
    _fake_classifier(monkeypatch, [(0.0, 4.0)])
    rc = cli.main(["check", clip["mp4"], "--separator", clip["stems"], "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == cli.EXIT_MUSIC and out["music_found"] == 1
    assert out["files"][0]["spans"] == [[0.0, 4.0]]
    assert out["files"][0]["melodic_dbfs"] > core.LOUD_FLOOR_DBFS


def test_classifier_says_no_music_nothing_written(clip, monkeypatch):
    _fake_classifier(monkeypatch, [])
    r = core.process(clip["mp4"], separator=clip["stems"])
    assert r["music"] is False and r["action"] == "none" and r["output"] is None
    assert not os.path.exists(core.default_output(clip["mp4"]))


def test_classifier_error_never_strips(clip, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model missing")
    monkeypatch.setattr(detect, "music_spans", boom)
    r = core.process(clip["mp4"], separator=clip["stems"])
    assert r["music"] is False and r["action"] == "none" and "classifier failed" in r["note"]


def test_quiet_melodic_stems_skip_classifier(clip, monkeypatch):
    called = []
    monkeypatch.setattr(detect, "music_spans", lambda *a, **k: called.append(1) or ([(0, 4)], 0.9))
    r = core.process(clip["mp4"], separator=clip["stems"], loud_floor=0.0)
    assert r["music"] is False and not called


def test_strip_whole_removes_music_keeps_dialogue(clip, monkeypatch):
    _fake_classifier(monkeypatch, [(0.0, 4.0)])
    r = core.process(clip["mp4"], separator=clip["stems"], mode="whole")
    assert r["action"] == "stripped" and r["output"].endswith("clip.demusic.mp4")
    x = ff.pcm(r["output"], 48000, 1)[48000:48000 * 3]
    o = ff.pcm(clip["mp4"], 48000, 1)[48000:48000 * 3]
    assert band_db(o, 48000, 440) - band_db(x, 48000, 440) > 30      # music gone
    assert abs(band_db(o, 48000, 300) - band_db(x, 48000, 300)) < 2  # dialogue kept
    assert ff.has_video(r["output"])


def test_strip_spans_only_touches_the_span(clip, monkeypatch):
    _fake_classifier(monkeypatch, [(0.0, 1.5)])
    r = core.process(clip["mp4"], separator=clip["stems"])
    x = ff.pcm(r["output"], 48000, 1)
    o = ff.pcm(clip["mp4"], 48000, 1)
    early, late = slice(int(0.3 * 48000), int(1.1 * 48000)), slice(int(2.2 * 48000), int(3.6 * 48000))
    assert band_db(o[early], 48000, 440) - band_db(x[early], 48000, 440) > 30
    assert abs(band_db(o[late], 48000, 440) - band_db(x[late], 48000, 440)) < 2


def test_force_skips_detection(clip, monkeypatch):
    monkeypatch.setattr(detect, "music_spans", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    r = core.process(clip["mp4"], separator=clip["stems"], mode="force")
    assert r["action"] == "stripped"


def test_credit_tag_written(clip, monkeypatch):
    _fake_classifier(monkeypatch, [(0.0, 4.0)])
    r = core.process(clip["mp4"], separator=clip["stems"])
    import subprocess
    info = subprocess.run([ff.exe(), "-hide_banner", "-i", r["output"]], capture_output=True, text=True).stderr
    assert "demusic" in info and "Dawnwood" in info


def test_out_dir_copies_unchanged(clip, monkeypatch, capsys):
    _fake_classifier(monkeypatch, [])
    out = clip["tmp"] / "clean"
    rc = cli.main(["strip", clip["mp4"], "--separator", clip["stems"], "--out-dir", str(out), "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert rc == 0 and rep["files"][0]["action"] == "copied"
    assert (out / "clip.mp4").exists()


def test_audio_only_input(clip, monkeypatch, tmp_path):
    _fake_classifier(monkeypatch, [(0.0, 4.0)])
    wav = tmp_path / "a.wav"
    ff.run(["-i", clip["mp4"], "-vn", str(wav)])
    r = core.process(str(wav), separator=clip["stems"])
    assert r["output"].endswith("a.demusic.wav") and not ff.has_video(r["output"])


def test_usage_errors(clip):
    assert cli.main(["strip", clip["mp4"], clip["mp4"], "-o", "x.mp4"]) == cli.EXIT_USAGE
    assert cli.main(["check", clip["mp4"], clip["mp4"], "--separator", clip["stems"]]) == cli.EXIT_USAGE
    assert cli.main(["nope"]) == cli.EXIT_USAGE


def test_missing_file_is_runtime_failure(tmp_path, capsys):
    assert cli.main(["check", str(tmp_path / "none.mp4"), "--json"]) == cli.EXIT_FAIL
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_stems_dir_two_stem_names(tmp_path):
    from demusic import separate
    for n in ("vocals.wav", "instrumental.wav", "notes.txt"):
        (tmp_path / n).write_bytes(b"")
    assert set(separate.from_dir(str(tmp_path))) == {"vocals", "other"}
