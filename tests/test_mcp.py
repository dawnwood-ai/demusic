import asyncio

from demusic import detect, mcp_server


def test_tools_listed():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    assert {t.name for t in tools} == {"check_music", "remove_music"}


def test_remove_music_refuses_in_place(tmp_path):
    p = str(tmp_path / "a.mp4")
    assert mcp_server.remove_music(p, output=p)["ok"] is False


def test_check_music_with_stems_dir(clip, monkeypatch):
    monkeypatch.setenv("DEMUSIC_SEPARATOR", clip["stems"])
    monkeypatch.setattr(detect, "music_spans", lambda x, threshold=detect.MUSIC_P: ([(0.0, 2.0)], 0.5))
    r = mcp_server.check_music(clip["mp4"])
    assert r["music"] is True and r["spans"] == [[0.0, 2.0]]
