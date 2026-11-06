"""MCP server (stdio) exposing demusic to AI agents.

    claude mcp add demusic -- demusic mcp

Tools: ``check_music`` (read-only) and ``remove_music`` (writes a new file, never overwrites
the input). The separator is ``demucs`` unless ``DEMUSIC_SEPARATOR`` says otherwise; the fal
key comes only from the server's environment, never from tool arguments.
"""
import os
from typing import Any

from . import core, detect
from .provenance import PROVENANCE

try:  # mcp >= 2: FastMCP was renamed MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
from mcp.types import ToolAnnotations

INSTRUCTIONS = f"""demusic {PROVENANCE['version']} by Dawnwood: removes background music that AI
video models (Seedance, Kling, Veo, Hailuo, Wan, ...) bake into generated clips, keeping dialogue,
sound effects and ambience. Use check_music first; remove_music writes a new file next to the
input (or to `output`) and returns where. It only replaces the audio where music was heard.
Separation takes roughly the clip's length on CPU. Use absolute paths."""

try:
    mcp = _Server(name="demusic", instructions=INSTRUCTIONS, version=PROVENANCE["version"],
                  website_url=PROVENANCE["url"])
except TypeError:  # mcp 1.x FastMCP has no version / website_url
    mcp = _Server(name="demusic", instructions=INSTRUCTIONS)

_RO = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
_WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)


def _sep() -> str:
    return os.environ.get("DEMUSIC_SEPARATOR", "demucs")


def _safe(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:500]}


@mcp.tool(name="check_music", annotations=_RO, description=(
    "Does this video/audio clip contain a background music bed, and where? Returns music (bool), "
    "spans [[start_s, end_s], ...], melodic_dbfs and strength (sustained music probability). "
    "Read-only. Conservative: it prefers missing a very quiet bed to flagging engine hum or "
    "room tone as music."))
def check_music(path: str, threshold: float = detect.MUSIC_P) -> dict[str, Any]:
    return _safe(lambda: core.process(path, strip=False, separator=_sep(), threshold=threshold))


@mcp.tool(name="remove_music", annotations=_WRITE, description=(
    "Write a copy of a clip with the background music removed: inside the detected music spans "
    "the audio becomes dialogue + percussion/impacts, elsewhere the original audio is kept; the "
    "picture is stream-copied. `output` defaults to <name>.demusic.<ext> next to the input; the "
    "input is never overwritten. mode: spans (default) | whole | force (strip without detection)."))
def remove_music(path: str, output: str | None = None, mode: str = "spans",
                 threshold: float = detect.MUSIC_P) -> dict[str, Any]:
    if output and os.path.abspath(output) == os.path.abspath(path):
        return {"ok": False, "error": "output must differ from the input"}
    return _safe(lambda: core.process(path, output, strip=True, separator=_sep(), mode=mode,
                                      threshold=threshold))
