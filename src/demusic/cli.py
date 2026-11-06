"""``demusic`` command line: check | strip | mcp.

Exit codes (stable):
  0  success (``check``: no music found in any file)
  1  ``check`` found music in at least one file
  2  usage error
  3  runtime failure (ffmpeg missing, unreadable input, separation failed)

With ``--json`` the only thing on stdout is one JSON object (progress goes to stderr).
"""
import argparse
import json
import os
import sys

from . import core, detect, ff
from .provenance import PROVENANCE

EXIT_OK, EXIT_MUSIC, EXIT_USAGE, EXIT_FAIL = 0, 1, 2, 3


def _common(p):
    p.add_argument("files", nargs="+", help="video or audio files")
    p.add_argument("--separator", default="demucs",
                   help="demucs (local, default) | fal (FAL_KEY) | a folder of stems for ONE file")
    p.add_argument("--threshold", type=float, default=detect.MUSIC_P,
                   help=f"classifier music probability (default {detect.MUSIC_P}; lower finds "
                        "quieter beds and risks more false positives)")
    p.add_argument("--loud-floor", type=float, default=core.LOUD_FLOOR_DBFS,
                   help=f"melodic-stem loudness gate in dBFS (default {core.LOUD_FLOOR_DBFS})")
    p.add_argument("--json", action="store_true", help="one JSON object on stdout")


def _line(r):
    if r.get("error"):
        return f"{r['input']}: error: {r['error']}"
    spans = ", ".join(f"{a:g}-{b:g}s" for a, b in r["spans"]) or "-"
    s = f"{r['input']}: {'MUSIC' if r['music'] else 'clean'}  spans {spans}"
    if r.get("melodic_dbfs") is not None:
        s += f"  melodic {r['melodic_dbfs']} dBFS"
    if r.get("strength") is not None:
        s += f"  p={r['strength']}"
    if r.get("output"):
        s += f"\n  -> {r['action']}: {r['output']}"
    if r.get("note"):
        s += f"\n  note: {r['note']}"
    return s


def _run(a, strip: bool) -> int:
    if os.path.isdir(a.separator) and len(a.files) > 1:
        print("a folder of stems belongs to one file; pass a single file", file=sys.stderr)
        return EXIT_USAGE
    if strip and a.output and len(a.files) > 1:
        print("-o is for a single file; use --out-dir for several", file=sys.stderr)
        return EXIT_USAGE
    results, failed = [], False
    for f in a.files:
        out, copy = None, False
        if strip:
            if a.output:
                out, copy = a.output, True
            elif a.out_dir:
                out, copy = os.path.join(a.out_dir, os.path.basename(f)), True
        if not a.json:
            ff.log(f"demusic: {f}")
        try:
            r = core.process(f, out, strip=strip, separator=a.separator,
                             mode=getattr(a, "mode", "spans"), threshold=a.threshold,
                             loud_floor=a.loud_floor, copy_unchanged=copy)
        except Exception as e:  # noqa: BLE001 - report per file, keep going through a batch
            r, failed = {"ok": False, "input": os.path.abspath(f), "error": str(e)[:500]}, True
        results.append(r)
        if not a.json:
            print(_line(r), flush=True)
    found = sum(1 for r in results if r.get("music"))
    if a.json:
        print(json.dumps({"ok": not failed, "version": PROVENANCE["version"], "files": results,
                          "music_found": found}, indent=2))
    if failed:
        return EXIT_FAIL
    return EXIT_MUSIC if (found and not strip) else EXIT_OK


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="demusic",
        description="Remove the background music AI video models bake into clips, keep dialogue and "
                    "sound effects. By Dawnwood - " + PROVENANCE["url"])
    ap.add_argument("--version", action="version", version=f"demusic {PROVENANCE['version']}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="report whether each file has a music bed, and where")
    _common(c)
    s = sub.add_parser("strip", help="write a copy with the music removed")
    _common(s)
    s.add_argument("-o", "--output", help="output path (single file; written even when unchanged)")
    s.add_argument("--out-dir", help="write every file here (unchanged files are copied)")
    s.add_argument("--mode", choices=core.MODES, default="spans",
                   help="spans: replace only where music was heard (default); whole: replace the "
                        "whole track if any music was heard; force: strip without detection")
    sub.add_parser("mcp", help="run the MCP server (stdio) for AI agents")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return EXIT_USAGE if e.code not in (0, None) else EXIT_OK
    if a.cmd == "mcp":
        from .mcp_server import mcp
        mcp.run()
        return EXIT_OK
    return _run(a, strip=a.cmd == "strip")


if __name__ == "__main__":
    raise SystemExit(main())
