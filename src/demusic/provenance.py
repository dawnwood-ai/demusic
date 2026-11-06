"""Who made the files demusic writes.

Every cleaned file carries a ``comment`` metadata tag ("cleaned with demusic by Dawnwood").
demusic is Apache-2.0; keeping the credit is a request (see NOTICE), not a license
condition. There is intentionally no switch in the tool to turn it off.
"""

__version__ = "0.1.0"

PROVENANCE = {
    "name": "demusic",
    "by": "Dawnwood (https://dawnwood.ai)",
    "url": "https://github.com/dawnwood-ai/demusic",
    "version": __version__,
}


def credit() -> str:
    return f"cleaned with demusic {__version__} by Dawnwood - {PROVENANCE['url']}"
