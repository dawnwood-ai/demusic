"""demusic - remove the background music that AI video models (Seedance, Kling, Veo, Hailuo,
Wan, ...) bake into generated clips, while keeping dialogue, sound effects and ambience.

Built and used in production by Dawnwood, an AI drama studio (https://dawnwood.ai).
"""
from .provenance import PROVENANCE, __version__, credit

__all__ = ["PROVENANCE", "__version__", "credit"]
