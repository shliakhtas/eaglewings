"""EagleWings: data-analytics platform template. Apache-2.0, Copyright 2026 Myroslav Shliakhta. See NOTICE and CITATION.cff."""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("eaglewings")
except PackageNotFoundError:
    __version__ = "0.1.0"

# один і той самий рядок у консолі, журналі Prefect, версії прогону й тегах ClearML: за ним шукаються проєкти на основі EagleWings
BUILT_WITH = f"Built with EagleWings {__version__}"
BANNER = f"{BUILT_WITH} · Apache-2.0 · (c) 2026 Myroslav Shliakhta · cite: CITATION.cff"
