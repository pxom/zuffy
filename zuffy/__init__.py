"""
Initialisation module for Zuffy.

Zuffy is a sklearn compatible open source python library for the exploration of Fuzzy Pattern Trees.
"""

from importlib.metadata import version, PackageNotFoundError

from .zuffy import ZuffyClassifier

try:
    __version__ = version("zuffy")
except PackageNotFoundError:
    # Fallback for development installs or if the package isn't formally installed
    __version__ = "0.0.dev0" # Use a development version string


__all__ = [
    "ZuffyClassifier",
    "visuals",
    "__version__",
]
