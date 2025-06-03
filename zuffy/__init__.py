"""
@author: POM <zuffy@mahoonium.ie>
License: BSD 3 clause
Initialisation module for Zuffy.

Zuffy is a sklearn compatible open source python library for the exploration of Fuzzy Pattern Trees.
"""

from .zuffy import ZuffyClassifier

try:
    from importlib.metadata import version
    __version__ = version("zuffy")
except Exception:
    __version__ = "0.0.dev0"

__all__ = [
    "ZuffyClassifier",
    "zwrapper",
    "functions",
    "visuals",
    "__version__",
]
