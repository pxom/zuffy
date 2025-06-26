"""
@author: POM <zuffy@mahoonium.ie>
License: BSD 3 clause

Initialisation module for Zuffy.

Zuffy is a sklearn compatible open source python library for the exploration of Fuzzy Pattern Trees.
"""

from .zuffy import ZuffyClassifier

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("zuffy")
except PackageNotFoundError:
    # Fallback for development installs or if the package isn't formally installed
    __version__ = "0.0.dev0" # Use a development version string


__all__ = [
    "ZuffyClassifier",
    "FuzzyTransformer", # functions
    "ZuffyFitIterator", # zwrapper
    "visuals",
    "__version__",
]