"""
Initialisation module for Zuffy.

Zuffy is a sklearn compatible open source python library for the exploration of Fuzzy Pattern Trees.
This file defines the package's public API and its version.
"""

from importlib.metadata import version, PackageNotFoundError

from .zuffy import ZuffyClassifier

try:
    __version__ = version("zuffy")
except PackageNotFoundError:
    # The version string for the Zuffy package. Fetched from package metadata if installed,
    # otherwise defaults to "0.0.dev0" for development environments.
    __version__ = "0.0.dev0"


"""
Defines the public API of the Zuffy package.

When `from zuffy import *` is used, only the names listed here
will be imported.
"""
__all__ = [
    "ZuffyClassifier",
    "__version__",
]
