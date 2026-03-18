"""Py utilities package."""

from importlib import metadata

try:
    __version__ = metadata.version("py_launch_blueprint")
except metadata.PackageNotFoundError:
    __version__ = "0.1.0"

from .projects import main  # Re-export main function
