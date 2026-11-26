"""Versions of trajectopy, Python and the dependencies, for bug reports."""

import platform
import sys
from importlib import metadata

_DEPENDENCIES = (
    "numpy",
    "scipy",
    "pandas",
    "pyproj",
    "autograd",
    "rosbags",
    "pyarrow",
    "PySide6-Essentials",
    "pyqtgraph",
    "PyOpenGL",
)


def versions() -> dict[str, str | None]:
    """Returns the versions of trajectopy, Python, the operating system and the (optional) dependencies.

    Returns:
        dict[str, str | None]: Name to version; None for packages that are not installed.
    """
    from trajectopy.__version__ import __version__

    result: dict[str, str | None] = {
        "trajectopy": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in _DEPENDENCIES:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def show_versions() -> None:
    """Prints the versions of trajectopy, Python, the operating system and the dependencies.

    Please add the output to bug reports.

    Examples:
        >>> import trajectopy as tpy
        >>> tpy.show_versions()  # doctest: +ELLIPSIS
        trajectopy         : ...
    """
    entries = versions()
    width = max(map(len, entries))
    for name, version in entries.items():
        print(f"{name:<{width}} : {version or 'not installed'}")
