"""Helpers for deprecating parts of the public API.

Deprecated names and arguments keep working for at least one minor release and emit a
:class:`trajectopy.exceptions.TrajectopyDeprecationWarning` (a subclass of :class:`DeprecationWarning`).
They are removed in the next major release, see the deprecation policy in the documentation.
"""

import warnings
from collections.abc import Callable
from typing import Any, TypeVar

from trajectopy.exceptions import TrajectopyDeprecationWarning

__all__ = ["TrajectopyDeprecationWarning", "warn_deprecated"]

_F = TypeVar("_F", bound=Callable[..., Any])


def warn_deprecated(message: str, stacklevel: int = 2) -> None:
    """Emits a :class:`TrajectopyDeprecationWarning` that points at the caller of the deprecated code."""
    warnings.warn(message, TrajectopyDeprecationWarning, stacklevel=stacklevel + 1)
