"""Helpers for deprecating parts of the public API.

Deprecated names and arguments keep working for at least one minor release and emit a
:class:`trajectopy.exceptions.TrajectopyDeprecationWarning` (a subclass of :class:`DeprecationWarning`).
They are removed in the next major release, see the deprecation policy in the documentation.
"""

import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

from trajectopy.exceptions import TrajectopyDeprecationWarning

__all__ = ["TrajectopyDeprecationWarning", "renamed_parameters", "warn_deprecated"]

_F = TypeVar("_F", bound=Callable[..., Any])


def warn_deprecated(message: str, stacklevel: int = 2) -> None:
    """Emits a :class:`TrajectopyDeprecationWarning` that points at the caller of the deprecated code."""
    warnings.warn(message, TrajectopyDeprecationWarning, stacklevel=stacklevel + 1)


def renamed_parameters(**renames: str) -> Callable[[_F], _F]:
    """Decorator that accepts the old names of renamed keyword arguments.

    Example:
        >>> @renamed_parameters(alignment_settings="settings")
        ... def align(trajectory, other, *, settings=None): ...

    Passing an old name emits a deprecation warning and forwards the value to the new name. Passing both
    raises a TypeError.
    """

    def decorator(func: _F) -> _F:
        signature = inspect.signature(func)
        for new in renames.values():
            if new not in signature.parameters:
                raise ValueError(f"{func.__qualname__} has no parameter {new!r}")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for old, new in renames.items():
                if old not in kwargs:
                    continue
                if new in kwargs:
                    raise TypeError(f"{func.__qualname__}() got values for both {old!r} and {new!r}")
                warn_deprecated(
                    f"The argument {old!r} of {func.__qualname__}() is deprecated, use {new!r} instead.",
                    stacklevel=2,
                )
                kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
