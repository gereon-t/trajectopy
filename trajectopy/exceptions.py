"""Exceptions defined by trajectopy. All of them derive from :class:`TrajectopyError`."""


class TrajectopyError(Exception):
    """Base exception for trajectopy."""

    pass


class TrajectoryError(TrajectopyError):
    """Trajectory-specific errors."""

    pass


class AlignmentError(TrajectopyError):
    """Alignment-specific errors."""

    pass


class AlignmentParametersError(TrajectopyError):
    """Alignment parameters errors."""

    pass


class TrajectopyIOError(TrajectopyError):
    """I/O errors."""

    pass


class ApproximationError(TrajectopyError):
    """Approximation-specific errors."""

    pass


class PointSetError(TrajectopyError):
    """Point set errors."""

    pass


class TrajectopyDeprecationWarning(DeprecationWarning):
    """Warning for parts of the API that will be removed in the next major release."""


__all__ = [
    "TrajectopyDeprecationWarning",
    "TrajectopyError",
    "TrajectoryError",
    "AlignmentError",
    "AlignmentParametersError",
    "TrajectopyIOError",
    "ApproximationError",
    "PointSetError",
]


def __getattr__(name: str) -> type:
    if name == "IOError":  # deprecated since it shadowed the builtin exception
        from trajectopy.utils.deprecation import warn_deprecated

        warn_deprecated("trajectopy.exceptions.IOError is deprecated, use TrajectopyIOError instead.")
        return TrajectopyIOError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
