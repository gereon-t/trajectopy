"""Exceptions defined by trajectopy. All of them derive from [`TrajectopyError`][trajectopy.exceptions.TrajectopyError]."""


class TrajectopyError(Exception):
    """Base exception for trajectopy."""

    pass


class TrajectoryError(TrajectopyError):
    """Trajectory-specific errors."""

    pass


class AlignmentError(TrajectopyError, ValueError):
    """The alignment cannot be estimated with the given data and settings.

    For example, no poses are left after the preprocessing, or the lever arm or time shift is estimated
    without the orientations or speeds it needs. It is also a ``ValueError``.
    """

    pass


class AlignmentParametersError(TrajectopyError):
    """Alignment parameters errors."""

    pass


class TrajectopyIOError(TrajectopyError, ValueError):
    """The structure of a file does not match its header or format.

    For example, the number of columns does not match the ``#fields`` header or the TUM, KITTI or EuRoC format,
    the position fields are missing, timestamps or positions are missing or not finite, or the ``#time_format``
    is unknown. Values that cannot be parsed (e.g. text in a numeric column) raise the ``ValueError`` of the
    parser, and errors of the file system (e.g. ``FileNotFoundError``) are passed on unchanged. It is also a
    ``ValueError``.
    """

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
