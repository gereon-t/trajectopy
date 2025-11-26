"""Custom exceptions for trajectopy"""


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


class IOError(TrajectopyError):
    """I/O errors."""

    pass


class ApproximationError(TrajectopyError):
    """Approximation-specific errors."""

    pass


class PointSetError(TrajectopyError):
    """Point set errors."""

    pass
