# Exceptions and Warnings

All exceptions defined by trajectopy derive from `TrajectopyError`; the exception classes are also available in the `trajectopy` namespace (except the warning). Besides them, invalid arguments raise the built-in `TypeError` or `ValueError`, and errors of the file system (e.g. `FileNotFoundError`) are passed on unchanged.

::: trajectopy.exceptions.TrajectopyError
    options:
      heading_level: 2

::: trajectopy.exceptions.TrajectoryError
    options:
      heading_level: 2

::: trajectopy.exceptions.AlignmentError
    options:
      heading_level: 2

::: trajectopy.exceptions.AlignmentParametersError
    options:
      heading_level: 2

::: trajectopy.exceptions.ApproximationError
    options:
      heading_level: 2

::: trajectopy.exceptions.PointSetError
    options:
      heading_level: 2

::: trajectopy.exceptions.TrajectopyIOError
    options:
      heading_level: 2

::: trajectopy.exceptions.TrajectopyDeprecationWarning
    options:
      heading_level: 2
