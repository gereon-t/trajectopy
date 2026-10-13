"""Deprecated location. Use :mod:`trajectopy.io.ascii` and :mod:`trajectopy.io.header` instead."""

from trajectopy.io.ascii import *  # noqa: F403
from trajectopy.io.header import (  # noqa: F401
    HANDLER_MAPPING,
    HEADER_KEYS,
    HeaderData,
    default_line_handler,
    delimiter_line_handler,
    float_line_handler,
    integer_line_handler,
)
