"""Approximation algorithms for trajectories."""

from trajectopy.processing.lib.approximation.cubic_approximation import (
    CubicApproximation,
)
from trajectopy.processing.lib.approximation.line_approximation import Line3D
from trajectopy.processing.lib.approximation.voxelizer import Voxelizer

__all__ = ["CubicApproximation", "Line3D", "Voxelizer"]
