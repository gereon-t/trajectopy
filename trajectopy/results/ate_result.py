import os
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from trajectopy.core.trajectory import Trajectory
from trajectopy.results.alignment_result import AlignmentResult
from trajectopy.results.correlation import DeviationCorrelation
from trajectopy.utils.definitions import Sorting
from trajectopy.utils.numeric import rms
from trajectopy.utils.statistics import DEFAULT_CONFIDENCE, block_bootstrap_ci, segment_ids

# cached properties that depend on the deviations and their order
_CACHED = ("rot_dev_xyz", "rot_bias_xyz", "rot_dev_comb", "pos_dev_comb", "correlation", "_confidence_intervals")


@dataclass
class AbsoluteTrajectoryDeviations:
    """Container holding absolute pose deviations between two trajectories.

    The arrays stored in this dataclass represent per-epoch deviations after
    an optional temporal/spatial alignment of a reference (ground truth) and
    a test trajectory. They are later aggregated by `ATEResult` to derive
    metrics such as mean/median Absolute Trajectory Error (ATE), directional
    bias, and RMS statistics.

    All deviations describe how far the evaluated (test) trajectory deviates from the reference, i.e. test
    minus reference: a positive x deviation means the test trajectory is further in x direction, a positive
    along-track deviation that it is ahead, a positive horizontal cross-track deviation that it is to the right and
    a positive vertical cross-track deviation that it is above.

    Attributes:
        pos_dev (np.ndarray): 3D array of absolute position deviation vectors
            in meters. Shape is (N, 3) for N deviation vectors with x, y, z
            components. These deviations are directly depending on the
            definition of the coordinate frame of the trajectories.
        directed_pos_dev (np.ndarray): 3D array of directed position deviation
            vectors (e.g. along-track, horizontal/vertical cross-track) in meters.
            Shape is (N, 3). These deviations are independent of the global
            coordinate frame, as they are derived based on the along-track
            direction of the trajectory.
        rot_dev (Rotation | None): Optional orientation deviations represented
            as a scipy `Rotation` (radians), the rotation from the reference to the test orientation
            (``test * reference^-1``). `None` if rotational evaluation
            was not performed.
        rotations_used (bool): Indicates whether trajectory rotation information
            was employed to derive the direction frame for `directed_pos_dev`.
            If False, directional decomposition has relied on positional
            differences only.
    """

    pos_dev: np.ndarray
    directed_pos_dev: np.ndarray
    rot_dev: Rotation | None = None
    rotations_used: bool = False


class ATEResult:
    """This class represents a set of absolute trajectory deviations

    Absolute trajectory deviations describe absolute pose deviations between
    two trajectories. The deviations are calculated by comparing pairs of
    positions and orientations in the test and reference trajectory.

    Attributes:
        trajectory (Trajectory): The trajectory the deviations are belonging to
        abs_dev (AbsoluteTrajectoryDeviations): The absolute trajectory deviations
        name (str): The name of the result
        alignment (AlignmentResult | None): The alignment that was applied to the trajectory before the
            deviations were computed, or None if unknown (e.g. for results read from a file).

    See Also:
        [`trajectopy.ate`][trajectopy.ate]: computes the result.

    Examples:
        >>> import trajectopy as tpy
        >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
        >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
        >>> ate_result = tpy.ate(estimate, reference)
        >>> ate_result
        <ATEResult 'KITTI ORB vs. KITTI Ground Truth': 4541 poses, position ATE 1.1533 m, rotation ATE 0.6204 deg>
        >>> len(ate_result.pos_dev_along) == len(ate_result)
        True
    """

    def __init__(
        self,
        trajectory: Trajectory,
        abs_dev: AbsoluteTrajectoryDeviations,
        name: str = "",
        alignment: AlignmentResult | None = None,
    ) -> None:
        self.name = name or trajectory.name
        self.trajectory = trajectory
        self.abs_dev = abs_dev
        self.alignment = alignment

    def __len__(self) -> int:
        """Number of compared poses."""
        return len(self.abs_dev.pos_dev)

    def __repr__(self) -> str:
        rotation = f", rotation ATE {np.rad2deg(self.rot_ate):.4f} deg" if self.has_orientation else ""
        return f"<ATEResult {self.name!r}: {len(self)} poses, position ATE {self.pos_ate:.4f} m{rotation}>"

    def describe(self) -> pd.Series:
        """Statistics of the deviations, like :meth:`pandas.DataFrame.describe`.

        Position statistics are in meters, rotation statistics in radians; rotation statistics are NaN
        without orientations. ``pos_mean`` is the position ATE, ``rot_mean`` the rotation ATE.

        Returns:
            pd.Series: Statistics, named after the result.

        Examples:
            >>> import trajectopy as tpy
            >>> reference = tpy.Trajectory.from_file("./example_data/KITTI_gt.traj")
            >>> estimate = tpy.Trajectory.from_file("./example_data/KITTI_ORB.traj")
            >>> ate_result = tpy.ate(estimate, reference)
            >>> ate_result.describe()[["poses", "pos_mean", "pos_rms"]].round(3)
            poses       4541.000
            pos_mean       1.153
            pos_rms        1.304
            Name: KITTI ORB vs. KITTI Ground Truth, dtype: float64
        """
        statistics: dict[str, float] = {
            "poses": float(len(self)),
            "pos_mean": float(self.pos_ate),
            "pos_median": float(self.pos_dev_median),
            "pos_rms": float(self.pos_dev_rms),
            "pos_std": float(self.pos_dev_std),
            "pos_p95": float(self.pos_dev_p95),
            "pos_min": float(self.pos_dev_min),
            "pos_max": float(self.pos_dev_max),
            "pos_bias_x": float(self.pos_bias_x),
            "pos_bias_y": float(self.pos_bias_y),
            "pos_bias_z": float(self.pos_bias_z),
            "pos_bias_along": float(self.pos_bias_along),
            "pos_bias_cross_h": float(self.pos_bias_cross_h),
            "pos_bias_cross_v": float(self.pos_bias_cross_v),
        }
        rotation_keys = ("rot_mean", "rot_rms", "rot_std", "rot_p95", "rot_max")
        if self.has_orientation:
            values = (self.rot_ate, self.rot_dev_rms, self.rot_dev_std, self.rot_dev_p95, self.rot_dev_max)
            statistics |= {key: float(value) for key, value in zip(rotation_keys, values)}
        else:
            statistics |= dict.fromkeys(rotation_keys, np.nan)
        return pd.Series(statistics, name=self.name)

    @property
    def property_dict(self) -> dict[str, str]:
        """Name, type, number of deviations and the main statistics as display texts (used by the GUI's
        properties table).
        """
        return {
            "Name": self.name,
            "Type": self.__class__.__name__,
            "Number of deviations": str(len(self.abs_dev.pos_dev)),
            "Deviation directions derived using": "Rotations"
            if self.abs_dev.rotations_used
            else "Positions / Unknown",
            "Sort by": self.trajectory.sorting.value,
            "Maximum position deviation [m]": f"{self.pos_dev_max:.4f}",
            "Mean position deviation [m]": f"{self.pos_ate:.4f}",
            "Median position deviation [m]": f"{self.pos_dev_median:.4f}",
            "Minimum position deviation [m]": f"{self.pos_dev_min:.4f}",
            "RMS Position [m]": f"{self.pos_dev_rms:.4f}",
            "STD Position [m]": f"{self.pos_dev_std:.4f}",
            "95th Percentile Position [m]": f"{self.pos_dev_p95:.4f}",
            "Mean Position 95% Confidence Interval [m]": self._interval_summary("position", "mean"),
            "RMS Position 95% Confidence Interval [m]": self._interval_summary("position", "rms"),
            "Bias X [m]": f"{self.pos_bias_x:.4f}",
            "Bias Y [m]": f"{self.pos_bias_y:.4f}",
            "Bias Z [m]": f"{self.pos_bias_z:.4f}",
            "RMS X [m]": f"{self.pos_rms_x:.4f}",
            "RMS Y [m]": f"{self.pos_rms_y:.4f}",
            "RMS Z [m]": f"{self.pos_rms_z:.4f}",
            "Bias Along-Track [m]": f"{self.pos_bias_along:.4f}",
            "Bias Horizontal Cross-Track [m]": f"{self.pos_bias_cross_h:.4f}",
            "Bias Vertical Cross-Track [m]": f"{self.pos_bias_cross_v:.4f}",
            "RMS Along-Track [m]": f"{self.pos_rms_along:.4f}",
            "RMS Horizontal Cross-Track [m]": f"{self.pos_rms_cross_h:.4f}",
            "RMS Vertical Cross-Track [m]": f"{self.pos_rms_cross_v:.4f}",
            "Maximum rotation deviation [°]": (
                f"{np.rad2deg(self.rot_dev_max):.4f}" if self.abs_dev.rot_dev is not None else "-"
            ),
            "Mean rotation deviation [°]": (
                f"{np.rad2deg(self.rot_ate):.4f}" if self.abs_dev.rot_dev is not None else "-"
            ),
            "Median rotation deviation [°]": (
                f"{np.rad2deg(self.rot_dev_median):.4f}" if self.abs_dev.rot_dev is not None else "-"
            ),
            "Minimum rotation deviation [°]": (
                f"{np.rad2deg(self.rot_dev_min):.4f}" if self.abs_dev.rot_dev is not None else "-"
            ),
            "RMS Rotation [°]": f"{np.rad2deg(self.rot_dev_rms):.4f}" if self.abs_dev.rot_dev is not None else "-",
            "STD Rotation [°]": f"{np.rad2deg(self.rot_dev_std):.4f}" if self.abs_dev.rot_dev is not None else "-",
            "95th Percentile Rotation [°]": (
                f"{np.rad2deg(self.rot_dev_p95):.4f}" if self.abs_dev.rot_dev is not None else "-"
            ),
            "RMS Roll [°]": f"{np.rad2deg(self.rot_rms_x):.4f}" if self.abs_dev.rot_dev is not None else "-",
            "RMS Pitch [°]": f"{np.rad2deg(self.rot_rms_y):.4f}" if self.abs_dev.rot_dev is not None else "-",
            "RMS Yaw [°]": f"{np.rad2deg(self.rot_rms_z):.4f}" if self.abs_dev.rot_dev is not None else "-",
            "Bias Roll [°]": f"{np.rad2deg(self.rot_bias_x):.4f}" if self.abs_dev.rot_dev is not None else "-",
            "Bias Pitch [°]": f"{np.rad2deg(self.rot_bias_y):.4f}" if self.abs_dev.rot_dev is not None else "-",
            "Bias Yaw [°]": f"{np.rad2deg(self.rot_bias_z):.4f}" if self.abs_dev.rot_dev is not None else "-",
            # temporal correlation of the deviations, see the correlation property
            "Correlation Time Along / Cross-H / Cross-V [s]": self._correlation_summary(
                ("along", "cross_h", "cross_v"), "correlation_time"
            ),
            "Correlation Time X / Y / Z [s]": self._correlation_summary(("x", "y", "z"), "correlation_time"),
            "White Noise Along / Cross-H / Cross-V [%]": self._correlation_summary(
                ("along", "cross_h", "cross_v"), "white_noise_share"
            ),
            "White Noise X / Y / Z [%]": self._correlation_summary(("x", "y", "z"), "white_noise_share"),
            "Independent Deviations Along / Cross-H / Cross-V": self._correlation_summary(
                ("along", "cross_h", "cross_v"), "effective_samples"
            ),
        }

    def __eq__(self, other) -> bool:
        if not isinstance(other, ATEResult):
            return NotImplemented

        return list(self.property_dict.values()) == list(other.property_dict.values())

    def remove_ate_above(self, threshold: float) -> None:
        """Clips the ATE at a given value.

        This is useful to remove outliers from the ATE.

        Args:
            threshold (float): Maximum ATE value to keep.
        """
        if threshold <= 0.0:
            return

        ate_below_idx = self.pos_dev_comb < threshold
        self.trajectory.mask(ate_below_idx, inplace=True)

        self.abs_dev.pos_dev = self.abs_dev.pos_dev[ate_below_idx]
        self.abs_dev.directed_pos_dev = self.abs_dev.directed_pos_dev[ate_below_idx]

        if self.abs_dev.rot_dev is not None:
            # selected as quaternions, a round trip through Euler angles is inexact near gimbal lock
            self.abs_dev.rot_dev = self.abs_dev.rot_dev[ate_below_idx]

        # the combined deviations and rotation statistics are cached and depend on the removed deviations
        for cached in _CACHED:
            self.__dict__.pop(cached, None)

    @cached_property
    def correlation(self) -> DeviationCorrelation:
        """Temporal correlation of the deviations: autocorrelation, correlation time and number of independent
        deviations of each component (cached, see DeviationCorrelation)."""
        return DeviationCorrelation.from_ate_result(self)

    def _correlation_summary(self, components: tuple[str, ...], quantity: str) -> str:
        if len(self.abs_dev.pos_dev) < 2:
            return "-"
        values = [getattr(self.correlation.components[name], quantity) for name in components]
        if quantity == "effective_samples":
            return " / ".join(f"{value:.0f}" for value in values)
        if quantity == "white_noise_share":
            return " / ".join(f"{100 * value:.0f}" for value in values)
        return " / ".join("> max. lag" if np.isinf(value) else f"{value:.2f}" for value in values)

    @cached_property
    def _confidence_intervals(self) -> dict[tuple[str, str, float], tuple[float, float]]:
        return {}

    def confidence_interval(
        self, quantity: str = "position", statistic: str = "mean", confidence: float = DEFAULT_CONFIDENCE
    ) -> tuple[float, float]:
        """Confidence interval of the mean (ATE) or RMS of the position or rotation deviations.

        The deviations are usually correlated in time, so that the interval is derived with a moving block
        bootstrap whose blocks are twice as long as the correlation time (see
        :func:`trajectopy.utils.statistics.block_bootstrap_ci`). It describes how precisely the deviations of
        this evaluation determine the statistic, e.g. whether two estimates differ significantly. It is
        reproducible (fixed seed) and cached.

        Args:
            quantity (str, optional): ``position`` (meters) or ``rotation`` (radians). Defaults to ``position``.
            statistic (str, optional): ``mean`` or ``rms``. Defaults to ``mean``.
            confidence (float, optional): Confidence level. Defaults to 0.95.

        Returns:
            tuple[float, float]: Lower and upper bound.
        """
        if quantity not in ("position", "rotation"):
            raise ValueError(f"Unknown quantity '{quantity}', use 'position' or 'rotation'.")
        key = (quantity, statistic, confidence)
        if key not in self._confidence_intervals:
            values = self.pos_dev_comb if quantity == "position" else self.rot_dev_comb
            order = np.argsort(self.trajectory.timestamps, kind="stable")  # chronological, also if sorted spatially
            self._confidence_intervals[key] = block_bootstrap_ci(
                values[order], self.trajectory.timestamps[order], statistic=statistic, confidence=confidence
            )
        return self._confidence_intervals[key]

    def _interval_summary(self, quantity: str, statistic: str) -> str:
        if quantity == "rotation" and not self.has_orientation:
            return "-"
        lower, upper = self.confidence_interval(quantity, statistic)
        if not np.isfinite(lower):
            return "-"
        return f"{lower:.4f} - {upper:.4f}"

    def pos_dev_percentile(self, percentile: float) -> float:
        """Percentile of the 3D position deviations, e.g. 95 for the value that 95 % of the deviations do not
        exceed.

        Args:
            percentile (float): Percentile between 0 and 100.

        Returns:
            float: The percentile in meters; NaN without deviations.
        """
        return float(np.percentile(self.pos_dev_comb, percentile)) if len(self.pos_dev_comb) else float("nan")

    def rot_dev_percentile(self, percentile: float) -> float:
        """Percentile of the rotation deviations (rotation angles of the deviation rotations).

        Args:
            percentile (float): Percentile between 0 and 100.

        Returns:
            float: The percentile in radians; NaN without rotation deviations.
        """
        return float(np.percentile(self.rot_dev_comb, percentile)) if len(self.rot_dev_comb) else float("nan")

    @property
    def pos_dev_p95(self) -> float:
        """95th percentile of the 3d position deviations in meters."""
        return self.pos_dev_percentile(95)

    @property
    def rot_dev_p95(self) -> float:
        """95th percentile of the rotation deviations in radians."""
        return self.rot_dev_percentile(95)

    def segment_statistics(
        self,
        segment_length: float | None = None,
        by: str = "time",
        labels: np.ndarray | list | None = None,
    ) -> pd.DataFrame:
        """Statistics of the deviations in segments of the trajectory, e.g. to compare sections of different
        conditions (tunnel and open sky) or to see whether the accuracy changes over time.

        Segments are either consecutive windows of ``segment_length`` seconds (``by="time"``) or meters
        (``by="path_length"``), or given by a label of each deviation (``labels``, e.g. from a classification of
        the environment).

        Args:
            segment_length (float | None, optional): Length of the windows. Required unless labels are given.
            by (str, optional): ``time`` or ``path_length``. Defaults to ``time``.
            labels (np.ndarray | list | None, optional): Segment label of each deviation (in the current order of
                the result). Overrides segment_length.

        Returns:
            pd.DataFrame: One row per segment with the columns ``segment``, ``start`` and ``end`` (smallest and
            largest time or path length), ``count``, ``pos_mean``, ``pos_median``, ``pos_rms``, ``pos_p95``,
            ``pos_max`` (meters) and, with orientations, ``rot_mean``, ``rot_rms`` and ``rot_max`` (degrees).
        """
        if by not in ("time", "path_length"):
            raise ValueError(f"Unknown segment index '{by}', use 'time' or 'path_length'.")
        index = self.trajectory.timestamps if by == "time" else self.trajectory.path_lengths
        if labels is not None:
            segment = np.asarray(labels)
            if len(segment) != len(self.pos_dev_comb):
                raise ValueError(f"{len(segment)} labels were given for {len(self.pos_dev_comb)} deviations.")
        elif segment_length is not None:
            segment = segment_ids(index, segment_length)
        else:
            raise ValueError("Give either a segment length or labels.")

        frame = pd.DataFrame({"segment": segment, "index": index, "pos": self.pos_dev_comb})
        if self.has_orientation:
            frame["rot"] = np.rad2deg(self.rot_dev_comb)

        def p95(values: pd.Series) -> float:
            return float(np.percentile(values, 95))

        def rms_of(values: pd.Series) -> float:
            return float(rms(values.to_numpy()))

        grouped = frame.groupby("segment", sort=True)
        table = grouped.agg(
            start=("index", "min"),
            end=("index", "max"),
            count=("pos", "size"),
            pos_mean=("pos", "mean"),
            pos_median=("pos", "median"),
            pos_rms=("pos", rms_of),
            pos_p95=("pos", p95),
            pos_max=("pos", "max"),
        )
        if self.has_orientation:
            table = table.join(grouped.agg(rot_mean=("rot", "mean"), rot_rms=("rot", rms_of), rot_max=("rot", "max")))
        return table.reset_index()

    @property
    def is_spatially_sorted(self) -> bool:
        """True if the trajectory was sorted spatially, i.e. its path lengths do not follow the chronological
        order. Only then, ordering by path length differs from ordering by time."""
        if self.trajectory.sorting == Sorting.PATH_LENGTH:
            return True
        return bool(np.any(np.diff(self.trajectory.path_lengths) < 0))

    def set_sorting(self, sorting: Sorting) -> None:
        """Orders the trajectory and its deviations by time or path length.

        Args:
            sorting (Sorting): New sorting.
        """
        if sorting == self.trajectory.sorting:
            return

        order = self.trajectory.sort_index(sorting)
        self.trajectory.set_sorting(sorting)

        self.abs_dev.pos_dev = self.abs_dev.pos_dev[order]
        self.abs_dev.directed_pos_dev = self.abs_dev.directed_pos_dev[order]
        if self.abs_dev.rot_dev is not None:
            self.abs_dev.rot_dev = self.abs_dev.rot_dev[order]

        # cached per-epoch deviations follow the old order
        for cached in _CACHED:
            self.__dict__.pop(cached, None)

    @property
    def index(self) -> np.ndarray:
        """Returns the variable that currently parametrizes the trajectory.

        Depending on the sorting of the trajectory, this is either time or arc length.

        Returns:
            np.ndarray: Current independent variable array (time or path-length).
        """
        return self.trajectory.index

    @property
    def has_orientation(self) -> bool:
        """Returns True if orientation is available.

        Returns:
            bool: True if rotation deviations are available, False otherwise.
        """
        return self.abs_dev.rot_dev is not None

    @property
    def pos_dev_x(self) -> np.ndarray:
        """Returns x deviations"""
        return self.abs_dev.pos_dev[:, 0]

    @property
    def pos_dev_y(self) -> np.ndarray:
        """Returns y deviations"""
        return self.abs_dev.pos_dev[:, 1]

    @property
    def pos_dev_z(self) -> np.ndarray:
        """Returns z deviations"""
        return self.abs_dev.pos_dev[:, 2]

    @property
    def pos_bias_x(self) -> float:
        """Returns x bias"""
        return np.mean(self.pos_dev_x)

    @property
    def pos_bias_y(self) -> float:
        """Returns y bias"""
        return np.mean(self.pos_dev_y)

    @property
    def pos_bias_z(self) -> float:
        """Returns z bias"""
        return np.mean(self.pos_dev_z)

    @property
    def pos_bias_cross_h(self) -> float:
        """Returns horizontal cross track bias"""
        return np.mean(self.pos_dev_cross_h)

    @property
    def pos_bias_cross_v(self) -> float:
        """Returns vertical cross track bias"""
        return np.mean(self.pos_dev_cross_v)

    @property
    def pos_bias_along(self) -> float:
        """Returns along track bias"""
        return np.mean(self.pos_dev_along)

    @property
    def rot_dev_x(self) -> np.ndarray:
        """Returns roll deviations"""
        return self.rot_dev_xyz[:, 0]

    @property
    def rot_dev_y(self) -> np.ndarray:
        """Returns pitch deviations"""
        return self.rot_dev_xyz[:, 1]

    @property
    def rot_dev_z(self) -> np.ndarray:
        """Returns yaw deviations"""
        return self.rot_dev_xyz[:, 2]

    @cached_property
    def rot_bias_xyz(self) -> np.ndarray:
        """Returns roll, pitch and yaw bias"""
        return self.abs_dev.rot_dev.mean().as_euler(seq="xyz") if self.abs_dev.rot_dev is not None else np.zeros(3)

    @property
    def rot_bias_x(self) -> np.ndarray:
        """Returns roll bias"""
        return self.rot_bias_xyz[0]

    @property
    def rot_bias_y(self) -> np.ndarray:
        """Returns pitch bias"""
        return self.rot_bias_xyz[1]

    @property
    def rot_bias_z(self) -> np.ndarray:
        """Returns yaw bias"""
        return self.rot_bias_xyz[2]

    @property
    def pos_dev_along(self) -> np.ndarray:
        """
        Returns deviations of along track deviations
        """
        return self.abs_dev.directed_pos_dev[:, 0]

    @property
    def pos_dev_cross_h(self) -> np.ndarray:
        """
        Returns deviations of horizontal cross track deviations
        """
        return self.abs_dev.directed_pos_dev[:, 1]

    @property
    def pos_dev_cross_v(self) -> np.ndarray:
        """
        Returns deviations of vertical cross track deviations
        """
        return self.abs_dev.directed_pos_dev[:, 2]

    @cached_property
    def rot_dev_xyz(self) -> np.ndarray:
        """
        Returns rpy deviations
        """
        return (
            self.abs_dev.rot_dev.as_euler(seq="xyz")
            if self.abs_dev.rot_dev is not None
            else np.zeros_like(self.abs_dev.pos_dev)
        )

    @cached_property
    def pos_dev_comb(self) -> np.ndarray:
        """
        Returns position deviations combined using the L2 norm (cached: the statistics use them several times)
        """
        return np.linalg.norm(self.abs_dev.pos_dev, axis=1)

    @cached_property
    def rot_dev_comb(self) -> np.ndarray:
        """
        Returns rotation deviations as single rotation angles (cached: the statistics use them several times)
        """
        return (
            self.abs_dev.rot_dev.magnitude()
            if self.abs_dev.rot_dev is not None
            else np.zeros_like(self.abs_dev.pos_dev)
        )

    @property
    def pos_dev_rms(self) -> float:
        """
        Returns RMS of 3d positions
        """
        return rms(self.pos_dev_comb)

    @property
    def pos_ate(self) -> float:
        """
        Returns mean of 3d position deviations
        """
        return np.mean(self.pos_dev_comb)

    @property
    def pos_dev_max(self) -> float:
        """
        Returns max of 3d position deviations
        """
        return np.max(self.pos_dev_comb)

    @property
    def pos_dev_min(self) -> float:
        """
        Returns min of 3d position deviations
        """
        return np.min(self.pos_dev_comb)

    @property
    def pos_dev_median(self) -> float:
        """
        Returns min of 3d position deviations
        """
        return np.median(self.pos_dev_comb)

    @property
    def pos_dev_std(self) -> float:
        """
        Returns std of 3d position deviations
        """
        return np.std(self.pos_dev_comb)

    @property
    def rot_dev_rms(self) -> float:
        """
        Returns RMS of rotations
        """
        return rms(self.rot_dev_comb) if self.abs_dev.rot_dev is not None else 0.0

    @property
    def rot_dev_std(self) -> float:
        """
        Returns STD of rotations
        """
        return float(np.std(self.rot_dev_comb))

    @property
    def rot_ate(self) -> float:
        """
        Returns mean of rotation deviations
        """
        return float(np.mean(self.rot_dev_comb))

    @property
    def rot_dev_median(self) -> float:
        """
        Returns median of rotations
        """
        return float(np.median(self.rot_dev_comb))

    @property
    def rot_dev_min(self) -> float:
        """
        Returns min of rotations
        """
        return np.min(self.rot_dev_comb)

    @property
    def rot_dev_max(self) -> float:
        """
        Returns max of rotations
        """
        return np.max(self.rot_dev_comb)

    @property
    def pos_rms_along(self) -> float:
        """
        Returns RMS of along track deviations
        """
        return rms(self.pos_dev_along)

    @property
    def pos_rms_cross_h(self) -> float:
        """
        Returns RMS of horizontal cross track deviations
        """
        return rms(self.pos_dev_cross_h)

    @property
    def pos_rms_cross_v(self) -> float:
        """
        Returns RMS of vertical cross track deviations
        """
        return rms(self.pos_dev_cross_v)

    @property
    def pos_rms_x(self) -> float:
        """
        Returns RMS of x deviations
        """
        return rms(self.pos_dev_x)

    @property
    def pos_rms_y(self) -> float:
        """
        Returns RMS of y deviations
        """
        return rms(self.pos_dev_y)

    @property
    def pos_rms_z(self) -> float:
        """
        Returns RMS of z deviations
        """
        return rms(self.pos_dev_z)

    @property
    def rot_rms_x(self) -> float:
        """
        Returns RMS of roll deviations
        """
        return rms(self.rot_dev_x)

    @property
    def rot_rms_y(self) -> float:
        """
        Returns RMS of pitch deviations
        """
        return rms(self.rot_dev_y)

    @property
    def rot_rms_z(self) -> float:
        """
        Returns RMS of yaw deviations
        """
        return rms(self.rot_dev_z)

    @property
    def columns(self) -> list[str]:
        """Returns the column names of the dataframe"""
        trajectory_columns = [
            "time",
            "arc_lengths",
            "pos_x",
            "pos_y",
            "pos_z",
        ]

        if self.trajectory.has_orientation:
            trajectory_columns.extend(["rot_x", "rot_y", "rot_z", "rot_w"])

        deviation_columns = [
            "pos_dev_x",
            "pos_dev_y",
            "pos_dev_z",
            "pos_dev_along",
            "pos_dev_cross_h",
            "pos_dev_cross_v",
        ]

        if self.abs_dev.rot_dev:
            deviation_columns.extend(["rot_dev_x", "rot_dev_y", "rot_dev_z", "rot_dev_w"])

        trajectory_columns.extend(deviation_columns)
        return trajectory_columns

    @classmethod
    def from_file(cls, filename: str | os.PathLike) -> "ATEResult":
        """Reads a result written by :meth:`to_file`.

        Args:
            filename (str | os.PathLike): Path of the file.

        Returns:
            ATEResult: The result.
        """
        from trajectopy.io import results_io

        filename = os.fspath(filename)
        return results_io.read_ate_result(filename)

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the deviations of all poses as a table.

        Returns:
            pd.DataFrame: One row per pose: ``time`` [s], ``arc_lengths`` (path length) [m], the position
            ``pos_x/y/z`` and, with orientations, the orientation ``rot_x/y/z/w`` (quaternion) of the evaluated
            trajectory, followed by the deviations (test minus reference) ``pos_dev_x/y/z`` and
            ``pos_dev_along/cross_h/cross_v`` [m] and, with orientations, the deviation rotation
            ``rot_dev_x/y/z/w`` (quaternion).
        """
        trajectory_data = np.c_[
            self.trajectory.timestamps,
            self.trajectory.path_lengths,
            self.trajectory.positions.xyz,
        ]

        if self.trajectory.rotations is not None:
            trajectory_data = np.c_[trajectory_data, self.trajectory.rotations.as_quat()]

        deviation_data = np.c_[self.abs_dev.pos_dev, self.abs_dev.directed_pos_dev]

        if self.abs_dev.rot_dev:
            deviation_data = np.c_[deviation_data, self.abs_dev.rot_dev.as_quat()]

        all_data = np.c_[trajectory_data, deviation_data]

        return pd.DataFrame(all_data, columns=self.columns)

    def to_kml(
        self, filename: str | os.PathLike, field: str = "position", max_std: float = 3.0, **options: Any
    ) -> None:
        """Exports the result to a Google Earth KML file: the trajectory colored by a deviation field, with
        playback in Google Earth's time slider. Requires a trajectory with a known EPSG code.

        Args:
            filename (str): The output filename (e.g., "ate.kml").
            field (str, optional): Deviation field that colors the trajectory, e.g. "position", "x",
                "along_track", "rotation" or "yaw" (see :func:`trajectopy.io.kml.ate_kml_fields`).
                Defaults to "position".
            max_std (float, optional): The color scale is clipped to mean ± ``max_std`` standard deviations
                (0 = minimum to maximum). Defaults to 3.0.
            **options: Further options of :func:`trajectopy.io.kml.write_ate_kml`, e.g. ``colormap="viridis"``,
                ``value_range=(0.0, 0.5)``, ``heights="ellipsoidal"`` or ``extrude=True``.
        """
        from trajectopy.io.kml import write_ate_kml

        filename = os.fspath(filename)
        write_ate_kml(self, filename, field=field, max_std=max_std, **options)

    def to_file(self, filename: str | os.PathLike, mode: str = "w") -> None:
        """Writes the result to a file that :meth:`from_file` and the GUI can read.

        Args:
            filename (str | os.PathLike): Path of the file.
            mode (str, optional): ``"w"`` (default) overwrites the file, ``"a"`` appends to it.
        """
        from trajectopy.io import results_io

        filename = os.fspath(filename)
        results_io.write_ate_result(self, filename, mode=mode)
