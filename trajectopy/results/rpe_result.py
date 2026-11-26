import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from trajectopy.core.settings import PairDistanceUnit


@dataclass
class RelativeTrajectoryDeviations:
    """Container holding relative (drift) pose deviations grouped by separation.

    Relative deviations quantify drift between pose pairs separated by a
    certain spatial distance (meters) or temporal offset (seconds). Each key
    in the dictionaries corresponds to a separation value (bucket) and its
    list contains per-pair drift samples for that bucket.

    Attributes:
        pos_dev (Dict[float, np.ndarray]): Mapping from pair distance to the
            position drift magnitudes. Units depend on
            `pair_distance_unit`; values are normalized per 100 m (expressed
            as percentage) when distance-based.
        rot_dev (Dict[float, np.ndarray]): Mapping from pair distance to the
            orientation drift samples in radians.
        pair_distance (Dict[float, np.ndarray]): Raw separation measurements
            (meters or seconds) for each bucket, used for computing mean
            bucket spacing and step size.
        pair_distance_unit (PairDistanceUnit): Enumeration indicating whether
            bucket keys represent spatial separation (`METER`) or temporal
            separation (`SECOND`). Affects drift normalization and reporting
            units in `RPEResult`.

    Properties:
        num_pairs (int): Total number of evaluated pose pairs across all buckets.
    """

    pos_dev: dict[float, np.ndarray]
    rot_dev: dict[float, np.ndarray]
    pair_distance: dict[float, np.ndarray]
    pair_distance_unit: PairDistanceUnit = PairDistanceUnit.METER

    def __post_init__(self) -> None:
        # arrays (lists are accepted): millions of pairs are copied and saved as a whole, not number by number
        for deviations in (self.pos_dev, self.rot_dev, self.pair_distance):
            for key, values in deviations.items():
                deviations[key] = np.asarray(values, dtype=float)

    @property
    def num_pairs(self) -> int:
        """Total number of compared pose pairs over all pair distances."""
        return sum(len(values) for values in self.pair_distance.values())


class RPEResult:
    """
    This class represents a set of relative trajectory deviations

    Relative trajectory deviations describe relative pose deviations between
    two trajectories. The deviations are calculated by comparing pairs of
    positions and orientations in the test and reference trajectory.

    Attributes:
        rpe_dev (RelativeTrajectoryDeviations): The relative trajectory deviations.
        name (str): Name of the deviation set.
    """

    def __init__(
        self,
        rpe_dev: RelativeTrajectoryDeviations,
        name: str,
    ) -> None:
        self.name = name
        self.rpe_dev = rpe_dev

    def __eq__(self, other) -> bool:
        if not isinstance(other, RPEResult):
            return NotImplemented

        return list(self.property_dict.values()) == list(other.property_dict.values())

    def __len__(self) -> int:
        """Number of compared pose pairs."""
        return self.rpe_dev.num_pairs

    def __repr__(self) -> str:
        rotation = f", rotation drift {np.rad2deg(self.rot_rpe):.4f} {self.rot_drift_unit}" if self.has_rot_dev else ""
        return (
            f"<RPEResult {self.name!r}: {len(self)} pose pairs, "
            f"position drift {self.pos_rpe:.4f} {self.pos_drift_unit}{rotation}>"
        )

    def describe(self) -> pd.DataFrame:
        """Statistics of the drift per pair distance, like :meth:`pandas.DataFrame.describe`.

        Position drifts are in % (pairs by distance) or m/s (pairs by time), rotation drifts in rad/100m or
        rad/s; rotation statistics are NaN without orientations.

        Returns:
            pd.DataFrame: One row per pair distance, indexed by the mean distance of the pairs.
        """
        columns: dict[str, list] = {
            "pairs": self.num_pairs,
            "pos_mean": self.pos_dev_mean,
            "pos_median": self.pos_dev_median,
            "pos_std": self.pos_std,
            "pos_min": self.pos_dev_min,
            "pos_max": self.pos_dev_max,
        }
        rotation = {
            "rot_mean": self.rot_dev_mean,
            "rot_median": self.rot_dev_median,
            "rot_std": self.rot_std,
            "rot_min": self.rot_dev_min,
            "rot_max": self.rot_dev_max,
        }
        for key, values in rotation.items():
            columns[key] = values if self.has_rot_dev else [np.nan] * len(self.num_pairs)
        index = pd.Index(self.mean_pair_distances, name=f"pair_distance [{self.pair_distance_unit}]")
        return pd.DataFrame(columns, index=index)

    @property
    def has_rot_dev(self) -> bool:
        """Whether rotation drifts are available (both trajectories have orientations)."""
        return any(len(values) for values in self.rpe_dev.rot_dev.values())

    @property
    def step(self) -> float:
        """Mean step between the pair distances (m or s); 0 for a single pair distance."""
        return float(np.mean(np.diff(self.mean_pair_distances))) if len(self.mean_pair_distances) > 1 else 0.0

    @property
    def pair_distance_unit(self) -> str:
        """Unit of the pair distances: ``"m"`` or ``"s"``."""
        return "m" if self.rpe_dev.pair_distance_unit == PairDistanceUnit.METER else "s"

    @property
    def pos_drift_unit(self) -> str:
        """Display unit of the position drift: ``"%"`` (pairs by distance) or ``"m/s"``."""
        return "%" if self.rpe_dev.pair_distance_unit == PairDistanceUnit.METER else "m/s"

    @property
    def rot_drift_unit(self) -> str:
        """Display unit of the rotation drift: ``"deg/100m"`` or ``"deg/s"``.

        The rotation values of the result are in radians; convert them with :func:`numpy.rad2deg` for display.
        """
        return "deg/100m" if self.rpe_dev.pair_distance_unit == PairDistanceUnit.METER else "deg/s"

    @property
    def drift_factor(self) -> float:
        """Factor from the stored relative deviations to the drift unit: 100 for pairs by distance (per 100 m or
        %), 1 for pairs by time.
        """
        return 100.0 if self.rpe_dev.pair_distance_unit == PairDistanceUnit.METER else 1.0

    def compute_metric(self, key: str, func: Callable[[Any], float], factor: float = 1.0) -> list[float]:
        """Applies a statistic to the deviations of each pair distance.

        Args:
            key (str): ``"pos_dev"``, ``"rot_dev"`` or ``"pair_distance"``.
            func (Callable): Statistic of an array, e.g. :func:`numpy.mean`.
            factor (float, optional): Factor applied to the statistic, e.g. :attr:`drift_factor`.
                Defaults to 1.

        Returns:
            list[float]: One value per pair distance with deviations.
        """
        return [float(func(values) * factor) for values in self.rpe_dev.__dict__[key].values() if len(values)]

    @property
    def num_pairs(self) -> list[int]:
        """Number of pose pairs per pair distance."""
        return [len(values) for values in self.rpe_dev.pair_distance.values() if len(values)]

    @property
    def mean_pair_distances(self) -> list[float]:
        """Mean distance (m or s) of the pose pairs per pair distance."""
        return self.compute_metric(key="pair_distance", func=np.mean)

    @property
    def pos_rpe(self) -> float:
        """Average position drift over all pair distances (see :attr:`pos_drift_unit`)."""
        return float(np.mean(self.pos_dev_mean))

    @property
    def rot_rpe(self) -> float:
        """Returns the average rotation drift in radians per 100 meters.

        Returns:
            float: Average rotation drift.
        """
        return float(np.mean(self.rot_dev_mean))

    @property
    def pos_std(self) -> list[float]:
        """Standard deviation of the position drift per pair distance. in % (pairs by distance) or m/s (pairs by time)."""
        return self.compute_metric(key="pos_dev", func=np.std, factor=self.drift_factor)

    @property
    def rot_std(self) -> list[float]:
        """Standard deviation of the rotation drift per pair distance (empty without rotations)."""
        if not self.has_rot_dev:
            return []

        return self.compute_metric(key="rot_dev", func=np.std, factor=self.drift_factor)

    @property
    def pos_dev_mean(self) -> list[float]:
        """Mean position drift per pair distance. in % (pairs by distance) or m/s (pairs by time)."""
        return self.compute_metric(key="pos_dev", func=np.mean, factor=self.drift_factor)

    @property
    def pos_dev_min(self) -> list[float]:
        """Minimum position drift per pair distance (see :attr:`pos_dev_mean` for the unit)."""
        return self.compute_metric(key="pos_dev", func=np.min, factor=self.drift_factor)

    @property
    def pos_dev_max(self) -> list[float]:
        """Maximum position drift per pair distance (see :attr:`pos_dev_mean` for the unit)."""
        return self.compute_metric(key="pos_dev", func=np.max, factor=self.drift_factor)

    @property
    def pos_dev_median(self) -> list[float]:
        """Median position drift per pair distance (see :attr:`pos_dev_mean` for the unit)."""
        return self.compute_metric(key="pos_dev", func=np.median, factor=self.drift_factor)

    @property
    def rot_dev_mean(self) -> list[float]:
        """Mean rotation drift per pair distance, in rad/100 m (pairs by distance) or rad/s (pairs by time) (empty without rotations)."""
        if not self.has_rot_dev:
            return []

        return self.compute_metric(key="rot_dev", func=np.mean, factor=self.drift_factor)

    @property
    def rot_dev_min(self) -> list[float]:
        """Minimum rotation drift per pair distance (see :attr:`rot_dev_mean` for the unit)."""
        if not self.has_rot_dev:
            return []

        return self.compute_metric(key="rot_dev", func=np.min, factor=self.drift_factor)

    @property
    def rot_dev_max(self) -> list[float]:
        """Maximum rotation drift per pair distance (see :attr:`rot_dev_mean` for the unit)."""
        if not self.has_rot_dev:
            return []

        return self.compute_metric(key="rot_dev", func=np.max, factor=self.drift_factor)

    @property
    def rot_dev_median(self) -> list[float]:
        """Median rotation drift per pair distance (see :attr:`rot_dev_mean` for the unit)."""
        if not self.has_rot_dev:
            return []

        return self.compute_metric(key="rot_dev", func=np.median, factor=self.drift_factor)

    def get_all(self, key: str) -> np.ndarray:
        """Returns the values of all pose pairs of one quantity, concatenated over the pair distances.

        Args:
            key (str): ``"pos_dev"``, ``"rot_dev"`` or ``"pair_distance"``.

        Returns:
            np.ndarray: The values as stored (relative deviations, not multiplied by :attr:`drift_factor`);
            empty if there are none.
        """
        model_dict = self.rpe_dev.__dict__
        if key not in model_dict or not model_dict[key]:
            return np.array([])
        return np.concatenate([np.asarray(values, dtype=float) for values in model_dict[key].values()])

    @property
    def all_pair_distances(self) -> np.ndarray:
        """Distances (m or s) of all pose pairs."""
        return self.get_all(key="pair_distance")

    @property
    def pos_dev_all(self) -> np.ndarray:
        """Relative position deviations of all pose pairs (m/m or m/s)."""
        return self.get_all(key="pos_dev")

    @property
    def all_rot_devs(self) -> np.ndarray:
        """Relative rotation deviations of all pose pairs (rad/m or rad/s)."""
        return self.get_all(key="rot_dev")

    @property
    def dynamic_pos_dict(self) -> dict[str, str]:
        """Mean position drift per pair distance as display texts (used by the GUI)."""
        return {
            f"Average position drift at {dist:.3f} (avg) {self.pair_distance_unit}": f"{dev:.3f} {self.pos_drift_unit}"
            for dist, dev in zip(self.mean_pair_distances, self.pos_dev_mean)
        }

    @property
    def dynamic_rot_dict(self) -> dict[str, str]:
        """Mean rotation drift per pair distance as display texts (used by the GUI)."""
        if not self.has_rot_dev:
            return {}

        return {
            f"Average rotation drift at {dist:.3f} (avg) {self.pair_distance_unit}": f"{np.rad2deg(dev):.3f} {self.rot_drift_unit}"
            for dist, dev in zip(self.mean_pair_distances, self.rot_dev_mean)
        }

    @property
    def property_dict(self) -> dict[str, str]:
        """Returns a dictionary containing the properties of the deviation set.

        This is relevant for time based comparisons, when pose-pairs are defined by a time difference.

        Returns:
            Dict[str, str]: Dictionary of property names and values.
        """
        basic_dict = {
            "Name": self.name,
            "Type": self.__class__.__name__,
            "Number of pose pairs": str(len(self)),
        }
        dynamic_pos_dict = self.dynamic_pos_dict
        static_pos_dict = {
            "Maximum Position Drift": f"{np.max(self.pos_dev_max):.3f} {self.pos_drift_unit}",
            "Minimum Position Drift": f"{np.min(self.pos_dev_min):.3f} {self.pos_drift_unit}",
            "Average Position Drift": f"{self.pos_rpe:.3f} {self.pos_drift_unit}",
            "Median Position Drift": f"{np.median(self.pos_dev_median):.3f} {self.pos_drift_unit}",
        }

        pos_dict = basic_dict.copy()
        pos_dict.update(dynamic_pos_dict)
        pos_dict.update(static_pos_dict)

        if not self.has_rot_dev:
            return pos_dict

        dynamic_rot_dict = self.dynamic_rot_dict
        static_rot_dict = {
            "Maximum Rotation Drift": f"{np.rad2deg(np.max(self.rot_dev_max)):.3f} {self.rot_drift_unit}",
            "Minimum Rotation Drift": f"{np.rad2deg(np.min(self.rot_dev_min)):.3f} {self.rot_drift_unit}",
            "Average Rotation Drift": f"{np.rad2deg(self.rot_rpe):.3f} {self.rot_drift_unit}",
            "Median Rotation Drift": f"{np.rad2deg(np.median(self.rot_dev_median)):.3f} {self.rot_drift_unit}",
        }
        pos_dict.update(dynamic_rot_dict)
        pos_dict.update(static_rot_dict)
        return pos_dict

    @property
    def columns(self) -> list[str]:
        """Column names of :meth:`to_dataframe`."""
        if self.has_rot_dev:
            return ["pair_distance", "pos_dev", "rot_dev"]

        return ["pair_distance", "pos_dev"]

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the deviations of all pose pairs as a table.

        Returns:
            pd.DataFrame: Columns ``pair_distance`` (m or s), ``pos_dev`` (m/m or m/s) and, with orientations,
            ``rot_dev`` (rad/m or rad/s); one row per pose pair.
        """
        if self.has_rot_dev:
            return pd.DataFrame(
                np.c_[self.all_pair_distances, self.pos_dev_all, self.all_rot_devs], columns=self.columns
            )
        else:
            return pd.DataFrame(np.c_[self.all_pair_distances, self.pos_dev_all], columns=self.columns)

    def to_file(self, filename: str | os.PathLike, mode: str = "w") -> None:
        """Writes the result to a file that :meth:`from_file` and the GUI can read.

        Args:
            filename (str | os.PathLike): Path of the file.
            mode (str, optional): ``"w"`` (default) overwrites the file, ``"a"`` appends to it.
        """
        from trajectopy.io import results_io

        filename = os.fspath(filename)
        results_io.write_rpe_result(self, filename, mode=mode)

    @classmethod
    def from_file(cls, filename: str | os.PathLike) -> "RPEResult":
        """Reads a result written by :meth:`to_file`.

        Args:
            filename (str | os.PathLike): Path of the file.

        Returns:
            RPEResult: The result.
        """
        from trajectopy.io import results_io

        filename = os.fspath(filename)
        return results_io.read_rpe_result(filename)
