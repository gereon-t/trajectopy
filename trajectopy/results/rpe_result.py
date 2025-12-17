import csv
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from trajectopy.core.settings import PairDistanceUnit
from trajectopy.readers.header import HeaderData


@dataclass
class RelativeTrajectoryDeviations:
    """Container holding relative (drift) pose deviations grouped by separation.

    Relative deviations quantify drift between pose pairs separated by a
    certain spatial distance (meters) or temporal offset (seconds). Each key
    in the dictionaries corresponds to a separation value (bucket) and its
    list contains per-pair drift samples for that bucket.

    Attributes:
        pos_dev (Dict[float, List[float]]): Mapping from pair distance to a
            list of position drift magnitudes. Units depend on
            `pair_distance_unit`; values are normalized per 100 m (expressed
            as percentage) when distance-based.
        rot_dev (Dict[float, List[float]]): Mapping from pair distance to a
            list of orientation drift samples in radians.
        pair_distance (Dict[float, List[float]]): Raw separation measurements
            (meters or seconds) for each bucket, used for computing mean
            bucket spacing and step size.
        pair_distance_unit (PairDistanceUnit): Enumeration indicating whether
            bucket keys represent spatial separation (`METER`) or temporal
            separation (`SECOND`). Affects drift normalization and reporting
            units in `RPEResult`.

    Properties:
        num_pairs (int): Total number of evaluated pose pairs across all buckets.
    """

    pos_dev: dict[float, list[float]]
    rot_dev: dict[float, list[float]]
    pair_distance: dict[float, list[float]]
    pair_distance_unit: PairDistanceUnit = PairDistanceUnit.METER

    @property
    def num_pairs(self) -> int:
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
        for self_value, other_value in zip(self.property_dict.values(), other.property_dict.values()):
            assert self_value == other_value

        return True

    def __len__(self) -> int:
        return self.rpe_dev.num_pairs

    @property
    def has_rot_dev(self) -> bool:
        return any(self.rpe_dev.rot_dev.values())

    @property
    def step(self) -> float:
        return np.mean(np.diff(self.mean_pair_distances)) if len(self.mean_pair_distances) > 1 else 0.0

    @property
    def pair_distance_unit(self) -> str:
        return "m" if self.rpe_dev.pair_distance_unit == PairDistanceUnit.METER else "s"

    @property
    def pos_drift_unit(self) -> str:
        return "%" if self.rpe_dev.pair_distance_unit == PairDistanceUnit.METER else "m/s"

    @property
    def rot_drift_unit(self) -> str:
        return "deg/100m" if self.rpe_dev.pair_distance_unit == PairDistanceUnit.METER else "deg/s"

    @property
    def drift_factor(self) -> float:
        return 100.0 if self.rpe_dev.pair_distance_unit == PairDistanceUnit.METER else 1.0

    def compute_metric(self, key: str, func: Callable[[Any], float], factor: float = 1.0) -> list[float]:
        return [float(func(values) * factor) for values in self.rpe_dev.__dict__[key].values() if values]

    @property
    def num_pairs(self) -> list[int]:
        return [len(values) for values in self.rpe_dev.pair_distance.values() if values]

    @property
    def mean_pair_distances(self) -> list[float]:
        return self.compute_metric(key="pair_distance", func=np.mean)

    @property
    def pos_rpe(self) -> float:
        return float(np.mean(self.pos_dev_mean))

    @property
    def rot_rpe(self) -> float:
        """Returns the average rotation drift in radians per 100 meters.

        Returns:
            float: Average rotation drift.
        """
        return np.mean(self.rot_dev_mean)

    @property
    def pos_std(self) -> list[float]:
        return self.compute_metric(key="pos_dev", func=np.std, factor=self.drift_factor)

    @property
    def rot_std(self) -> list[float]:
        if not self.has_rot_dev:
            return []

        return self.compute_metric(key="rot_dev", func=np.std, factor=self.drift_factor)

    @property
    def pos_dev_mean(self) -> list[float]:
        return self.compute_metric(key="pos_dev", func=np.mean, factor=self.drift_factor)

    @property
    def pos_dev_min(self) -> list[float]:
        return self.compute_metric(key="pos_dev", func=np.min, factor=self.drift_factor)

    @property
    def pos_dev_max(self) -> list[float]:
        return self.compute_metric(key="pos_dev", func=np.max, factor=self.drift_factor)

    @property
    def pos_dev_median(self) -> list[float]:
        return self.compute_metric(key="pos_dev", func=np.median, factor=self.drift_factor)

    @property
    def rot_dev_mean(self) -> list[float]:
        if not self.has_rot_dev:
            return []

        return self.compute_metric(key="rot_dev", func=np.mean, factor=self.drift_factor)

    @property
    def rot_dev_min(self) -> list[float]:
        if not self.has_rot_dev:
            return []

        return self.compute_metric(key="rot_dev", func=np.min, factor=self.drift_factor)

    @property
    def rot_dev_max(self) -> list[float]:
        if not self.has_rot_dev:
            return []

        return self.compute_metric(key="rot_dev", func=np.max, factor=self.drift_factor)

    @property
    def rot_dev_median(self) -> list[float]:
        if not self.has_rot_dev:
            return []

        return self.compute_metric(key="rot_dev", func=np.median, factor=self.drift_factor)

    def get_all(self, key: str) -> list[float]:
        ret_list: list[float] = []
        model_dict = self.rpe_dev.__dict__

        if key not in model_dict:
            return ret_list

        for value in model_dict[key].values():
            ret_list.extend(value)

        return ret_list

    @property
    def all_pair_distances(self) -> list[float]:
        return self.get_all(key="pair_distance")

    @property
    def pos_dev_all(self) -> list[float]:
        return self.get_all(key="pos_dev")

    @property
    def all_rot_devs(self) -> list[float]:
        return self.get_all(key="rot_dev")

    @property
    def dynamic_pos_dict(self) -> dict[str, str]:
        return {
            f"Average position drift at {dist:.3f} (avg) {self.pair_distance_unit}": f"{dev:.3f} {self.pos_drift_unit}"
            for dist, dev in zip(self.mean_pair_distances, self.pos_dev_mean)
        }

    @property
    def dynamic_rot_dict(self) -> dict[str, str]:
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
        if self.has_rot_dev:
            return ["pair_distance", "pos_dev", "rot_dev"]

        return ["pair_distance", "pos_dev"]

    def to_dataframe(self) -> pd.DataFrame:
        if self.has_rot_dev:
            return pd.DataFrame(
                np.c_[self.all_pair_distances, self.pos_dev_all, self.all_rot_devs], columns=self.columns
            )
        else:
            return pd.DataFrame(np.c_[self.all_pair_distances, self.pos_dev_all], columns=self.columns)

    def to_file(self, filename: str, mode: str = "a") -> None:
        with open(filename, mode, encoding="utf-8", newline="") as file:
            file.write(f"#relative_dist_unit {self.rpe_dev.pair_distance_unit.name}\n")
            file.write(f"#name {self.name}\n")

            writer = csv.writer(file)
            file.write("#num_pairs ")
            writer.writerow(self.num_pairs)
        self.to_dataframe().to_csv(filename, index=False, mode="a", float_format="%.12f")

    @classmethod
    def from_file(cls, filename: str):
        """Reads a set of relative trajectory deviations from a file."""
        header_data = HeaderData.from_file(filename)
        deviation_data = pd.read_csv(filename, comment="#")

        pos_dev: dict[float, list[float]] = {}
        rot_dev: dict[float, list[float]] = {}
        pair_distance: dict[float, list[float]] = {}

        last_index = 0
        for index in header_data.num_pairs:
            dev_block = deviation_data.iloc[last_index : last_index + index, :]
            pair_distances = dev_block["pair_distance"].to_numpy(dtype=float)
            pos_devs = dev_block["pos_dev"].to_numpy(dtype=float)
            rot_devs = dev_block["rot_dev"].to_numpy(dtype=float) if "rot_dev" in deviation_data.columns else []

            mean_dist = np.mean(pair_distances)
            pair_distance[mean_dist] = list(pair_distances)
            pos_dev[mean_dist] = list(pos_devs)
            rot_dev[mean_dist] = list(rot_devs)

            last_index += index

        rpe_dev = RelativeTrajectoryDeviations(
            pair_distance=pair_distance,
            pos_dev=pos_dev,
            rot_dev=rot_dev,
            pair_distance_unit=header_data.relative_dist_unit,
        )

        return cls(rpe_dev=rpe_dev, name=header_data.name)
