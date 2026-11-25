from dataclasses import dataclass

import numpy as np


@dataclass
class Line3D:
    """A 3D line defined by a mean point and a direction vector.

    Attributes:
        mean (np.ndarray): The mean point of the line.
        direction (np.ndarray): The direction vector of the line.
    """

    # rest of the class implementation
    mean: np.ndarray
    direction: np.ndarray

    @classmethod
    def from_points(cls, points: np.ndarray) -> "Line3D":
        """Create a 3D line from a set of points.

        This method calculates the direction vector of the line
        from the eigenvector corresponding to the largest eigenvalue
        of the covariance matrix of the input points. The mean of the
        points is used as the mean point of the line.

        Args:
            cls (Line3D): The class object.
            points (np.ndarray): The input points.

        Returns:
            Line3D: A 3D line defined by a mean point and a direction vector.
        """
        cov_matrix = np.cov(points, rowvar=False)
        direction = np.linalg.eigh(cov_matrix)[1][:, -1]

        # mean of the points
        mean = np.mean(points, axis=0)
        return cls(mean=mean, direction=direction)

    def evaluate_at(self, location: np.ndarray) -> list[float]:
        """Evaluates the 3D line at a given location.

        This method calculates the projection of the input location
        onto the 3D line and returns the resulting point.

        Args:
            location (np.ndarray): The input location.

        Returns:
            list[float]: The resulting point on the 3D line.
        """
        tr = (
            (location[0] - self.mean[0]) * self.direction[0]
            + (location[1] - self.mean[1]) * self.direction[1]
            + (location[2] - self.mean[2]) * self.direction[2]
        ) * self.direction
        return [self.mean[0] + tr[0], self.mean[1] + tr[1], self.mean[2] + tr[2]]
