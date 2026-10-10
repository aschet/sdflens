# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Point cloud model: the measured points of an x3p file that are not on a regular grid."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from .surface import (
    AUTO_HEIGHT_FRACTION,
    AUTO_Z_FACTOR_MAX,
    ROBUST_PERCENTILES,
    NoMeasuredPointsError,
    ValueRange,
    data_range,
)
from .zscale import clamp_z_factor

__all__ = ["PointCloudModel"]


@dataclass(frozen=True, eq=False)
class PointCloudModel:
    """Measured points in meters with the statistics of their heights.

    Only points with three finite coordinates are kept. The lateral extent is the bounding
    box of the points, so a cloud that is no wider than a point is shown at a size of one
    nanometer.

    The points of an irregular surface, whose x and y are stored for every point, also keep
    their matrix as :attr:`grid`: the neighbours of a point in its rows and columns are its
    neighbours in space, so the points can be connected into a surface. A point cloud has none.
    """

    points: NDArray[np.float64]
    value_range: ValueRange
    z_mean: float
    scale: float
    z_center: float
    center_x: float
    center_y: float
    extent_x: float
    extent_y: float
    grid: NDArray[np.float64] | None = None

    @classmethod
    def from_grid(cls, grid: NDArray[np.float64]) -> PointCloudModel:
        """Build the model from a ``(rows, columns, 3)`` array of ``x``, ``y``, ``z`` in meters.

        The matrix is kept, with ``NaN`` for the coordinates of a point that is not measured.

        :raises NoMeasuredPointsError: If no point has three finite coordinates.
        """
        grid = np.asarray(grid, dtype=np.float64)
        return replace(cls.from_points(grid.reshape(-1, 3)), grid=grid)

    @classmethod
    def from_points(cls, points: NDArray[np.float64]) -> PointCloudModel:
        """Build the model from an ``(N, 3)`` array of ``x``, ``y``, ``z`` in meters.

        :raises NoMeasuredPointsError: If no point has three finite coordinates.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            raise NoMeasuredPointsError("The file contains no measured points")
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        value_range = data_range(z)
        extent_x = float(x.max() - x.min())
        extent_y = float(y.max() - y.min())
        return cls(
            points=points,
            value_range=value_range,
            z_mean=float(z.mean()),
            scale=max(extent_x, extent_y, 1e-9),
            z_center=0.5 * (value_range.lo + value_range.hi),
            center_x=0.5 * float(x.max() + x.min()),
            center_y=0.5 * float(y.max() + y.min()),
            extent_x=extent_x,
            extent_y=extent_y,
        )

    @property
    def num_points(self) -> int:
        """Number of measured points."""
        return len(self.points)

    @property
    def invalid_count(self) -> int:
        """Number of non-measured points: none for a point cloud, which keeps no such points."""
        return 0 if self.grid is None else self.grid.shape[0] * self.grid.shape[1] - self.num_points

    @property
    def size_x(self) -> float:
        """Size along x in meters."""
        return self.extent_x

    @property
    def size_y(self) -> float:
        """Size along y in meters."""
        return self.extent_y

    @property
    def height_fraction(self) -> float:
        """Height range relative to the lateral extent, at true scale."""
        return (self.value_range.hi - self.value_range.lo) / self.scale

    @property
    def color_scale(self) -> float:
        """Multiplier turning a normalized vertex z into a color fraction (with offset)."""
        return self.scale / (self.value_range.hi - self.value_range.lo)

    @property
    def color_offset(self) -> float:
        """Color fraction of a vertex at normalized z = 0."""
        return (self.z_center - self.value_range.lo) / (self.value_range.hi - self.value_range.lo)

    def auto_z_factor(self) -> float:
        """Return the exaggeration making the typical height span a part of the extent."""
        low, high = np.percentile(self.points[:, 2], ROBUST_PERCENTILES)
        fraction = (high - low) / self.scale
        if fraction <= 0:
            return 1.0
        return clamp_z_factor(min(AUTO_HEIGHT_FRACTION / fraction, AUTO_Z_FACTOR_MAX))
