# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Surface model: coordinates and statistics of the height data of an SDF file."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sdfio import SdfFile

from .zscale import clamp_z_factor

__all__ = ["NoMeasuredPointsError", "SurfaceModel", "ValueRange", "data_range"]

#: Fraction of the lateral extent the surface height should span by default.
AUTO_HEIGHT_FRACTION = 0.15

#: The automatic exaggeration stops here, since more mostly amplifies noise.
AUTO_Z_FACTOR_MAX = 100.0

#: Percentiles of the measured heights the automatic exaggeration fits, ignoring outliers.
ROBUST_PERCENTILES = (0.5, 99.5)


class NoMeasuredPointsError(ValueError):
    """The file contains no measured (finite) height values."""


@dataclass(frozen=True)
class ValueRange:
    """Closed interval of height values in meters."""

    lo: float
    hi: float


def data_range(data: NDArray[np.float64]) -> ValueRange:
    """Return the min/max of the measured points, widened if the surface is constant.

    :raises NoMeasuredPointsError: If every point is non-measured.

    >>> data_range(np.array([[1.0, np.nan], [3.0, 2.0]]))
    ValueRange(lo=1.0, hi=3.0)
    """
    valid = np.isfinite(data)
    if not valid.any():
        raise NoMeasuredPointsError("The file contains no measured points")
    lo = float(data[valid].min())
    hi = float(data[valid].max())
    if hi == lo:
        pad = 0.5 * abs(lo) if lo != 0.0 else 1e-9
        return ValueRange(lo - pad, hi + pad)
    return ValueRange(lo, hi)


@dataclass(frozen=True, eq=False)
class SurfaceModel:
    """Height data of an SDF file with its coordinate scales and value statistics."""

    data: NDArray[np.float64]
    x_scale: float
    y_scale: float
    valid: NDArray[np.bool_]
    value_range: ValueRange
    z_mean: float
    scale: float
    z_center: float

    @classmethod
    def from_sdf(cls, sdf: SdfFile) -> SurfaceModel:
        """Build the model from a parsed SDF file.

        :raises NoMeasuredPointsError: If every point is non-measured.
        """
        data = np.asarray(sdf.data, dtype=np.float64)
        x = sdf.x_axis
        y = sdf.y_axis
        valid = np.isfinite(data)
        value_range = data_range(data)

        extent_x = float(x[-1] - x[0])
        extent_y = float(y[-1] - y[0])
        scale = max(extent_x, extent_y) or max(sdf.header.x_scale, 1e-12)
        z_center = 0.5 * (value_range.lo + value_range.hi)

        return cls(
            data=data,
            x_scale=float(sdf.header.x_scale),
            y_scale=float(sdf.header.y_scale),
            valid=valid,
            value_range=value_range,
            z_mean=float(data[valid].mean()),
            scale=scale,
            z_center=z_center,
        )

    @property
    def num_profiles(self) -> int:
        """Number of rows (y direction)."""
        return int(self.data.shape[0])

    @property
    def num_points(self) -> int:
        """Number of columns (x direction)."""
        return int(self.data.shape[1])

    @property
    def is_profile(self) -> bool:
        """Whether the data is a single profile or column rather than an areal surface."""
        return self.num_profiles < 2 or self.num_points < 2

    @property
    def extent_x(self) -> float:
        """Lateral extent along x in meters."""
        return (self.num_points - 1) * self.x_scale

    @property
    def extent_y(self) -> float:
        """Lateral extent along y in meters; zero for a profile."""
        return (self.num_profiles - 1) * self.y_scale

    @property
    def size_x(self) -> float:
        """Measured size along x in meters: each point covers one sampling interval."""
        return self.num_points * self.x_scale

    @property
    def size_y(self) -> float:
        """Measured size along y in meters; zero for a profile."""
        return self.num_profiles * self.y_scale if self.num_profiles > 1 else 0.0

    @property
    def pixel_size_y(self) -> float:
        """Row spacing in meters for display; a profile has no meaningful ``y_scale``."""
        return self.y_scale if self.num_profiles > 1 else self.x_scale

    @property
    def invalid_count(self) -> int:
        """Number of non-measured points."""
        return int(self.valid.size - np.count_nonzero(self.valid))

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
        """Return the exaggeration making the typical height span a part of the extent.

        Outliers are ignored (see :data:`ROBUST_PERCENTILES`) and the result is at most
        :data:`AUTO_Z_FACTOR_MAX`; a surface without relief stays at 1.
        """
        low, high = np.percentile(self.data[self.valid], ROBUST_PERCENTILES)
        fraction = (high - low) / self.scale
        if fraction <= 0:
            return 1.0
        return clamp_z_factor(min(AUTO_HEIGHT_FRACTION / fraction, AUTO_Z_FACTOR_MAX))
