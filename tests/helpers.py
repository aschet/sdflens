# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared test data builders."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sdfio import SdfFile, SdfHeader
from x3pio import X3pFile

ROWS, COLS = 4, 5


def make_ramp() -> NDArray[np.float64]:
    """Asymmetric ramp (meters) with a landmark NaN at row 0, last column."""
    data: NDArray[np.float64] = np.fromfunction(lambda i, j: (i * COLS + j) * 1e-9, (ROWS, COLS))
    data[0, COLS - 1] = np.nan
    return data


def make_sdf(data: NDArray[np.float64], x_scale: float = 1e-6, y_scale: float = 2e-6) -> SdfFile:
    """Build an in-memory SDF file around ``data``."""
    header = SdfHeader(
        num_points=data.shape[1],
        num_profiles=data.shape[0],
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=1e-9,
    )
    return SdfFile(header=header, data=data)


def make_x3p(
    data: NDArray[np.float64] | None = None, x_scale: float = 1e-6, y_scale: float = 2e-6
) -> X3pFile:
    """Build an in-memory x3p surface around ``data``, with two vendor extensions."""
    x3p = X3pFile.from_array(
        make_ramp() if data is None else data, x_scale=x_scale, y_scale=y_scale
    )
    x3p.update_metadata(creator="Jane Doe", comment="Test surface")
    x3p.extensions["http://www.vendor.com/mypath/a.xml"] = b"<a/>"
    x3p.extensions["http://www.vendor.com/image.png"] = b"x" * 1500
    return x3p


def make_cloud() -> NDArray[np.float64]:
    """Points of a tilted plane in meters; a point cloud file cannot hold missing points."""
    rng = np.random.default_rng(7)
    points = rng.uniform(0.0, 1e-3, (200, 3))
    points[:, 2] = 0.1 * points[:, 0]
    return points
