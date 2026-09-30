# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared test data builders."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sdfio import SdfFile, SdfHeader

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
