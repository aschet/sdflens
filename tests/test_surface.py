# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np
import pytest

from helpers import COLS, ROWS, make_ramp, make_sdf
from sdflens.surface import (
    NoMeasuredPointsError,
    SurfaceModel,
    data_range,
)


def test_range_and_mean_ignore_nan() -> None:
    model = SurfaceModel.from_sdf(make_sdf(make_ramp()))

    assert model.value_range.lo == 0.0
    assert model.value_range.hi == pytest.approx((ROWS * COLS - 1) * 1e-9)
    valid = model.data[np.isfinite(model.data)]
    assert model.z_mean == pytest.approx(valid.mean())


def test_all_nan_is_rejected() -> None:
    with pytest.raises(NoMeasuredPointsError):
        SurfaceModel.from_sdf(make_sdf(np.full((3, 3), np.nan)))


def test_constant_surface_gets_a_usable_range() -> None:
    value_range = data_range(np.full((3, 3), 2e-6))

    assert value_range.lo < 2e-6 < value_range.hi
    assert data_range(np.zeros((2, 2))).hi > 0.0


def test_size_counts_every_point_as_one_sampling_interval() -> None:
    model = SurfaceModel.from_sdf(make_sdf(make_ramp(), x_scale=1e-6, y_scale=2e-6))

    assert model.size_x == pytest.approx(COLS * 1e-6)
    assert model.size_y == pytest.approx(ROWS * 2e-6)
    profile = SurfaceModel.from_sdf(make_sdf(np.array([[0.0, 1e-9, 2e-9]])))
    assert profile.size_y == 0.0


def test_profile_pixel_size_falls_back_to_the_x_scale() -> None:
    model = SurfaceModel.from_sdf(make_sdf(np.array([[0.0, 1e-9, 2e-9]])))

    assert model.is_profile
    assert model.pixel_size_y == model.x_scale
