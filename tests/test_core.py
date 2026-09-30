# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np
import pytest

from sdflens.camera import Camera
from sdflens.colormap import apply_colormap, build_lut, colormap_names
from sdflens.units import format_length, nice_ticks, unit_for
from sdflens.view2d import Bounds, ViewTransform2D
from sdflens.zscale import (
    SLIDER_MAX,
    Z_FACTOR_MAX,
    Z_FACTOR_MIN,
    factor_to_slider,
    slider_to_factor,
)


@pytest.mark.parametrize("name", colormap_names())
def test_luts_are_valid_and_reversible(name: str) -> None:
    lut = build_lut(name)

    assert lut.shape == (256, 3)
    assert lut.dtype == np.uint8
    assert np.array_equal(build_lut(name, reverse=True), lut[::-1])


def test_color_maps_are_the_published_tables() -> None:
    assert colormap_names() == ["Viridis", "Plasma", "Inferno", "Magma", "Turbo", "Grayscale"]
    assert build_lut("Viridis")[[0, 255]].tolist() == [[68, 1, 84], [253, 231, 37]]
    assert build_lut("Magma")[[0, 255]].tolist() == [[0, 0, 4], [252, 253, 191]]
    assert build_lut("Turbo")[[0, 255]].tolist() == [[48, 18, 59], [122, 4, 3]]


def test_nan_and_out_of_range_values_never_use_the_lut_incorrectly() -> None:
    lut = build_lut("Grayscale")
    rgb = apply_colormap(np.array([[np.nan, -5.0, 5.0, np.inf]]), lut, 0.0, 1.0)

    assert rgb[0, 0].tolist() == [0, 0, 0, 0]
    assert rgb[0, 1].tolist() == [0, 0, 0, 255]
    assert rgb[0, 2].tolist() == [255, 255, 255, 255]
    assert rgb[0, 3].tolist() == [0, 0, 0, 0]


def test_zero_span_range_maps_without_error() -> None:
    rgb = apply_colormap(np.array([[1.0, np.nan]]), build_lut("Viridis"), 1.0, 1.0)

    assert rgb.shape == (1, 2, 4)


def test_slider_mapping_round_trips() -> None:
    assert factor_to_slider(Z_FACTOR_MIN) == 0
    assert factor_to_slider(Z_FACTOR_MAX) == SLIDER_MAX
    for factor in (0.5, 1.0, 37.0, 900.0):
        assert slider_to_factor(factor_to_slider(factor)) == pytest.approx(factor, rel=0.01)


def test_units_and_ticks() -> None:
    assert unit_for(3e-3) == (1e-3, "mm")
    assert unit_for(2.5e-2) == (1e-2, "cm")
    assert unit_for(0.5) == (1e-2, "cm")
    assert unit_for(2.0) == (1.0, "m")
    assert format_length(0.015) == "1.5 cm"
    assert unit_for(0.0) == (1.0, "m")
    assert format_length(-2.5e-9) == "-2.5 nm"
    assert nice_ticks(0.0, 0.0) == [0.0]
    assert nice_ticks(float("nan"), 1.0) == []
    ticks = nice_ticks(-1.3, 5.6)
    assert ticks[0] >= -1.3
    assert ticks[-1] <= 5.6
    assert 3 <= len(ticks) <= 12


def test_camera_home_shows_x_to_the_right_with_z_up() -> None:
    camera = Camera()
    camera.fit(1.0)
    view, proj = camera.matrices(1.0)

    assert view[0, 0] > 0  # world +x maps to screen right
    assert view[1, 2] > 0  # world +z maps to screen up
    origin = proj @ view @ np.array([0.0, 0.0, 0.0, 1.0])
    ndc = origin[:3] / origin[3]
    assert abs(ndc[0]) < 1e-9
    assert abs(ndc[1]) < 1e-9
    assert -1.0 < ndc[2] < 1.0


def test_camera_set_view_looks_along_an_axis_and_limits_the_elevation() -> None:
    camera = Camera()
    camera.fit(1.0)
    camera.set_view(0.0, 0.0)
    assert np.allclose(camera.eye() / camera.distance, [1.0, 0.0, 0.0])
    camera.set_view(90.0, 0.0)
    assert np.allclose(camera.eye() / camera.distance, [0.0, 1.0, 0.0])
    camera.set_view(-90.0, 90.0)
    assert camera.elevation == 89.0


def test_camera_limits_and_pan() -> None:
    camera = Camera()
    camera.fit(1.0)
    camera.rotate(0.0, 10_000.0)
    assert camera.elevation == 89.0
    camera.zoom(1e9)
    assert camera.distance == pytest.approx(0.05)
    before = camera.target.copy()
    camera.pan(10.0, 0.0, 500.0)
    assert not np.allclose(camera.target, before)
    camera.fit(1.0)
    assert np.allclose(camera.target, 0.0)


def test_view_transform_zoom_keeps_the_anchor_fixed() -> None:
    transform = ViewTransform2D()
    transform.fit(400, 300, Bounds(0.0, 0.0, 4.0, 3.0))
    anchor = transform.screen_to_world(120.0, 80.0)
    transform.zoom_at(3.0, 120.0, 80.0)

    assert transform.screen_to_world(120.0, 80.0) == pytest.approx(anchor)
    transform.zoom_at(1e12, 0.0, 0.0)
    assert transform.scale < 1e12


def test_view_transform_pan_moves_content_with_the_cursor() -> None:
    transform = ViewTransform2D()
    transform.fit(400, 300, Bounds(0.0, 0.0, 4.0, 3.0))
    before = transform.world_to_screen(1.0, 1.0)
    transform.pan(10.0, -5.0)
    after = transform.world_to_screen(1.0, 1.0)

    assert after[0] - before[0] == pytest.approx(10.0)
    assert after[1] - before[1] == pytest.approx(-5.0)
