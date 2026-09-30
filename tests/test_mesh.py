# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np
import pytest

from helpers import COLS, ROWS, make_ramp, make_sdf
from sdflens.mesh import SurfaceMesh
from sdflens.surface import SurfaceModel


def build(data: np.ndarray, **scales: float) -> tuple[SurfaceModel, SurfaceMesh]:
    model = SurfaceModel.from_sdf(make_sdf(data, **scales))
    return model, SurfaceMesh.from_model(model)


def test_mesh_never_references_or_contains_nan() -> None:
    data = make_ramp()
    data[1, 1] = np.nan
    model, mesh = build(data)

    assert np.isfinite(mesh.vertices).all()
    # Four cells touch (1, 1) and the corner cell touches (0, COLS - 1).
    assert mesh.triangles.size == 2 * 3 * ((ROWS - 1) * (COLS - 1) - 4 - 1)
    assert (1 * COLS + 1) not in mesh.triangles
    assert (0 * COLS + COLS - 1) not in mesh.triangles
    assert model.invalid_count == 2


def test_coordinates_are_right_handed_and_normalized() -> None:
    _, mesh = build(make_ramp(), x_scale=1e-6, y_scale=2e-6)
    grid = mesh.vertices.reshape(ROWS, COLS, 3)

    assert grid[0, -1, 0] > grid[0, 0, 0]  # x grows with the column index
    assert grid[-1, 0, 1] > grid[0, 0, 1]  # y grows with the row index
    assert grid[1, 1, 2] > grid[0, 0, 2]  # z is the height
    x_extent = grid[..., 0].max() - grid[..., 0].min()
    y_extent = grid[..., 1].max() - grid[..., 1].min()
    assert max(x_extent, y_extent) == pytest.approx(1.0)
    assert y_extent / x_extent == pytest.approx(3 * 2e-6 / (4 * 1e-6))


def test_triangles_face_positive_z() -> None:
    _, mesh = build(np.zeros((3, 3)))
    a, b, c = (mesh.vertices[i] for i in mesh.triangles[:3])

    assert np.cross(b - a, c - a)[2] > 0


def test_profile_uses_line_segments_and_skips_nan() -> None:
    model, mesh = build(np.array([[0.0, 1e-9, np.nan, 3e-9, 4e-9]]))

    assert model.is_profile
    assert mesh.triangles.size == 0
    assert mesh.lines.tolist() == [0, 1, 3, 4]


def test_areal_lines_connect_neighboring_measured_points() -> None:
    _, mesh = build(np.array([[0.0, 0.0], [0.0, np.nan]]))

    assert mesh.lines.tolist() == [0, 1, 0, 2]


def test_large_grids_are_thinned_for_the_mesh_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sdflens.mesh.MAX_MESH_POINTS", 6)
    full = make_ramp()
    model, mesh = build(full)

    assert mesh.step > 1
    assert model.data.shape == full.shape
    assert model.valid.shape == full.shape
    assert mesh.vertices.shape[0] == mesh.valid.size
    assert mesh.valid.size < full.size
    assert mesh.triangles.max() < mesh.vertices.shape[0]
    x_extent = mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min()
    assert x_extent == pytest.approx(model.extent_x / model.scale, rel=1e-5)


def test_small_grids_keep_every_point() -> None:
    model, mesh = build(make_ramp())

    assert mesh.step == 1
    assert mesh.vertices.shape[0] == model.data.size


def test_color_mapping_spans_the_range() -> None:
    model, mesh = build(make_ramp())
    fractions = mesh.vertices[:, 2].astype(np.float64) * model.color_scale + model.color_offset
    measured = fractions[np.isfinite(model.data).ravel()]

    assert measured.min() == pytest.approx(0.0, abs=1e-5)
    assert measured.max() == pytest.approx(1.0, abs=1e-5)
