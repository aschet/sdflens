# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import numpy as np
import pytest

from helpers import COLS, ROWS, make_cloud, make_ramp, make_sdf
from sdflens.camera import Camera
from sdflens.mesh import SurfaceMesh
from sdflens.pick import pick, pick_face, pick_vertex, project
from sdflens.pointcloud import PointCloudModel
from sdflens.surface import SurfaceModel

SIZE = (200.0, 200.0)


def _scene() -> tuple[np.ndarray, np.ndarray]:
    return Camera().matrices(1.0)


def test_the_nearest_vertex_within_the_radius_is_picked() -> None:
    view, proj = _scene()
    vertices = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.4, 0.0, 0.0]], np.float32)
    screen, _ = project(vertices, view, proj, 1.0, *SIZE)
    valid = np.ones(3, dtype=bool)

    for index in range(3):
        position = (float(screen[index, 0]) + 0.5, float(screen[index, 1]))
        assert pick_vertex(vertices, valid, view, proj, 1.0, position, SIZE, 6.0) == index
    far = (float(screen[2, 0]) + 30.0, float(screen[2, 1]))
    assert pick_vertex(vertices, valid, view, proj, 1.0, far, SIZE, 6.0) is None


def test_points_that_are_not_measured_are_not_picked() -> None:
    view, proj = _scene()
    vertices = np.zeros((2, 3), np.float32)
    position = (100.0, 100.0)

    assert pick_vertex(vertices, np.array([False, True]), view, proj, 1.0, position, SIZE, 6.0) == 1
    assert (
        pick_vertex(vertices, np.array([False, False]), view, proj, 1.0, position, SIZE, 6.0)
        is None
    )


def test_the_surface_in_front_hides_the_one_behind_it() -> None:
    camera = Camera()
    camera.set_view(-90.0, 0.0)  # look along +y: the larger y is the farther
    view, proj = camera.matrices(1.0)
    vertices = np.array([[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]], np.float32)
    valid = np.ones(2, dtype=bool)

    assert pick_vertex(vertices, valid, view, proj, 1.0, (100.0, 100.0), SIZE, 6.0) == 1


def test_points_behind_the_camera_are_not_picked() -> None:
    camera = Camera()
    camera.set_view(-90.0, 0.0)
    view, proj = camera.matrices(1.0)
    behind = np.array([[0.0, -100.0, 0.0]], np.float32)

    assert pick_vertex(behind, np.ones(1, bool), view, proj, 1.0, (100.0, 100.0), SIZE, 1e9) is None


def test_the_height_exaggeration_moves_the_points_on_the_screen() -> None:
    view, proj = _scene()
    vertices = np.array([[0.0, 0.0, 0.2]], np.float32)

    flat, _ = project(vertices, view, proj, 1.0, *SIZE)
    tall, _ = project(vertices, view, proj, 3.0, *SIZE)

    assert tall[0, 1] < flat[0, 1]  # higher on the screen, where y grows downwards


def test_the_mesh_gives_the_position_of_a_vertex_in_meters() -> None:
    data = make_ramp()
    model = SurfaceModel.from_sdf(make_sdf(data, x_scale=1e-6, y_scale=2e-6))
    mesh = SurfaceMesh.from_model(model)

    for row, column in ((0, 0), (2, 3), (ROWS - 1, COLS - 1)):
        x, y, z = mesh.position(row * COLS + column)
        assert x == pytest.approx(column * 1e-6, abs=1e-12)
        assert y == pytest.approx(row * 2e-6, abs=1e-12)
        assert z == pytest.approx(data[row, column], abs=1e-12)


def test_the_mesh_of_a_point_cloud_gives_the_positions_of_the_points() -> None:
    points = make_cloud()
    mesh = SurfaceMesh.from_point_cloud(PointCloudModel.from_points(points))

    for index in (0, 17, 199):
        # The vertices are float32: the positions agree to a tenth of a nanometer.
        np.testing.assert_allclose(mesh.position(index), points[index], rtol=0, atol=1e-10)


def _triangle() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    view, proj = _scene()
    vertices = np.array([[-0.3, -0.3, 0.0], [0.3, -0.3, 0.0], [0.0, 0.3, 0.0]], np.float32)
    screen, _ = project(vertices, view, proj, 1.0, *SIZE)
    return vertices, np.array([0, 1, 2], np.uint32), view, proj, screen


def test_a_click_on_a_face_picks_its_nearest_corner_far_from_any_vertex() -> None:
    vertices, triangles, view, proj, screen = _triangle()
    middle = screen.mean(axis=0)

    for corner in range(3):
        spot = middle + 0.4 * (screen[corner] - middle)  # inside, tens of pixels from the corner
        assert np.hypot(*(spot - screen[corner])) > 6.0
        assert pick_face(vertices, triangles, view, proj, 1.0, tuple(spot), SIZE) == corner
        valid = np.ones(3, dtype=bool)
        assert pick(vertices, valid, triangles, view, proj, 1.0, tuple(spot), SIZE, 6.0) == corner


def test_a_click_beside_the_faces_picks_a_vertex_only_when_one_is_near() -> None:
    vertices, triangles, view, proj, screen = _triangle()
    valid = np.ones(3, dtype=bool)
    outside = (float(screen[:, 0].min()) - 40.0, float(screen[0, 1]))
    near_corner = (float(screen[0, 0]) - 3.0, float(screen[0, 1]))

    assert pick_face(vertices, triangles, view, proj, 1.0, outside, SIZE) is None
    assert pick(vertices, valid, triangles, view, proj, 1.0, outside, SIZE, 6.0) is None
    assert pick(vertices, valid, triangles, view, proj, 1.0, near_corner, SIZE, 6.0) == 0


def test_the_front_face_wins_over_one_behind_it() -> None:
    camera = Camera()
    camera.set_view(-90.0, 0.0)
    view, proj = camera.matrices(1.0)
    # Two triangles seen head on, one at y = -0.3 (near) and one at y = 0.3 (far).
    far = [[-0.2, 0.3, -0.2], [0.2, 0.3, -0.2], [0.0, 0.3, 0.2]]
    near = [[-0.2, -0.3, -0.2], [0.2, -0.3, -0.2], [0.0, -0.3, 0.2]]
    vertices = np.array([*far, *near], np.float32)
    triangles = np.arange(6, dtype=np.uint32)

    assert pick_face(vertices, triangles, view, proj, 1.0, (100.0, 100.0), SIZE) in (3, 4, 5)
