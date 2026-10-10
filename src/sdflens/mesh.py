# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Render mesh of a surface or point cloud (NaN-aware vertices, triangles, lines), with numpy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .pointcloud import PointCloudModel
from .surface import SurfaceModel

__all__ = ["MAX_MESH_POINTS", "SurfaceMesh"]

#: Grids with more points are thinned out for the mesh (about 2000 x 2000).
MAX_MESH_POINTS = 4_000_000


@dataclass(frozen=True, eq=False)
class SurfaceMesh:
    """Vertices and primitive indices drawing a :class:`SurfaceModel` or a point cloud.

    Vertex positions are centered and divided by :attr:`SurfaceModel.scale` (the largest
    lateral extent in meters), keeping float32 well conditioned for micrometer-sized data.
    Non-measured points keep a placeholder vertex that no primitive references. Grids larger
    than :data:`MAX_MESH_POINTS` are thinned out by ``step`` along both axes; ``valid`` matches
    the thinned grid. A point cloud has no triangles or lines and is always drawn as points. An
    irregular surface is a grid like any other, with the positions of its points.
    """

    vertices: NDArray[np.float32]
    triangles: NDArray[np.uint32]
    lines: NDArray[np.uint32]
    valid: NDArray[np.bool_]
    step: int
    is_cloud: bool = False

    @classmethod
    def from_model(cls, model: SurfaceModel) -> SurfaceMesh:
        """Build the mesh of the measured points of ``model``."""
        step = _step(model.num_profiles, model.num_points)
        rows = _sample_indices(model.num_profiles, step)
        cols = _sample_indices(model.num_points, step)
        data, valid = model.data, model.valid
        if step > 1:
            data, valid = data[np.ix_(rows, cols)], valid[np.ix_(rows, cols)]

        grid_x, grid_y = np.meshgrid(
            cols * model.x_scale - 0.5 * model.extent_x,
            rows * model.y_scale - 0.5 * model.extent_y,
        )
        vertices = np.stack(
            [
                grid_x / model.scale,
                grid_y / model.scale,
                np.where(valid, (data - model.z_center) / model.scale, 0.0),
            ],
            axis=-1,
        ).reshape(-1, 3)
        return cls(
            vertices=vertices.astype(np.float32),
            triangles=_triangle_indices(valid),
            lines=_line_indices(valid),
            valid=valid,
            step=step,
        )

    @classmethod
    def from_point_cloud(cls, model: PointCloudModel) -> SurfaceMesh:
        """Build the mesh of ``model``: its points, or for an irregular surface its grid."""
        if model.grid is not None:
            return cls._from_point_grid(model, model.grid)
        points = model.points
        vertices = np.stack(
            [
                (points[:, 0] - model.center_x) / model.scale,
                (points[:, 1] - model.center_y) / model.scale,
                (points[:, 2] - model.z_center) / model.scale,
            ],
            axis=-1,
        )
        empty = np.empty(0, dtype=np.uint32)
        return cls(
            vertices=vertices.astype(np.float32),
            triangles=empty,
            lines=empty,
            valid=np.ones(len(points), dtype=np.bool_),
            step=1,
            is_cloud=True,
        )

    @classmethod
    def _from_point_grid(cls, model: PointCloudModel, grid: NDArray[np.float64]) -> SurfaceMesh:
        """Build the mesh of an irregular surface: vertices at the points, joined as a grid."""
        step = _step(grid.shape[0], grid.shape[1])
        if step > 1:
            rows = _sample_indices(grid.shape[0], step)
            cols = _sample_indices(grid.shape[1], step)
            grid = grid[np.ix_(rows, cols)]
        valid = np.isfinite(grid).all(axis=-1)
        vertices = np.stack(
            [
                (grid[..., 0] - model.center_x) / model.scale,
                (grid[..., 1] - model.center_y) / model.scale,
                (grid[..., 2] - model.z_center) / model.scale,
            ],
            axis=-1,
        )
        vertices = np.where(valid[..., np.newaxis], vertices, 0.0).reshape(-1, 3)
        return cls(
            vertices=vertices.astype(np.float32),
            triangles=_triangle_indices(valid),
            lines=_line_indices(valid),
            valid=valid,
            step=step,
        )


def _step(rows: int, cols: int) -> int:
    """Return the stride along each axis that keeps the mesh within :data:`MAX_MESH_POINTS`."""
    ratio = rows * cols / MAX_MESH_POINTS
    if ratio <= 1:
        return 1
    return math.ceil(ratio if min(rows, cols) < 2 else math.sqrt(ratio))


def _sample_indices(count: int, step: int) -> NDArray[np.intp]:
    """Return every ``step``-th index of ``count`` items, always including the last one."""
    return np.unique(np.append(np.arange(0, count, step), count - 1))


def _triangle_indices(valid: NDArray[np.bool_]) -> NDArray[np.uint32]:
    """Two counter-clockwise (seen from +z) triangles per cell whose four corners are measured."""
    rows, cols = valid.shape
    if rows < 2 or cols < 2:
        return np.empty(0, dtype=np.uint32)
    cell_ok = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    row, col = np.nonzero(cell_ok)
    i00 = row.astype(np.int64) * cols + col
    i01 = i00 + 1
    i10 = i00 + cols
    i11 = i10 + 1
    return np.stack([i00, i01, i11, i00, i11, i10], axis=1).reshape(-1).astype(np.uint32)


def _line_indices(valid: NDArray[np.bool_]) -> NDArray[np.uint32]:
    """Line segments between all pairs of grid-neighboring measured points."""
    index = np.arange(valid.size, dtype=np.int64).reshape(valid.shape)
    along_x = valid[:, :-1] & valid[:, 1:]
    along_y = valid[:-1, :] & valid[1:, :]
    segments = np.concatenate(
        [
            np.stack([index[:, :-1][along_x], index[:, 1:][along_x]], axis=1),
            np.stack([index[:-1, :][along_y], index[1:, :][along_y]], axis=1),
        ]
    )
    return segments.reshape(-1).astype(np.uint32)
