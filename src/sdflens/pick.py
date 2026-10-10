# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Picking in the 3D view: the measured point nearest to a position on the screen."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["pick", "pick_face", "pick_vertex", "project"]

_CHUNK = 1 << 20  # triangles tested at a time
# Barycentric coordinates may be this far below zero, so that a click on an edge hits.
_EDGE_TOLERANCE = 1e-6

# Points of the visible surface lie at most this many pixel footprints behind the nearest point
# within the radius; points that are farther behind belong to a surface that is hidden.
_DEPTH_SLOPE = 3.0


def project(
    vertices: NDArray[np.float32],
    view: NDArray[np.float64],
    proj: NDArray[np.float64],
    z_factor: float,
    width: float,
    height: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return the screen positions in pixels, y down, and the depths of ``vertices``.

    The depth is the distance in front of the camera; it is not positive for a point behind it.

    >>> from sdflens.camera import Camera
    >>> view, proj = Camera().matrices(1.0)
    >>> screen, depth = project(np.zeros((1, 3), np.float32), view, proj, 1.0, 100.0, 100.0)
    >>> screen.round(3).tolist(), bool(depth[0] > 0)
    ([[50.0, 50.0]], True)
    """
    matrix = (proj @ view).astype(np.float32)
    scaled = vertices * np.array([1.0, 1.0, z_factor], dtype=np.float32)
    clip = scaled @ matrix[:, :3].T + matrix[:, 3]
    depth = clip[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        screen_x = (clip[:, 0] / depth + 1.0) * 0.5 * width
        screen_y = (1.0 - clip[:, 1] / depth) * 0.5 * height
    return np.stack([screen_x, screen_y], axis=1), depth


def pick_face(
    vertices: NDArray[np.float32],
    triangles: NDArray[np.uint32],
    view: NDArray[np.float64],
    proj: NDArray[np.float64],
    z_factor: float,
    position: tuple[float, float],
    size: tuple[float, float],
) -> int | None:
    """Return the vertex nearest to ``position`` in the front triangle there, if any.

    ``triangles`` holds three vertex indices per triangle. The vertex is the corner of the
    triangle that the position is closest to; ``None`` if no triangle covers the position.

    >>> from sdflens.camera import Camera
    >>> view, proj = Camera().matrices(1.0)
    >>> vertices = np.array([[-0.3, -0.3, 0], [0.3, -0.3, 0], [0.0, 0.3, 0]], np.float32)
    >>> triangles = np.array([0, 1, 2], np.uint32)
    >>> pick_face(vertices, triangles, view, proj, 1.0, (83.5, 105.5), (200.0, 200.0))
    0
    >>> pick_face(vertices, triangles, view, proj, 1.0, (5.0, 5.0), (200.0, 200.0)) is None
    True
    """
    screen, depth = project(vertices, view, proj, z_factor, *size)
    corners = triangles.reshape(-1, 3)
    best_depth, best_vertex = np.inf, None
    px, py = position
    for start in range(0, len(corners), _CHUNK):
        chunk = corners[start : start + _CHUNK]
        ax, ay = screen[chunk[:, 0], 0], screen[chunk[:, 0], 1]
        bx, by = screen[chunk[:, 1], 0], screen[chunk[:, 1], 1]
        cx, cy = screen[chunk[:, 2], 0], screen[chunk[:, 2], 1]
        covers = (
            (np.minimum(np.minimum(ax, bx), cx) <= px)
            & (np.maximum(np.maximum(ax, bx), cx) >= px)
            & (np.minimum(np.minimum(ay, by), cy) <= py)
            & (np.maximum(np.maximum(ay, by), cy) >= py)
        )
        hit = np.flatnonzero(covers)
        if hit.size == 0:
            continue
        ax, ay, bx, by, cx, cy = (v[hit] for v in (ax, ay, bx, by, cx, cy))
        dx0, dy0, dx1, dy1 = bx - ax, by - ay, cx - ax, cy - ay
        dx2, dy2 = px - ax, py - ay
        d00, d01, d11 = dx0 * dx0 + dy0 * dy0, dx0 * dx1 + dy0 * dy1, dx1 * dx1 + dy1 * dy1
        d20, d21 = dx2 * dx0 + dy2 * dy0, dx2 * dx1 + dy2 * dy1
        denominator = d00 * d11 - d01 * d01
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (d11 * d20 - d01 * d21) / denominator
            w = (d00 * d21 - d01 * d20) / denominator
        u = 1.0 - v - w
        weights = np.stack([u, v, w], axis=1)
        corner_depth = depth[chunk[hit]]
        inside = (
            (denominator != 0.0)
            & (weights >= -_EDGE_TOLERANCE).all(axis=1)
            & (corner_depth > 0.0).all(axis=1)
        )
        if not inside.any():
            continue
        # The inverse of the depth is linear on the screen.
        weights, corner_depth = weights[inside], corner_depth[inside]
        covered = 1.0 / (weights / corner_depth).sum(axis=1)
        nearest = int(np.argmin(covered))
        if covered[nearest] < best_depth:
            best_depth = float(covered[nearest])
            best_vertex = int(chunk[hit[inside][nearest], int(np.argmax(weights[nearest]))])
    return best_vertex


def pick(
    vertices: NDArray[np.float32],
    valid: NDArray[np.bool_],
    triangles: NDArray[np.uint32],
    view: NDArray[np.float64],
    proj: NDArray[np.float64],
    z_factor: float,
    position: tuple[float, float],
    size: tuple[float, float],
    radius: float,
) -> int | None:
    """Return the index of the measured vertex that a click at ``position`` picks, if any.

    A click on a triangle picks its nearest corner. Otherwise the nearest vertex within
    ``radius`` pixels is picked, which is how the points of a cloud and the points that no
    triangle uses are picked.
    """
    if triangles.size:
        index = pick_face(vertices, triangles, view, proj, z_factor, position, size)
        if index is not None:
            return index
    return pick_vertex(vertices, valid, view, proj, z_factor, position, size, radius)


def pick_vertex(
    vertices: NDArray[np.float32],
    valid: NDArray[np.bool_],
    view: NDArray[np.float64],
    proj: NDArray[np.float64],
    z_factor: float,
    position: tuple[float, float],
    size: tuple[float, float],
    radius: float,
) -> int | None:
    """Return the index of the visible measured vertex nearest to ``position``, if any.

    Only vertices within ``radius`` pixels count. Of these the ones on the front surface are
    kept, and the one nearest to ``position`` on the screen is returned; ``None`` if there is
    none, as when the position is on the background.

    >>> from sdflens.camera import Camera
    >>> view, proj = Camera().matrices(1.0)
    >>> vertices = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], np.float32)
    >>> pick_vertex(vertices, np.ones(2, bool), view, proj, 1.0, (52.0, 50.0), (100.0, 100.0), 5.0)
    0
    """
    candidates = np.flatnonzero(valid.ravel())
    screen, depth = project(vertices[candidates], view, proj, z_factor, *size)
    distance = np.hypot(screen[:, 0] - position[0], screen[:, 1] - position[1])
    near = np.flatnonzero((depth > 0.0) & (distance <= radius))
    if near.size == 0:
        return None
    nearest = float(depth[near].min())
    footprint = 2.0 * nearest / (proj[1, 1] * size[1])
    front = near[depth[near] <= nearest + _DEPTH_SLOPE * radius * footprint]
    return int(candidates[front[np.argmin(distance[front])]])
