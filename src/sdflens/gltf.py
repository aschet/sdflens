# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The geometry of the 3D view as a binary glTF (GLB) file, written with numpy only."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from . import __version__
from .mesh import SurfaceMesh
from .surface import Model3D

__all__ = ["glb_bytes", "write_glb"]

_FLOAT, _UNSIGNED_INT = 5126, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963
_POINTS, _TRIANGLES = 0, 4


def _normals(vertices: NDArray[np.float32], triangles: NDArray[np.integer]) -> NDArray[np.float32]:
    """Return unit vertex normals: the face normals around each vertex, weighted by area."""
    corners = triangles.reshape(-1, 3)
    a, b, c = (vertices[corners[:, i]].astype(np.float64) for i in range(3))
    faces = np.cross(b - a, c - a)
    sums = np.zeros(vertices.shape, dtype=np.float64)
    for i in range(3):
        for axis in range(3):
            sums[:, axis] += np.bincount(
                corners[:, i], weights=faces[:, axis], minlength=len(vertices)
            )
    length = np.linalg.norm(sums, axis=1, keepdims=True)
    length[length == 0] = 1.0
    return (sums / length).astype(np.float32)


def _upright(vertices: NDArray[np.float32]) -> NDArray[np.float32]:
    """Turn z up into y up, which glTF has."""
    return np.stack([vertices[:, 0], vertices[:, 2], -vertices[:, 1]], axis=-1)


def _pad(data: bytes, filler: bytes = b"\x00") -> bytes:
    return data + filler * (-len(data) % 4)


def glb_bytes(mesh: SurfaceMesh, model: Model3D) -> bytes:
    """Return the GLB file of ``mesh``: its triangles with normals, or its points.

    The geometry is the real one, in meters, which glTF uses for its units, so that distances can
    be measured in the file. The height is not exaggerated and the vertices have no colors; a
    viewer can do both. The model is centered at the origin; where it lay in the coordinates of
    the data is stored in the ``extras`` of the asset.
    """
    if mesh.is_cloud:
        used = np.arange(len(mesh.vertices))
        triangles = np.empty(0, dtype=np.uint32)
    else:
        # Only the vertices of measured points are written; the others have placeholders.
        used = np.flatnonzero(mesh.valid.reshape(-1))
        renumber = np.full(len(mesh.vertices), -1, dtype=np.int64)
        renumber[used] = np.arange(len(used))
        triangles = renumber[mesh.triangles].astype(np.uint32)
    if len(used) == 0:
        raise ValueError("There are no measured points to export")

    # The mesh is normalized; the normals do not depend on the size, the positions are in meters.
    vertices = np.ascontiguousarray(_upright(mesh.vertices[used]))
    positions = np.ascontiguousarray((vertices.astype(np.float64) * mesh.scale).astype(np.float32))
    arrays: list[tuple[bytes, int, str, int]] = [
        (positions.tobytes(), _ARRAY_BUFFER, "VEC3", _FLOAT)
    ]
    attributes = {"POSITION": 0}
    counts = [len(vertices)]
    if len(triangles):
        arrays.append((_normals(vertices, triangles).tobytes(), _ARRAY_BUFFER, "VEC3", _FLOAT))
        attributes["NORMAL"] = 1
        counts.append(len(vertices))
        arrays.append((triangles.tobytes(), _ELEMENT_ARRAY_BUFFER, "SCALAR", _UNSIGNED_INT))
        counts.append(triangles.size)

    binary = b""
    views: list[dict[str, object]] = []
    accessors: list[dict[str, object]] = []
    for (data, target, kind, component), count in zip(arrays, counts, strict=True):
        views.append(
            {"buffer": 0, "byteOffset": len(binary), "byteLength": len(data), "target": target}
        )
        accessors.append(
            {"bufferView": len(views) - 1, "componentType": component, "count": count, "type": kind}
        )
        binary += _pad(data)
    accessors[0]["min"] = [float(value) for value in positions.min(axis=0)]
    accessors[0]["max"] = [float(value) for value in positions.max(axis=0)]

    primitive: dict[str, object] = {
        "attributes": attributes,
        "material": 0,
        "mode": _TRIANGLES if len(triangles) else _POINTS,
    }
    if len(triangles):
        primitive["indices"] = 2
    document = {
        "asset": {
            "version": "2.0",
            "generator": f"sdflens {__version__}",
            "extras": {
                "unit": "meters",
                "extent_meters": [model.extent_x, model.extent_y],
                "origin_meters": list(mesh.origin),
            },
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "surface"}],
        "meshes": [{"primitives": [primitive]}],
        # Without a material glTF means a fully metallic one, which looks dark in most viewers.
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.3, 0.3, 0.3, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.85,
                },
                "doubleSided": True,
            }
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    text = _pad(json.dumps(document, separators=(",", ":")).encode(), b" ")
    total = 12 + 8 + len(text) + 8 + len(binary)
    return (
        struct.pack("<III", 0x46546C67, 2, total)  # "glTF", version 2
        + struct.pack("<II", len(text), 0x4E4F534A)  # "JSON"
        + text
        + struct.pack("<II", len(binary), 0x004E4942)  # "BIN"
        + binary
    )


def write_glb(path: str | Path, mesh: SurfaceMesh, model: Model3D) -> None:
    """Write the GLB file of ``mesh`` to ``path``; see :func:`glb_bytes`."""
    Path(path).write_bytes(glb_bytes(mesh, model))
