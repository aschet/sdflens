# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The 3D view as a binary glTF (GLB) file, written with numpy only."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from . import __version__
from .mesh import SurfaceMesh
from .surface import Model3D

__all__ = ["ViewShape", "glb_bytes", "write_glb"]

_FLOAT, _UNSIGNED_BYTE, _UNSIGNED_INT = 5126, 5121, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963
_POINTS, _TRIANGLES = 0, 4


class ViewShape(NamedTuple):
    """What the 3D view shows: the model, its mesh and the color table of the colors."""

    model: Model3D
    mesh: SurfaceMesh
    lut: NDArray[np.uint8]


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


def _upright(vertices: NDArray[np.float32], z_factor: float) -> NDArray[np.float32]:
    """Apply the height exaggeration and turn z up into y up, which glTF has."""
    height = vertices[:, 2] * np.float32(z_factor)
    return np.stack([vertices[:, 0], height, -vertices[:, 1]], axis=-1)


def _linear(srgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return the sRGB bytes as linear bytes, which glTF asks for in vertex colors."""
    fraction = srgb.astype(np.float64) / 255.0
    linear = np.where(fraction <= 0.04045, fraction / 12.92, ((fraction + 0.055) / 1.055) ** 2.4)
    return np.rint(linear * 255.0).astype(np.uint8)


def _colors(
    mesh: SurfaceMesh, model: Model3D, lut: NDArray[np.uint8], used: NDArray[np.intp]
) -> NDArray[np.uint8]:
    """Return the linear RGBA color of the ``used`` vertices, looked up from the height."""
    fraction = mesh.vertices[used, 2].astype(np.float64) * model.color_scale + model.color_offset
    indices = np.rint(np.clip(fraction, 0.0, 1.0) * (len(lut) - 1)).astype(np.intp)
    rgba = np.empty((len(indices), 4), dtype=np.uint8)
    rgba[:, :3] = _linear(lut[indices])
    rgba[:, 3] = 255
    return rgba


def _pad(data: bytes, filler: bytes = b"\x00") -> bytes:
    return data + filler * (-len(data) % 4)


def glb_bytes(shape: ViewShape, z_factor: float = 1.0) -> bytes:
    """Return the GLB file of the mesh of ``shape``: its triangles with normals, or its points.

    The vertices have the colors of the color table over the heights, and the heights the
    exaggeration ``z_factor``, as the 3D view shows them. The size is scaled so that the longest
    lateral side is 1; the real size is stored in the ``extras`` of the asset.
    """
    model, mesh, lut = shape
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

    vertices = np.ascontiguousarray(_upright(mesh.vertices[used], z_factor))
    arrays: list[tuple[bytes, int, str, int]] = [
        (vertices.tobytes(), _ARRAY_BUFFER, "VEC3", _FLOAT)
    ]
    attributes = {"POSITION": 0}
    counts = [len(vertices)]
    if len(triangles):
        normals = _normals(vertices, triangles)
        arrays.append((normals.tobytes(), _ARRAY_BUFFER, "VEC3", _FLOAT))
        attributes["NORMAL"] = len(arrays) - 1
        counts.append(len(vertices))
    arrays.append(
        (_colors(mesh, model, lut, used).tobytes(), _ARRAY_BUFFER, "VEC4", _UNSIGNED_BYTE)
    )
    attributes["COLOR_0"] = len(arrays) - 1
    counts.append(len(vertices))
    if len(triangles):
        arrays.append((triangles.tobytes(), _ELEMENT_ARRAY_BUFFER, "SCALAR", _UNSIGNED_INT))
        counts.append(triangles.size)
    index = len(arrays) - 1

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
        if component == _UNSIGNED_BYTE:
            accessors[-1]["normalized"] = True
        binary += _pad(data)
    accessors[0]["min"] = [float(value) for value in vertices.min(axis=0)]
    accessors[0]["max"] = [float(value) for value in vertices.max(axis=0)]

    primitive: dict[str, object] = {
        "attributes": attributes,
        "material": 0,
        "mode": _TRIANGLES if len(triangles) else _POINTS,
    }
    if len(triangles):
        primitive["indices"] = index
    document = {
        "asset": {
            "version": "2.0",
            "generator": f"sdflens {__version__}",
            "extras": {
                "unit": "the longest lateral side is 1",
                "extent_meters": [model.extent_x, model.extent_y],
                "scale_meters": model.scale,
                "z_factor": z_factor,
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
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
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


def write_glb(path: str | Path, shape: ViewShape, z_factor: float = 1.0) -> None:
    """Write the GLB file of ``shape`` to ``path``; see :func:`glb_bytes`."""
    Path(path).write_bytes(glb_bytes(shape, z_factor))
