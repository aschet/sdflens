# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from helpers import COLS, ROWS, make_cloud, make_ramp, make_sdf
from sdflens.gltf import glb_bytes, write_glb
from sdflens.mesh import SurfaceMesh
from sdflens.pointcloud import PointCloudModel
from sdflens.surface import SurfaceModel


def parse(data: bytes) -> tuple[dict[str, Any], bytes]:
    magic, version, length = struct.unpack_from("<III", data)
    assert (magic, version, length) == (0x46546C67, 2, len(data))
    text_length, text_type = struct.unpack_from("<II", data, 12)
    assert text_type == 0x4E4F534A
    document = json.loads(data[20 : 20 + text_length])
    binary_length, binary_type = struct.unpack_from("<II", data, 20 + text_length)
    assert binary_type == 0x004E4942
    start = 28 + text_length
    return document, data[start : start + binary_length]


def accessor(document: dict[str, Any], binary: bytes, index: int) -> NDArray[Any]:
    entry = document["accessors"][index]
    view = document["bufferViews"][entry["bufferView"]]
    dtype = {5126: np.float32, 5125: np.uint32}[entry["componentType"]]
    width = {"SCALAR": 1, "VEC3": 3}[entry["type"]]
    array = np.frombuffer(
        binary, dtype=dtype, count=entry["count"] * width, offset=view["byteOffset"]
    )
    return array.reshape(-1, width) if width > 1 else array


def position(document: dict[str, Any], binary: bytes) -> NDArray[Any]:
    primitive = document["meshes"][0]["primitives"][0]
    return accessor(document, binary, primitive["attributes"]["POSITION"])


def surface(data: NDArray[np.float64] | None = None) -> tuple[SurfaceModel, SurfaceMesh]:
    model = SurfaceModel.from_sdf(make_sdf(make_ramp() if data is None else data))
    return model, SurfaceMesh.from_model(model)


def test_a_surface_becomes_triangles_with_normals() -> None:
    model, mesh = surface()
    document, binary = parse(glb_bytes(mesh, model))
    primitive = document["meshes"][0]["primitives"][0]

    assert primitive["mode"] == 4
    positions = accessor(document, binary, primitive["attributes"]["POSITION"])
    normals = accessor(document, binary, primitive["attributes"]["NORMAL"])
    indices = accessor(document, binary, primitive["indices"])
    # The point that is not measured has no vertex, and every index refers to one that is written.
    assert len(positions) == ROWS * COLS - 1
    assert len(normals) == len(positions)
    assert indices.max() < len(positions)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    box = document["accessors"][primitive["attributes"]["POSITION"]]
    assert np.allclose(box["min"], positions.min(axis=0))
    assert np.allclose(box["max"], positions.max(axis=0))


def test_there_are_no_colors_and_the_material_is_not_metallic() -> None:
    model, mesh = surface()
    document, _binary = parse(glb_bytes(mesh, model))
    primitive = document["meshes"][0]["primitives"][0]

    assert "COLOR_0" not in primitive["attributes"]
    material = document["materials"][0]["pbrMetallicRoughness"]
    assert material["metallicFactor"] == 0.0


def test_the_surface_stands_up_in_meters() -> None:
    model, mesh = surface()
    points = position(*parse(glb_bytes(mesh, model)))
    height = model.value_range.hi - model.value_range.lo

    # Distances in the file are the real ones, so that they can be measured.
    assert points[:, 0].max() - points[:, 0].min() == pytest.approx(model.extent_x, rel=1e-5)
    assert points[:, 2].max() - points[:, 2].min() == pytest.approx(model.extent_y, rel=1e-5)
    assert points[:, 1].max() - points[:, 1].min() == pytest.approx(height, rel=1e-5)  # y is up


def test_a_flat_surface_has_normals_that_point_up() -> None:
    model, mesh = surface(np.ones((3, 4)) * 1e-9)
    document, binary = parse(glb_bytes(mesh, model))
    normals = accessor(
        document, binary, document["meshes"][0]["primitives"][0]["attributes"]["NORMAL"]
    )

    assert np.allclose(normals, [0.0, 1.0, 0.0], atol=1e-6)


def test_a_point_cloud_becomes_points() -> None:
    model = PointCloudModel.from_points(make_cloud())
    mesh = SurfaceMesh.from_point_cloud(model)
    document, binary = parse(glb_bytes(mesh, model))
    primitive = document["meshes"][0]["primitives"][0]

    assert primitive["mode"] == 0
    assert "indices" not in primitive
    assert "NORMAL" not in primitive["attributes"]
    assert len(position(document, binary)) == model.num_points


def test_the_unit_and_the_origin_of_the_data_are_kept_in_the_asset() -> None:
    model, mesh = surface()
    extras = parse(glb_bytes(mesh, model))[0]["asset"]["extras"]

    assert extras["unit"] == "meters"
    assert extras["extent_meters"] == [model.extent_x, model.extent_y]
    assert extras["origin_meters"] == list(mesh.origin)


def test_nothing_is_written_without_measured_points() -> None:
    model, mesh = surface()
    empty = SurfaceMesh(
        vertices=mesh.vertices,
        triangles=mesh.triangles[:0],
        lines=mesh.lines,
        valid=np.zeros_like(mesh.valid),
        step=1,
    )
    with pytest.raises(ValueError, match="measured"):
        glb_bytes(empty, model)


def test_write_glb_saves_the_file(tmp_path: Path) -> None:
    model, mesh = surface()
    target = tmp_path / "surface.glb"
    write_glb(target, mesh, model)

    assert target.read_bytes() == glb_bytes(mesh, model)
