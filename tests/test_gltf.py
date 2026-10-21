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
from sdflens.colormap import build_lut
from sdflens.gltf import ViewShape, _linear, glb_bytes, write_glb
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


def position(document: dict[str, Any], binary: bytes) -> NDArray[Any]:
    primitive = document["meshes"][0]["primitives"][0]
    return accessor(document, binary, primitive["attributes"]["POSITION"])


def accessor(document: dict[str, Any], binary: bytes, index: int) -> NDArray[Any]:
    entry = document["accessors"][index]
    view = document["bufferViews"][entry["bufferView"]]
    dtype = {5126: np.float32, 5121: np.uint8, 5125: np.uint32}[entry["componentType"]]
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[entry["type"]]
    array = np.frombuffer(
        binary, dtype=dtype, count=entry["count"] * width, offset=view["byteOffset"]
    )
    return array.reshape(-1, width) if width > 1 else array


LUT = build_lut("Viridis")


def shape_of(model: SurfaceModel | PointCloudModel) -> ViewShape:
    mesh = (
        SurfaceMesh.from_point_cloud(model)
        if isinstance(model, PointCloudModel)
        else SurfaceMesh.from_model(model)
    )
    return ViewShape(model, mesh, LUT)


def surface() -> tuple[SurfaceModel, ViewShape]:
    model = SurfaceModel.from_sdf(make_sdf(make_ramp()))
    return model, shape_of(model)


def test_a_surface_becomes_triangles_with_normals() -> None:
    _model, shape = surface()
    document, binary = parse(glb_bytes(shape))
    primitive = document["meshes"][0]["primitives"][0]

    assert primitive["mode"] == 4
    positions = accessor(document, binary, primitive["attributes"]["POSITION"])
    normals = accessor(document, binary, primitive["attributes"]["NORMAL"])
    indices = accessor(document, binary, primitive["indices"])
    # The point that is not measured has no vertex, and every index refers to one that is written.
    assert len(positions) == ROWS * COLS - 1
    assert len(normals) == len(positions)
    assert "COLOR_0" in primitive["attributes"]
    assert indices.max() < len(positions)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    box = document["accessors"][primitive["attributes"]["POSITION"]]
    assert np.allclose(box["min"], positions.min(axis=0))
    assert np.allclose(box["max"], positions.max(axis=0))


def test_the_surface_stands_up_at_true_scale() -> None:
    model, shape = surface()
    points = position(*parse(glb_bytes(shape)))

    assert points[:, 0].max() - points[:, 0].min() == pytest.approx(
        model.extent_x / model.scale, rel=1e-5
    )
    assert points[:, 2].max() - points[:, 2].min() == pytest.approx(
        model.extent_y / model.scale, rel=1e-5
    )
    assert points[:, 1].max() - points[:, 1].min() == pytest.approx(
        model.height_fraction, rel=1e-5
    )  # y is the height


def test_a_flat_surface_has_normals_that_point_up() -> None:
    flat = SurfaceModel.from_sdf(make_sdf(np.ones((3, 4)) * 1e-9))
    document, binary = parse(glb_bytes(shape_of(flat)))
    normals = accessor(
        document, binary, document["meshes"][0]["primitives"][0]["attributes"]["NORMAL"]
    )

    assert np.allclose(normals, [0.0, 1.0, 0.0], atol=1e-6)


def test_a_point_cloud_becomes_points() -> None:
    model = PointCloudModel.from_points(make_cloud())
    document, binary = parse(glb_bytes(shape_of(model)))
    primitive = document["meshes"][0]["primitives"][0]

    assert primitive["mode"] == 0
    assert "indices" not in primitive
    assert "NORMAL" not in primitive["attributes"]
    assert len(position(document, binary)) == model.num_points


def test_the_real_size_is_kept_in_the_asset() -> None:
    model, shape = surface()
    extras = parse(glb_bytes(shape))[0]["asset"]["extras"]

    assert extras["extent_meters"] == [model.extent_x, model.extent_y]
    assert extras["scale_meters"] == model.scale


def test_nothing_is_written_without_measured_points(tmp_path: Path) -> None:
    model = SurfaceModel.from_sdf(make_sdf(make_ramp()))
    mesh = SurfaceMesh.from_model(model)
    empty = SurfaceMesh(
        vertices=mesh.vertices,
        triangles=mesh.triangles[:0],
        lines=mesh.lines,
        valid=np.zeros_like(mesh.valid),
        step=1,
    )
    with pytest.raises(ValueError, match="measured"):
        glb_bytes(ViewShape(model, empty, LUT))


def test_write_glb_saves_the_file(tmp_path: Path) -> None:
    _model, shape = surface()
    target = tmp_path / "surface.glb"
    write_glb(target, shape)

    assert target.read_bytes() == glb_bytes(shape)


def test_the_height_is_exaggerated_on_request() -> None:
    _model, shape = surface()
    true = position(*parse(glb_bytes(shape)))
    tall = position(*parse(glb_bytes(shape, 4.0)))

    assert np.allclose(tall[:, 1], 4.0 * true[:, 1])
    assert np.allclose(tall[:, [0, 2]], true[:, [0, 2]])


def test_colors_follow_the_height_through_the_color_table() -> None:
    _model, shape = surface()
    document, binary = parse(glb_bytes(shape))
    primitive = document["meshes"][0]["primitives"][0]
    colors = accessor(document, binary, primitive["attributes"]["COLOR_0"])
    heights = position(document, binary)[:, 1]
    entry = document["accessors"][primitive["attributes"]["COLOR_0"]]

    assert entry["normalized"] is True
    assert colors[heights.argmin(), :3].tolist() == _linear(LUT[:1])[0].tolist()
    assert colors[heights.argmax(), :3].tolist() == _linear(LUT[-1:])[0].tolist()


def test_colors_are_written_linear_as_gltf_asks() -> None:
    srgb = np.array([[0, 128, 255]], dtype=np.uint8)

    assert _linear(srgb)[0].tolist() == [0, 55, 255]


def test_a_colored_point_cloud_has_colors_too() -> None:
    document, binary = parse(glb_bytes(shape_of(PointCloudModel.from_points(make_cloud()))))
    attributes = document["meshes"][0]["primitives"][0]["attributes"]

    assert len(accessor(document, binary, attributes["COLOR_0"])) == len(position(document, binary))
