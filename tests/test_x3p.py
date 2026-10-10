# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sdfio
import x3pio

from helpers import make_cloud, make_ramp, make_sdf, make_x3p
from sdflens.glwidget import RenderMode, _primitives
from sdflens.infopanel import extension_names, format_metadata, metadata_rows
from sdflens.mesh import SurfaceMesh
from sdflens.pointcloud import PointCloudModel
from sdflens.surface import NoMeasuredPointsError, SurfaceModel
from sdflens.surfacefile import build_mesh, build_model, layer_count, read_surface_file


def test_x3p_surface_is_a_grid_model() -> None:
    model = build_model(make_x3p())

    assert isinstance(model, SurfaceModel)
    assert model.data.shape == make_ramp().shape
    assert (model.x_scale, model.y_scale) == (1e-6, 2e-6)
    assert model.invalid_count == 1


def test_x3p_profile_is_a_profile_model() -> None:
    x3p = x3pio.Profile.from_array(make_ramp()[0], x_scale=1e-6)
    model = build_model(x3p)

    assert isinstance(model, SurfaceModel)
    assert model.is_profile


def test_the_layer_that_is_asked_for_is_shown() -> None:
    data = np.stack([make_ramp(), make_ramp() + 1.0])
    x3p = x3pio.Surface.from_array(data, x_scale=1e-6, y_scale=1e-6)
    first, second = build_model(x3p), build_model(x3p, 1)

    assert layer_count(x3p) == 2
    assert isinstance(first, SurfaceModel)
    assert isinstance(second, SurfaceModel)
    assert first.data.shape == second.data.shape == make_ramp().shape
    assert np.nanmin(first.data) == pytest.approx(0.0)  # the first layer is the default
    assert np.nanmin(second.data) == pytest.approx(1.0)
    with pytest.raises(IndexError):
        build_model(x3p, 2)


def test_every_layer_has_its_own_value_range() -> None:
    data = np.stack([make_ramp(), 10.0 * make_ramp()])
    x3p = x3pio.Surface.from_array(data, x_scale=1e-6, y_scale=1e-6)

    assert build_model(x3p, 1).value_range.hi == pytest.approx(
        10.0 * build_model(x3p).value_range.hi
    )


def test_a_file_without_layers_of_its_own_has_one() -> None:
    assert layer_count(make_sdf(make_ramp())) == 1
    assert layer_count(make_x3p()) == 1
    assert layer_count(x3pio.PointCloud.from_points(make_cloud())) == 1


def test_x3p_point_cloud_is_a_point_cloud_model() -> None:
    model = build_model(x3pio.PointCloud.from_points(make_cloud()))

    assert isinstance(model, PointCloudModel)
    assert model.num_points == 200
    assert model.invalid_count == 0
    assert 0.0 < model.extent_x <= 1e-3
    assert model.value_range.hi > model.value_range.lo
    assert 1.0 <= model.auto_z_factor() <= 100.0


def test_point_cloud_model_drops_points_that_are_not_measured() -> None:
    points = make_cloud()
    points[3, 2] = np.nan
    points[4, 0] = np.inf

    assert PointCloudModel.from_points(points).num_points == 198


def test_matrix_with_absolute_axes_is_a_point_cloud_model() -> None:
    rows, columns = make_ramp().shape
    x, y = np.meshgrid(np.arange(columns) * 1e-6, np.arange(rows) * 2e-6)
    model = build_model(x3pio.Surface.from_points(np.stack([x, y, make_ramp()], axis=-1)))

    assert isinstance(model, PointCloudModel)
    assert model.num_points == make_ramp().size - 1

    assert isinstance(model, PointCloudModel)
    assert model.num_points == make_ramp().size - 1


def test_file_without_measured_points_is_rejected() -> None:
    nothing = np.full((3, 3), np.nan)

    with pytest.raises(NoMeasuredPointsError):
        build_model(x3pio.Surface.from_array(nothing, x_scale=1e-6, y_scale=1e-6))
    with pytest.raises(NoMeasuredPointsError):
        PointCloudModel.from_points(np.full((3, 3), np.nan))


def test_point_cloud_mesh_is_drawn_as_points() -> None:
    model = PointCloudModel.from_points(make_cloud())
    mesh = build_mesh(model)

    assert mesh.is_cloud
    assert mesh.vertices.shape == (model.num_points, 3)
    assert np.abs(mesh.vertices[:, :2]).max() <= 0.5 + 1e-6
    for mode in RenderMode:
        _, indices = _primitives(mesh, mode)
        assert len(indices) == model.num_points


def test_surface_mesh_is_not_a_cloud() -> None:
    mesh = SurfaceMesh.from_model(SurfaceModel.from_sdf(make_sdf(make_ramp())))

    assert not mesh.is_cloud


def test_read_surface_file_tells_the_formats_apart(tmp_path: Path) -> None:
    x3p_path = tmp_path / "surface.x3p"
    make_x3p().save(x3p_path)
    sdf_path = tmp_path / "ramp.dat"  # the extension does not matter
    sdfio.write(sdf_path, make_ramp(), x_scale=1e-6, y_scale=1e-6)

    assert isinstance(read_surface_file(str(x3p_path)), x3pio.X3pFile)
    assert isinstance(read_surface_file(str(sdf_path)), sdfio.SdfFile)


def test_x3p_metadata_rows_show_both_records() -> None:
    rows = dict(metadata_rows(make_x3p()))

    assert rows["Revision"] == "ISO25178-72:2017/DAM1"
    assert rows["FeatureType"] == "SUR"
    assert rows["MatrixDimension.SizeX"] == "5"
    assert rows["MatrixDimension.SizeY"] == "4"
    assert rows["CX.AxisType"] == "incremental"
    assert rows["CZ.DataType"] == "float64"
    assert rows["CX.Increment"] == "1e-06"
    assert rows["Creator"] == "Jane Doe"
    assert rows["Comment"] == "Test surface"
    assert rows["ProbingSystem.Type"] == "Software"
    assert "Rotation" not in rows
    assert "CalibrationDate" not in rows  # an optional field without a value is left out


def test_point_cloud_metadata_rows_have_a_list_dimension() -> None:
    rows = dict(metadata_rows(x3pio.PointCloud.from_points(make_cloud())))

    assert rows["FeatureType"] == "PCL"
    assert rows["ListDimension"] == "200"
    assert "MatrixDimension.SizeX" not in rows


def test_rotation_and_vendor_id_of_the_old_dialect_are_shown() -> None:
    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    x3p = make_x3p().with_revision(x3pio.Revision.ISO5436_2000, drop_extensions=True)
    x3p = x3p.with_placement(x3pio.Placement(quarter_turn, offset=[1e-3, 0.0, 0.0]))
    x3p.extensions.add("http://www.vendor.com/format", "a.bin", b"1")
    rows = dict(metadata_rows(x3p))

    assert rows["Revision"] == "ISO5436 - 2000"
    assert rows["Rotation"] == "0 -1 0; 1 0 0; 0 0 1"
    assert rows["CX.Offset"] == "0.001"
    assert rows["VendorSpecificID"] == "http://www.vendor.com/format"


def test_vendor_extensions_are_listed_by_their_id() -> None:
    assert extension_names(make_x3p()) == [
        "http://www.vendor.com/mypath/a.xml",
        "http://www.vendor.com/image.png",
    ]
    assert extension_names(make_sdf(make_ramp())) == []


def test_exported_information_lists_the_vendor_extensions() -> None:
    text = format_metadata(make_x3p())
    lines = text.splitlines()

    assert any(line.startswith("Creator") for line in lines)
    assert lines[-3].startswith("VendorExtensions")
    assert lines[-2:] == [
        "http://www.vendor.com/mypath/a.xml",
        "http://www.vendor.com/image.png",
    ]
    assert "VendorExtensions" not in format_metadata(x3pio.PointCloud.from_points(make_cloud()))
