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
from sdflens.infopanel import extension_rows, format_metadata, metadata_rows
from sdflens.mesh import SurfaceMesh
from sdflens.pointcloud import PointCloudModel
from sdflens.surface import NoMeasuredPointsError, SurfaceModel
from sdflens.surfacefile import build_mesh, build_model, read_surface_file


def test_x3p_surface_is_a_grid_model() -> None:
    model = build_model(make_x3p())

    assert isinstance(model, SurfaceModel)
    assert model.data.shape == make_ramp().shape
    assert (model.x_scale, model.y_scale) == (1e-6, 2e-6)
    assert model.invalid_count == 1


def test_x3p_profile_is_a_profile_model() -> None:
    profile = make_ramp()[:1, :]
    x3p = x3pio.X3pFile.from_array(profile, x_scale=1e-6, y_scale=1e-6)
    x3p.header.feature_type = x3pio.FeatureType.PROFILE
    model = build_model(x3p)

    assert isinstance(model, SurfaceModel)
    assert model.is_profile


def test_first_layer_of_a_layered_x3p_is_shown() -> None:
    data = np.stack([make_ramp(), make_ramp() + 1.0])
    model = build_model(x3pio.X3pFile.from_array(data, x_scale=1e-6, y_scale=1e-6))

    assert isinstance(model, SurfaceModel)
    assert model.data.shape == make_ramp().shape
    assert np.nanmin(model.data) == pytest.approx(0.0)


def test_x3p_point_cloud_is_a_point_cloud_model() -> None:
    model = build_model(x3pio.X3pFile.from_points(make_cloud()))

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
    x3p = make_x3p()
    x3p.header.x.axis_type = x3pio.AxisType.ABSOLUTE
    x3p.x = np.broadcast_to(np.arange(5) * 1e-6, x3p.data.shape).copy()
    model = build_model(x3p)

    assert isinstance(model, PointCloudModel)
    assert model.num_points == make_ramp().size - 1


def test_file_without_measured_points_is_rejected() -> None:
    nothing = np.full((3, 3), np.nan)

    with pytest.raises(NoMeasuredPointsError):
        build_model(x3pio.X3pFile.from_array(nothing, x_scale=1e-6, y_scale=1e-6))
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
    rows = dict(metadata_rows(x3pio.X3pFile.from_points(make_cloud())))

    assert rows["FeatureType"] == "PCL"
    assert rows["ListDimension"] == "200"
    assert "MatrixDimension.SizeX" not in rows


def test_rotation_and_vendor_id_of_the_old_dialect_are_shown() -> None:
    x3p = make_x3p().with_dialect(x3pio.X3pDialect.ISO5436_2000, drop_extensions=True)
    x3p.header.rotation = np.eye(3)
    assert isinstance(x3p.extensions, x3pio.VendorFiles)
    x3p.extensions.vendor_id = "http://www.vendor.com/format"
    rows = dict(metadata_rows(x3p))

    assert rows["Revision"] == "ISO5436 - 2000"
    assert rows["Rotation"] == "1 0 0; 0 1 0; 0 0 1"
    assert rows["VendorSpecificID"] == "http://www.vendor.com/format"


def test_vendor_extensions_are_listed_with_their_size() -> None:
    assert extension_rows(make_x3p()) == [
        ("http://www.vendor.com/mypath/a.xml", "4 bytes"),
        ("http://www.vendor.com/image.png", "1500 bytes"),
    ]
    assert extension_rows(make_sdf(make_ramp())) == []


def test_exported_information_lists_the_vendor_extensions() -> None:
    text = format_metadata(make_x3p())
    lines = text.splitlines()

    assert any(line.startswith("Creator") for line in lines)
    assert lines[-3].startswith("VendorExtensions")
    assert lines[-2:] == [
        "http://www.vendor.com/mypath/a.xml (4 bytes)",
        "http://www.vendor.com/image.png (1500 bytes)",
    ]
    assert "VendorExtensions" not in format_metadata(x3pio.X3pFile.from_points(make_cloud()))
