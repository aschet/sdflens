# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sdfio
import x3pio
from sdfio import DataType, SdfDialect, SdfFile, SdfFormatError

from helpers import make_cloud, make_ramp, make_sdf, make_x3p
from sdflens.convert import (
    allowed_data_types,
    convert_file,
    convert_for_export,
    extensions_fit,
    format_problem,
    sdf_to_x3p,
    x3p_to_sdf,
)
from sdflens.infopanel import trailer_text


def test_allowed_data_types_follow_the_standard() -> None:
    assert allowed_data_types(SdfDialect.ISO_1_0) == [
        DataType.INT16,
        DataType.INT32,
        DataType.BINARY64,
    ]
    assert DataType.BINARY32 in allowed_data_types(SdfDialect.ISO_2_0)
    assert DataType.INT8 in allowed_data_types(SdfDialect.BCR_1_0)


def test_unsigned_data_types_are_only_allowed_for_bcr() -> None:
    unsigned = [DataType.UINT8, DataType.UINT16, DataType.UINT32]

    assert allowed_data_types(SdfDialect.BCR_1_0)[:3] == unsigned
    for dialect in (SdfDialect.ISO_1_0, SdfDialect.ISO_2_0):
        assert not set(unsigned) & set(allowed_data_types(dialect))


def test_conversion_keeps_the_scale_when_the_data_fits() -> None:
    sdf = make_sdf(make_ramp())
    converted = convert_for_export(sdf, SdfDialect.ISO_1_0, DataType.INT32)

    assert converted.header.dialect is SdfDialect.ISO_1_0
    assert converted.data_type is DataType.INT32
    assert converted.header.z_scale == sdf.header.z_scale
    assert sdf.header.dialect is SdfDialect.ISO_2_0  # the input stays untouched


def test_conversion_rescales_when_the_data_overflows_the_type() -> None:
    data = np.array([[0.0, 1e-6, np.nan, -1e-6]])
    sdf = make_sdf(data)
    sdf.header.z_scale = 1e-12
    converted = convert_for_export(sdf, SdfDialect.ISO_2_0, DataType.INT8)

    assert converted.header.z_scale > 1e-12
    raw = converted.raw_data()
    assert raw.dtype == np.int8


def test_incompatible_trailer_is_reported() -> None:
    sdf = make_sdf(make_ramp())
    sdf.header.dialect = SdfDialect.BCR_1_0
    sdf.trailer = "free text, not tagged"

    with pytest.raises(SdfFormatError):
        convert_for_export(sdf, SdfDialect.ISO_2_0, DataType.BINARY64)


@pytest.mark.parametrize("dialect", list(SdfDialect))
@pytest.mark.parametrize("binary", [True, False])
def test_saved_files_round_trip(dialect: SdfDialect, binary: bool, tmp_path: Path) -> None:
    path = tmp_path / "out.sdf"
    file_format = sdfio.FileFormat.BINARY if binary else sdfio.FileFormat.ASCII
    convert_for_export(make_sdf(make_ramp()), dialect, DataType.INT32).save(
        path, format=file_format
    )
    loaded = sdfio.read(path)

    assert loaded.header.dialect is dialect
    assert loaded.header.binary is binary
    assert loaded.data_type is DataType.INT32
    assert np.array_equal(np.isnan(loaded.data), np.isnan(make_ramp()))
    np.testing.assert_allclose(loaded.data, make_ramp(), atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("dialect", list(SdfDialect))
def test_format_problem_matches_what_sdfio_can_write(dialect: SdfDialect) -> None:
    limit = 2**32 - 1 if dialect is SdfDialect.ISO_2_0 else 2**16 - 1
    for count in (limit, limit + 1):
        sdf = make_sdf(np.zeros((1, 3)))
        sdf.header.num_points = count  # only the limit is checked, not the data shape
        problem = format_problem(sdf, dialect, sdfio.FileFormat.BINARY)
        assert (problem is None) == (count <= limit)
        assert format_problem(sdf, dialect, sdfio.FileFormat.ASCII) is None


def test_format_problem_reports_binary_limit_by_actual_write(tmp_path: Path) -> None:
    data = np.zeros((1, 2**16))
    for dialect, fits in ((SdfDialect.ISO_1_0, False), (SdfDialect.ISO_2_0, True)):
        sdf = make_sdf(data)
        sdf.header.dialect = dialect
        assert (format_problem(sdf, dialect, sdfio.FileFormat.BINARY) is None) is fits
        if fits:
            sdf.save(tmp_path / "ok.sdf", format=sdfio.FileFormat.BINARY)
        else:
            with pytest.raises(SdfFormatError):
                sdf.save(tmp_path / "bad.sdf", format=sdfio.FileFormat.BINARY)


@pytest.mark.parametrize("data_type", [DataType.UINT8, DataType.UINT16, DataType.UINT32])
@pytest.mark.parametrize("binary", [True, False])
def test_unsigned_bcr_files_round_trip(data_type: DataType, binary: bool, tmp_path: Path) -> None:
    path = tmp_path / "out.sdf"
    file_format = sdfio.FileFormat.BINARY if binary else sdfio.FileFormat.ASCII
    convert_for_export(make_sdf(make_ramp()), SdfDialect.BCR_1_0, data_type).save(
        path, format=file_format
    )
    loaded = sdfio.read(path)

    assert loaded.header.dialect is SdfDialect.BCR_1_0
    assert loaded.data_type is data_type
    assert np.array_equal(np.isnan(loaded.data), np.isnan(make_ramp()))
    np.testing.assert_allclose(loaded.data, make_ramp(), atol=1e-9, equal_nan=True)


def test_sdf_converts_to_x3p_with_its_scales_and_metadata() -> None:
    sdf = make_sdf(make_ramp())
    sdf.trailer = "a note"
    x3p = sdf_to_x3p(sdf)

    assert isinstance(x3p, x3pio.Surface)
    assert (x3p.header.x.increment, x3p.header.y.increment) == (1e-6, 2e-6)
    np.testing.assert_allclose(x3p.layers[0].z, make_ramp(), equal_nan=True)
    assert x3p.metadata is not None
    assert x3p.metadata.instrument.manufacturer == sdf.header.manufacturer_id
    assert x3p.metadata.comment == "a note"


def test_sdf_profile_converts_to_an_x3p_profile() -> None:
    x3p = sdf_to_x3p(make_sdf(make_ramp()[:1, :]))

    assert isinstance(x3p, x3pio.Profile)


def test_x3p_converts_to_sdf_with_its_metadata_in_the_trailer() -> None:
    x3p = make_x3p()
    x3p = x3p.with_metadata(manufacturer="Ünïcode manufacturer name")
    sdf = x3p_to_sdf(x3p)

    assert (sdf.header.x_scale, sdf.header.y_scale) == (1e-6, 2e-6)
    assert sdf.header.manufacturer_id.isascii()
    assert len(sdf.header.manufacturer_id) <= 10
    fields = sdfio.parse_tagged_fields(trailer_text(sdf))
    assert fields["Creator"] == "Jane Doe"
    assert fields["Comment"] == "Test surface"
    np.testing.assert_allclose(sdf.data, make_ramp(), equal_nan=True)


def test_only_a_grid_of_heights_converts_to_sdf() -> None:
    cloud = x3pio.PointCloud.from_points(make_cloud())

    with pytest.raises(SdfFormatError):
        x3p_to_sdf(cloud)
    with pytest.raises(SdfFormatError):
        convert_file(cloud, SdfDialect.ISO_2_0, DataType.BINARY64)


def test_the_chosen_layer_of_an_x3p_is_saved_as_sdf() -> None:
    layers = np.stack([make_ramp(), make_ramp() + 1.0])
    x3p = x3pio.Surface.from_array(layers, x_scale=1e-6, y_scale=1e-6)

    np.testing.assert_allclose(x3p_to_sdf(x3p).data, make_ramp(), equal_nan=True)  # the first
    np.testing.assert_allclose(x3p_to_sdf(x3p, 1).data, make_ramp() + 1.0, equal_nan=True)
    converted = convert_file(x3p, SdfDialect.ISO_2_0, DataType.BINARY64, 1)
    assert isinstance(converted, SdfFile)
    np.testing.assert_allclose(converted.data, make_ramp() + 1.0, equal_nan=True)


def test_an_x3p_can_be_saved_with_only_the_chosen_layer() -> None:
    layers = np.stack([make_ramp(), make_ramp() + 1.0, make_ramp() + 2.0])
    x3p = x3pio.Surface.from_array(layers, x_scale=1e-6, y_scale=2e-6).with_metadata(
        creator="Jane Doe"
    )
    x3p.extensions.add("http://www.vendor.com", "a.xml", b"<a/>")
    converted = convert_file(
        x3p, x3pio.Revision.ISO25178_72_2017_DAM1, x3pio.DataType.FLOAT64, 1, single_layer=True
    )

    assert isinstance(converted, x3pio.X3pFile)
    assert len(converted.layers) == 1
    np.testing.assert_allclose(converted.layers[0].z, make_ramp() + 1.0, equal_nan=True)
    assert converted.metadata is not None
    assert converted.metadata.creator == "Jane Doe"  # what belongs to the file is kept
    assert list(converted.extensions) == list(x3p.extensions)
    assert len(x3p.layers) == 3  # the file that is shown is not changed


def test_a_file_of_one_layer_is_saved_as_it_is_with_the_single_layer_option() -> None:
    x3p = make_x3p()
    converted = convert_file(
        x3p, x3pio.Revision.ISO5436_2000, x3pio.DataType.FLOAT64, 0, single_layer=True
    )

    assert isinstance(converted, x3pio.X3pFile)
    assert len(converted.layers) == 1


def test_an_x3p_keeps_all_its_layers_when_it_is_converted() -> None:
    layers = np.stack([make_ramp(), make_ramp() + 1.0])
    x3p = x3pio.Surface.from_array(layers, x_scale=1e-6, y_scale=1e-6)
    converted = convert_file(x3p, x3pio.Revision.ISO5436_2000, x3pio.DataType.FLOAT64, 1)

    assert isinstance(converted, x3pio.X3pFile)
    assert len(converted.layers) == 2


@pytest.mark.parametrize("storage", list(x3pio.DataStorage))
@pytest.mark.parametrize("dialect", list(x3pio.Revision))
def test_conversion_to_x3p_round_trips(
    dialect: x3pio.Revision, storage: x3pio.DataStorage, tmp_path: Path
) -> None:
    path = tmp_path / "out.x3p"
    converted = convert_file(make_sdf(make_ramp()), dialect, x3pio.DataType.INT32)
    assert isinstance(converted, x3pio.X3pFile)
    converted.save(path, storage=storage)
    loaded = x3pio.read(path)

    assert loaded.revision is dialect
    assert loaded.header.z.data_type is x3pio.DataType.INT32
    np.testing.assert_allclose(loaded.layers[0].z, make_ramp(), atol=1e-9, equal_nan=True)


def test_point_cloud_converts_between_x3p_versions_and_types(tmp_path: Path) -> None:
    cloud = x3pio.PointCloud.from_points(make_cloud())
    converted = convert_file(cloud, x3pio.Revision.ISO5436_2000, x3pio.DataType.FLOAT32)
    assert isinstance(converted, x3pio.X3pFile)
    converted.save(tmp_path / "cloud.x3p", storage=x3pio.DataStorage.XML)

    assert x3pio.read(tmp_path / "cloud.x3p").layers[0].z.shape == (200,)


def test_vendor_extensions_are_kept_when_they_fit_and_dropped_when_not() -> None:
    x3p = make_x3p()  # two IDs: the standard before the amendment has a place for one only

    assert extensions_fit(x3p, x3pio.Revision.ISO25178_72_2017_DAM1)
    assert not extensions_fit(x3p, x3pio.Revision.ISO5436_2000)
    kept = convert_file(x3p, x3pio.Revision.ISO25178_72_2017_DAM1, x3pio.DataType.FLOAT64)
    dropped = convert_file(x3p, x3pio.Revision.ISO5436_2000, x3pio.DataType.FLOAT64)
    assert isinstance(kept, x3pio.X3pFile)
    assert isinstance(dropped, x3pio.X3pFile)
    assert len(kept.extensions) == 2
    assert len(dropped.extensions) == 0


def test_a_data_type_of_the_other_format_is_rejected() -> None:
    with pytest.raises(TypeError):
        convert_file(make_sdf(make_ramp()), SdfDialect.ISO_2_0, x3pio.DataType.INT16)
