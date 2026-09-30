# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sdfio
from sdfio import DataType, SdfDialect, SdfFormatError

from helpers import make_ramp, make_sdf
from sdflens.convert import allowed_data_types, convert_for_export, format_problem


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
