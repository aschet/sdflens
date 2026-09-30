# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Conversion of a loaded SDF file to another version, format and data type."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sdfio import (
    DataType,
    FileFormat,
    SdfDialect,
    SdfFile,
    SdfFormatError,
    get_data_type,
    suggest_z_scale,
    validate_trailer_tagged,
)

__all__ = [
    "ExportProblem",
    "ProblemKind",
    "allowed_data_types",
    "convert_for_export",
    "format_problem",
]

# Largest NumPoints/NumProfiles of the binary format: uint16 for 1.0, uint32 for 2.0 (Table 2).
_MAX_BINARY_COUNT = {
    SdfDialect.ISO_1_0: 2**16 - 1,
    SdfDialect.BCR_1_0: 2**16 - 1,
    SdfDialect.ISO_2_0: 2**32 - 1,
}


class ProblemKind(StrEnum):
    """Why a file cannot be saved in a version and format."""

    BINARY_LIMIT = "binary limit"
    UNTAGGED_TRAILER = "untagged trailer"


@dataclass(frozen=True)
class ExportProblem:
    """A reason a file cannot be saved in a version and format; ``limit`` is the point count."""

    kind: ProblemKind
    limit: int = 0


def allowed_data_types(dialect: SdfDialect) -> list[DataType]:
    """Return the data types the standard allows for ``dialect``.

    >>> [t.name for t in allowed_data_types(SdfDialect.ISO_1_0)]
    ['INT16', 'INT32', 'BINARY64']
    """
    return [data_type for data_type in DataType if get_data_type(data_type).is_supported(dialect)]


def format_problem(
    sdf: SdfFile, dialect: SdfDialect, file_format: FileFormat
) -> ExportProblem | None:
    """Return why ``sdf`` can't be saved as ``dialect`` in ``file_format``, or ``None``.

    Data types are checked separately, see :func:`allowed_data_types`.
    """
    limit = _MAX_BINARY_COUNT[dialect]
    if file_format is FileFormat.BINARY and max(sdf.header.shape) > limit:
        return ExportProblem(ProblemKind.BINARY_LIMIT, limit)
    try:
        validate_trailer_tagged(dialect, sdf.trailer)
    except SdfFormatError:
        return ExportProblem(ProblemKind.UNTAGGED_TRAILER)
    return None


def convert_for_export(sdf: SdfFile, dialect: SdfDialect, data_type: DataType) -> SdfFile:
    """Return ``sdf`` retargeted to ``dialect`` and ``data_type``, ready to save.

    The Z-scale is kept when the data fits the new type, otherwise the finest scale that fits is
    used. The input is not modified.

    :raises SdfFormatError: If the trailer is invalid for ``dialect`` or the data can't be encoded.
    """
    converted = sdf.with_dialect(dialect, data_type=data_type)
    try:
        converted.raw_data()
    except SdfFormatError:
        converted.header.z_scale = suggest_z_scale(
            converted.data, get_data_type(data_type), dialect
        )
    return converted
