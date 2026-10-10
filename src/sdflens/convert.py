# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Conversion of a loaded SDF or x3p file to another version, format and data type."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

import numpy as np
import x3pio
from sdfio import (
    DataType,
    FileFormat,
    SdfDialect,
    SdfFile,
    SdfFormatError,
    SdfHeader,
    format_tagged_fields,
    get_data_type,
    suggest_z_scale,
    validate_trailer_tagged,
)

from .surfacefile import SurfaceFile, is_grid

__all__ = [
    "ExportProblem",
    "ProblemKind",
    "allowed_data_types",
    "convert_file",
    "convert_for_export",
    "convert_x3p_for_export",
    "extensions_fit",
    "format_problem",
    "not_a_grid",
    "sdf_to_x3p",
    "x3p_data_types",
    "x3p_to_sdf",
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


#: The longest manufacturer ID of an SDF file.
_MANUFACTURER_ID_LENGTH = 10


def not_a_grid(file: SurfaceFile) -> bool:
    """Whether ``file`` cannot be saved as an SDF file, which holds a grid of heights only."""
    return not is_grid(file)


def x3p_data_types() -> list[x3pio.DataType]:
    """Return the storage types of the heights of an x3p file."""
    return list(x3pio.DataType)


def _ascii(text: str, limit: int | None = None) -> str:
    """Return ``text`` as 7-bit ASCII on one line, cut to ``limit`` characters."""
    text = " ".join(text.split()).encode("ascii", errors="replace").decode("ascii")
    return text[:limit] if limit is not None else text


def sdf_to_x3p(sdf: SdfFile) -> x3pio.X3pFile:
    """Return the x3p file that holds the surface of ``sdf``, as float64 in the newest dialect.

    The trailer becomes the comment, the manufacturer ID the manufacturer of the instrument and
    the creation date the date.
    """
    header = sdf.header
    trailer = sdf.trailer
    comment = trailer if isinstance(trailer, str) else trailer.decode("ascii", errors="replace")
    metadata = x3pio.Metadata(
        date=header.create_date or datetime.now(UTC),
        instrument=x3pio.Instrument(manufacturer=header.manufacturer_id),
        comment=comment.strip() or None,
    )
    data = np.asarray(sdf.data, dtype=np.float64)
    if header.num_profiles == 1:
        return x3pio.Profile.from_array(data[0], x_scale=float(header.x_scale), metadata=metadata)
    return x3pio.Surface.from_array(
        data, x_scale=float(header.x_scale), y_scale=float(header.y_scale), metadata=metadata
    )


def x3p_to_sdf(x3p: x3pio.X3pFile, layer: int = 0) -> SdfFile:
    """Return the SDF file that holds the surface of ``layer`` of ``x3p``, as binary64 in ISO-2.0.

    Only a grid of heights can be saved, and an SDF file holds one: of several layers the one
    asked for, counted from 0. The metadata becomes the
    tagged trailer, the manufacturer of the instrument the manufacturer ID and the date the
    creation date. Offsets, the rotation and vendor extensions are lost.

    :raises SdfFormatError: If ``x3p`` is a point cloud or has x or y coordinates for each point.
    """
    if not is_grid(x3p):
        raise SdfFormatError("Only a grid of heights can be saved as an SDF file")
    chosen = x3p.layers[layer]
    data = np.array(np.atleast_2d(chosen.z), dtype=np.float64)
    metadata = x3p.metadata
    fields: dict[str, str] = {}
    create_date = None
    manufacturer = ""
    if metadata is not None:
        instrument = metadata.instrument
        manufacturer = instrument.manufacturer
        probing = metadata.probing_system
        if metadata.date is not None:
            create_date = metadata.date.astimezone(UTC)
        calibration = metadata.calibration_date
        fields = {
            "Creator": metadata.creator or "",
            "Manufacturer": instrument.manufacturer,
            "Model": instrument.model,
            "Serial": instrument.serial,
            "Version": instrument.version,
            "CalibrationDate": calibration.isoformat() if calibration else "",
            "ProbingType": probing.type.value if probing.type else "",
            "ProbingIdentification": probing.identification,
            "Comment": metadata.comment or "",
        }
    header = SdfHeader(
        manufacturer_id=_ascii(manufacturer, _MANUFACTURER_ID_LENGTH) or "sdflens",
        create_date=create_date,
        mod_date=None,
        num_points=data.shape[1],
        num_profiles=data.shape[0],
        x_scale=abs(float(chosen.header.x.increment)),
        y_scale=abs(float(chosen.header.y.increment)),
    )
    trailer = format_tagged_fields({name: _ascii(value) for name, value in fields.items() if value})
    return SdfFile(header=header, data=data, trailer=trailer)


def extensions_fit(x3p: x3pio.X3pFile, revision: x3pio.Revision) -> bool:
    """Whether the vendor extensions of ``x3p`` can be kept in ``revision``."""
    try:
        x3p.with_revision(revision)
    except x3pio.X3pFormatError:
        return False
    return True


def convert_x3p_for_export(
    x3p: x3pio.X3pFile, revision: x3pio.Revision, data_type: x3pio.DataType
) -> x3pio.X3pFile:
    """Return ``x3p`` retargeted to ``revision`` and the storage type of the heights.

    The vendor extensions are left out if ``revision`` cannot hold them, see
    :func:`extensions_fit`. The scale of the heights is the finest that fits ``data_type``. The
    input is not modified.

    :raises X3pFormatError: If the data cannot be stored in ``data_type``.
    """
    converted = x3p.with_revision(revision, drop_extensions=True)
    if converted.header.z.data_type is not data_type:
        converted = converted.with_z_type(data_type)
    return converted


def convert_file(
    file: SurfaceFile,
    dialect: SdfDialect | x3pio.Revision,
    data_type: DataType | x3pio.DataType,
    layer: int = 0,
) -> SdfFile | x3pio.X3pFile:
    """Return ``file`` converted to the format of ``dialect``, ready to save.

    An SDF file holds one grid, so of several layers the one asked for, counted from 0, is saved
    as SDF. An x3p file keeps all its layers.

    :raises SdfFormatError: If an SDF file cannot be made of the data, or encode it.
    :raises X3pFormatError: If an x3p file cannot encode the data.
    :raises TypeError: If ``data_type`` is not one of the format of ``dialect``.
    """
    if isinstance(dialect, SdfDialect) and isinstance(data_type, DataType):
        sdf = file if isinstance(file, SdfFile) else x3p_to_sdf(file, layer)
        return convert_for_export(sdf, dialect, data_type)
    if isinstance(dialect, x3pio.Revision) and isinstance(data_type, x3pio.DataType):
        x3p = sdf_to_x3p(file) if isinstance(file, SdfFile) else file
        return convert_x3p_for_export(x3p, dialect, data_type)
    raise TypeError("The data type does not belong to the format of the dialect")
