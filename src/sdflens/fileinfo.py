# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What the information panel says about a file: labelled values, the same for SDF and x3p."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import sdfio
import x3pio

from .surfacefile import SurfaceFile, profile_of
from .units import format_length

__all__ = ["InfoSection", "describe_file"]

#: How the data types of an SDF file are named; x3p names them like this already.
_SDF_TYPES = {
    sdfio.DataType.UINT8: "uint8",
    sdfio.DataType.UINT16: "uint16",
    sdfio.DataType.UINT32: "uint32",
    sdfio.DataType.BINARY32: "float32",
    sdfio.DataType.INT8: "int8",
    sdfio.DataType.INT16: "int16",
    sdfio.DataType.INT32: "int32",
    sdfio.DataType.BINARY64: "float64",
}

#: The multiplication sign between the sizes of a surface.
_TIMES = "\u00d7"

_PLACEHOLDER = re.compile(r"(n/a|not available|not set|none|unknown|-)?", re.IGNORECASE)


@dataclass(frozen=True)
class InfoSection:
    """A group of labelled values, or a text of its own such as the comment of a file."""

    title: str
    rows: tuple[tuple[str, str], ...] = ()
    text: str = ""


def _given(value: str | None) -> str:
    """Return ``value``, or a dash if it is empty or stands for a value that is not given."""
    text = (value or "").strip()
    return "-" if _PLACEHOLDER.fullmatch(text) else text


def _iso_date(value: datetime) -> str:
    local = value.astimezone() if value.tzinfo is not None else value
    return local.strftime("%Y-%m-%d %H:%M")


def _date(value: datetime | None, format_date: Callable[[datetime], str]) -> str:
    return format_date(value) if value is not None else "-"


def _kinds(*incremental: bool) -> str:
    return ", ".join("inc" if flag else "abs" for flag in incremental)


def _dimension(is_surface: bool, columns: int, rows: int) -> str:
    return (
        f"{columns:,} {_TIMES} {rows:,} points" if is_surface else f"{max(columns, rows):,} points"
    )


def _describe_sdf(sdf: sdfio.SdfFile, format_date: Callable[[datetime], str]) -> list[InfoSection]:
    header = sdf.header
    rows, columns = sdf.data.shape
    surface = not sdf.is_profile
    step = format_length(header.x_scale)
    if surface:
        step = f"{step} {_TIMES} {format_length(header.y_scale)}"
    data = [
        ("Axis", _kinds(True, True, False)),
        ("Type", _SDF_TYPES[header.data_type]),
        ("Dimension", _dimension(surface, columns, rows)),
        ("Step", step),
    ]
    if header.z_resolution > 0:
        data.append(("Height resolution", format_length(header.z_resolution)))
    file = (
        ("Format", f"SDF {'binary' if header.format is sdfio.FileFormat.BINARY else 'ASCII'}"),
        ("Revision", header.revision.value),
        ("Feature", "Surface" if surface else "Profile"),
        ("Creator", "-"),
        ("Date", _date(header.create_date, format_date)),
    )
    return [
        InfoSection("File", file),
        InfoSection("Data", tuple(data)),
        InfoSection("Instrument", _instrument(_given(header.manufacturer_id))),
        InfoSection("Probing", (("Type", "-"), ("Identification", "-"))),
    ]


def _instrument(
    manufacturer: str = "-",
    model: str = "-",
    serial: str = "-",
    version: str = "-",
    calibration: str = "-",
) -> tuple[tuple[str, str], ...]:
    return (
        ("Manufacturer", manufacturer),
        ("Model", model),
        ("Serial", serial),
        ("Version", version),
        ("Calibration", calibration),
    )


def _x3p_step(x3p: x3pio.X3pFile) -> str:
    """Return the spacing of the points of an x3p file, or ``irregular``."""
    layer = x3p.layers[0]
    header = x3p.header
    is_profile = isinstance(x3p, x3pio.Profile)
    if isinstance(x3p, x3pio.PointCloud):
        return "irregular"
    if layer.is_regular_grid:
        step = format_length(abs(header.x.increment))
        return step if is_profile else f"{step} {_TIMES} {format_length(abs(header.y.increment))}"
    spacing = profile_of(layer)
    if spacing is None or spacing.step is None:
        return "irregular"
    step = format_length(spacing.step)
    return f"{step}, assumed" if spacing.assumed else step


def _describe_x3p(x3p: x3pio.X3pFile, format_date: Callable[[datetime], str]) -> list[InfoSection]:
    header = x3p.header
    feature = {
        x3pio.FeatureType.PROFILE: "Profile",
        x3pio.FeatureType.SURFACE: "Surface",
        x3pio.FeatureType.POINT_CLOUD: "Point cloud",
    }[x3p.feature_type]
    shape = x3p.layers[0].shape
    columns, rows = (shape[-1], shape[0]) if len(shape) > 1 else (shape[0], 1)
    metadata = x3p.metadata
    instrument = metadata.instrument if metadata is not None else None
    probing = metadata.probing_system if metadata is not None else None
    instrument_rows = _instrument()
    if metadata is not None and instrument is not None:
        instrument_rows = _instrument(
            _given(instrument.manufacturer),
            _given(instrument.model),
            _given(instrument.serial),
            _given(instrument.version),
            _date(metadata.calibration_date, format_date),
        )
    file = (
        ("Format", f"x3p {'binary' if x3p.storage is x3pio.DataStorage.BINARY else 'XML'}"),
        ("Revision", x3p.revision.value),
        ("Feature", feature),
        ("Creator", _given(metadata.creator) if metadata is not None else "-"),
        ("Date", _date(metadata.date if metadata is not None else None, format_date)),
    )
    types = ", ".join(axis.data_type.name.lower() for axis in (header.x, header.y, header.z))
    data = [
        ("Axis", _kinds(*(axis.is_incremental for axis in (header.x, header.y, header.z)))),
        ("Type", types),
        ("Dimension", _dimension(isinstance(x3p, x3pio.Surface), columns, rows)),
        ("Step", _x3p_step(x3p)),
    ]
    placement = x3p.placement
    if placement.has_rotation or any(placement.offset):
        data.append(("Placement", "stored in the file, not applied"))
    sections = [
        InfoSection("File", file),
        InfoSection("Data", tuple(data)),
        InfoSection("Instrument", instrument_rows),
        InfoSection(
            "Probing",
            (
                ("Type", probing.type.value if probing is not None and probing.type else "-"),
                ("Identification", _given(probing.identification) if probing else "-"),
            ),
        ),
    ]
    comment = _given(metadata.comment) if metadata is not None else "-"
    if comment != "-":
        sections.append(InfoSection("Comment", text=comment))
    return sections


def describe_file(
    file: SurfaceFile, format_date: Callable[[datetime], str] = _iso_date
) -> list[InfoSection]:
    """Return the sections that describe ``file``, in the order they are shown.

    These are File, Data, Instrument and Probing, and Comment for an x3p file that has one.
    The SDF trailer and the vendor extensions of an x3p file are not part of them: the panel shows
    them as they are. ``format_date`` writes a date, by default as ``2026-01-31 13:00`` in local
    time.
    """
    if isinstance(file, sdfio.SdfFile):
        return _describe_sdf(file, format_date)
    return _describe_x3p(file, format_date)
