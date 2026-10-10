# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Header metadata of the loaded file: panel widget and text export."""

from __future__ import annotations

from datetime import datetime
from typing import cast

import x3pio
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QHeaderView,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from sdfio import SdfFile

from .surfacefile import SurfaceFile

__all__ = [
    "InfoPanel",
    "extension_names",
    "format_metadata",
    "metadata_rows",
    "trailer_text",
]

#: Width in characters that fits the longest values, such as the revision of an x3p file.
_MIN_COLUMNS = 48

#: A table of at most this many rows is shown completely; a longer one scrolls.
_MAX_FIXED_ROWS = 12


def metadata_rows(file: SurfaceFile) -> list[tuple[str, str]]:
    """Return the header fields as ``(name, value)`` rows, named as in the standard.

    Lengths are in meters, like in ``sdfio info``. For an x3p file these are the fields of
    Record 1 (``Revision``, the data layout, the axes and the rotation) and of Record 2.
    """
    if isinstance(file, SdfFile):
        return _sdf_rows(file)
    return _x3p_rows(file)


def _sdf_rows(sdf: SdfFile) -> list[tuple[str, str]]:
    header = sdf.header
    return [
        ("Version", header.magic),
        ("ManufacID", header.manufacturer_id),
        ("CreateDate", str(header.create_date) if header.create_date else ""),
        ("ModDate", str(header.mod_date) if header.mod_date else ""),
        ("NumPoints", str(header.num_points)),
        ("NumProfiles", str(header.num_profiles)),
        ("Xscale", f"{header.x_scale:g}"),
        ("Yscale", f"{header.y_scale:g}"),
        ("Zscale", f"{header.z_scale:g}"),
        ("Zresolution", f"{header.z_resolution:g}"),
        ("DataType", sdf.data_type.name.lower()),
    ]


def _date(value: datetime | None) -> str:
    return value.isoformat(timespec="milliseconds") if value is not None else ""


def _x3p_rows(x3p: x3pio.X3pFile) -> list[tuple[str, str]]:
    header = x3p.header
    placement = x3p.placement
    shape = x3p.layers[0].shape
    rows = [("Revision", x3p.revision.value), ("FeatureType", x3p.feature_type.value)]
    if isinstance(x3p, x3pio.PointCloud):
        rows.append(("ListDimension", str(shape[0])))
    else:
        num_rows, num_columns = (1, shape[0]) if isinstance(x3p, x3pio.Profile) else shape
        rows += [
            ("MatrixDimension.SizeX", str(num_columns)),
            ("MatrixDimension.SizeY", str(num_rows)),
            ("MatrixDimension.SizeZ", str(len(x3p.layers))),
        ]
    for index, (name, axis) in enumerate((("CX", header.x), ("CY", header.y), ("CZ", header.z))):
        rows += [
            (f"{name}.AxisType", axis.axis_type.name.lower()),
            (f"{name}.DataType", axis.data_type.name.lower()),
            (f"{name}.Increment", f"{axis.increment:g}"),
            (f"{name}.Offset", f"{placement.offset[index]:g}"),
        ]
    if placement.has_rotation:
        rotation = "; ".join(" ".join(f"{value:g}" for value in row) for row in placement.rotation)
        rows.append(("Rotation", rotation))
    if not x3p.revision.has_vendor_ids:
        vendor_ids = x3p.extensions.vendor_ids
        rows.append(("VendorSpecificID", vendor_ids[0] if vendor_ids else ""))
    metadata = x3p.metadata
    if metadata is not None:
        optional = [
            ("Creator", metadata.creator or ""),
            ("CalibrationDate", _date(metadata.calibration_date)),
            ("Comment", metadata.comment or ""),
        ]
        probing = metadata.probing_system
        rows += [
            ("Date", _date(metadata.date)),
            ("Instrument.Manufacturer", metadata.instrument.manufacturer),
            ("Instrument.Model", metadata.instrument.model),
            ("Instrument.Serial", metadata.instrument.serial),
            ("Instrument.Version", metadata.instrument.version),
            ("ProbingSystem.Type", probing.type.value if probing.type else ""),
            ("ProbingSystem.Identification", probing.identification),
        ]
        rows += [row for row in optional if row[1]]
    return rows


def extension_names(file: SurfaceFile) -> list[str]:
    """Return the IDs (or paths) of the vendor extensions of an x3p file, in file order.

    An SDF file has no extensions.
    """
    return [] if isinstance(file, SdfFile) else list(file.extensions)


def trailer_text(sdf: SdfFile) -> str:
    """Return the trailer (record 3) as text."""
    trailer = sdf.trailer
    # The trailer isn't validated on read, so it may not decode cleanly.
    return trailer if isinstance(trailer, str) else trailer.decode("ascii", errors="replace")


def format_metadata(file: SurfaceFile) -> str:
    """Return the header fields and the trailer or extensions as text.

    The layout is that of ``sdfio info``: ``name = value`` lines. An SDF file ends with its
    trailer, an x3p file with the list of its vendor extensions.
    """
    rows = metadata_rows(file)
    width = max(len(name) for name, _ in rows)
    lines = [f"{name:<{width}} = {value}" for name, value in rows]
    if isinstance(file, SdfFile):
        text = trailer_text(file)
        if text.strip():
            lines.append(f"{'Trailer':<{width}} =")
            lines.extend(text.splitlines())
    else:
        extensions = extension_names(file)
        if extensions:
            lines.append(f"{'VendorExtensions':<{width}} =")
            lines.extend(extensions)
    return "\n".join(lines) + "\n"


class InfoPanel(QWidget):
    """Read-only table of the header fields with the trailer or the vendor extensions below.

    ``extension_activated(name)`` is emitted when a vendor extension is double-clicked; ``name``
    is its ID, or its path for the standard before the amendment.
    """

    extension_activated = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty panel."""
        super().__init__(parent)
        self.setMinimumWidth(_MIN_COLUMNS * self.fontMetrics().averageCharWidth())
        self.table = QTableWidget(0, 2, self)
        self.table.verticalHeader().hide()
        self.table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header = self.table.horizontalHeader()
        header.hide()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        self.trailer = QPlainTextEdit(self)
        self.trailer.setReadOnly(True)
        self._trailer_label = QLabel(self.tr("Trailer"), self)
        self.extensions = QListWidget(self)
        self.extensions.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.extensions.setToolTip(self.tr("Double-click to save the file of an extension"))
        self.extensions.itemDoubleClicked.connect(
            lambda item: self.extension_activated.emit(item.text())
        )
        self._extensions_label = QLabel(self.tr("Vendor extensions"), self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        layout.addWidget(self._trailer_label)
        layout.addWidget(self.trailer, 1)
        layout.addWidget(self._extensions_label)
        layout.addWidget(self.extensions, 1)
        self._show_extensions(False)

    def set_file(self, file: SurfaceFile | None) -> None:
        """Show the header of ``file`` and its trailer or vendor extensions, or nothing."""
        rows = metadata_rows(file) if file is not None else []
        self.table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(value))
        self._fit_table(len(rows))

        is_sdf = isinstance(file, SdfFile)
        self._show_extensions(file is not None and not is_sdf)
        self.trailer.setPlainText(trailer_text(file) if isinstance(file, SdfFile) else "")
        self.extensions.clear()
        if file is not None and not isinstance(file, SdfFile):
            self.extensions.addItems(extension_names(file))

    def _fit_table(self, rows: int) -> None:
        """Show a short table completely, and let a long one scroll in the space it gets."""
        fixed = rows <= _MAX_FIXED_ROWS
        scrollbar = (
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff if fixed else Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        vertical = QSizePolicy.Policy.Fixed if fixed else QSizePolicy.Policy.Expanding
        self.table.setVerticalScrollBarPolicy(scrollbar)
        self.table.setSizePolicy(QSizePolicy.Policy.Preferred, vertical)
        self.table.updateGeometry()
        stretch = 0 if fixed else 2
        cast(QVBoxLayout, self.layout()).setStretchFactor(self.table, stretch)

    def _show_extensions(self, extensions: bool) -> None:
        self._trailer_label.setVisible(not extensions)
        self.trailer.setVisible(not extensions)
        self._extensions_label.setVisible(extensions)
        self.extensions.setVisible(extensions)
