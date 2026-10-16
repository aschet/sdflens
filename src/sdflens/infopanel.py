# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Header metadata of the loaded file: panel widget and text export."""

from __future__ import annotations

from datetime import datetime
from typing import cast

import x3pio
from PySide6.QtCore import QDateTime, QLocale, QPoint, Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPlainTextEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from sdfio import SdfFile

from .fileinfo import InfoSection, describe_file
from .icons import load_icon
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

#: The trailer and the list of extensions show at most this many lines before they scroll.
_MAX_BLOCK_LINES = 10


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
        ("NumPoints", str(sdf.data.shape[1])),
        ("NumProfiles", str(sdf.data.shape[0])),
        ("Xscale", f"{header.x_scale:g}"),
        ("Yscale", f"{header.y_scale:g}"),
        ("Zscale", f"{header.z_scale:g}"),
        ("Zresolution", f"{header.z_resolution:g}"),
        ("DataType", header.data_type.name.lower()),
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


def _format_date(value: datetime) -> str:
    """Write ``value`` in the local time and the language of the system."""
    return QLocale().toString(QDateTime(value).toLocalTime(), QLocale.FormatType.ShortFormat)


class InfoPanel(QScrollArea):
    """Read-only description of the loaded file in sections of labelled values.

    The sections are those of :func:`~sdflens.fileinfo.describe_file`, as in sdflens web. Below them
    an SDF file has its trailer and an x3p file its vendor extensions, each only if it has some.

    ``extension_save_requested(name)`` is emitted when "Save as" is chosen in the context menu of a
    vendor extension; ``name`` is its ID, or its path for ISO 25178-72:2017.
    """

    extension_save_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty panel."""
        super().__init__(parent)
        self.setMinimumWidth(_MIN_COLUMNS * self.fontMetrics().averageCharWidth())
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        #: The sections that are shown, in order; empty without a file.
        self.sections: list[InfoSection] = []

        self.trailer = QPlainTextEdit(self)
        self.trailer.setReadOnly(True)
        self.extensions = QListWidget(self)
        self.extensions.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.extensions.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.extensions.customContextMenuRequested.connect(self._show_extension_menu)
        self.trailer.hide()
        self.extensions.hide()

        self._content = QWidget(self)
        self._layout = QVBoxLayout(self._content)
        self._layout.setContentsMargins(12, 10, 12, 10)
        self._layout.setSpacing(6)
        self.setWidget(self._content)

    def set_file(self, file: SurfaceFile | None) -> None:
        """Show the description of ``file`` and its trailer or vendor extensions, or nothing."""
        self._clear()
        self.sections = describe_file(file, _format_date) if file is not None else []
        # The values of all sections start at the same place, after the longest label.
        metrics = self.fontMetrics()
        label_width = max(
            (
                metrics.horizontalAdvance(self.tr(name))
                for section in self.sections
                for name, _ in section.rows
            ),
            default=0,
        )
        for section in self.sections:
            self._add_heading(section.title)
            if section.text:
                self._layout.addWidget(self._selectable(section.text, muted=False))
            else:
                self._layout.addWidget(self._rows(section.rows, label_width))

        text = trailer_text(file).strip() if isinstance(file, SdfFile) else ""
        self.trailer.setPlainText(text)
        names = extension_names(file) if file is not None else []
        self.extensions.clear()
        self.extensions.addItems(names)
        self.extensions.setCurrentRow(-1)
        self.extensions.clearSelection()
        if text:
            self._add_heading("Trailer")
            self._show_block(self.trailer, text.count("\n") + 1)
        if names:
            self._add_heading("Vendor extensions")
            self._show_block(self.extensions, len(names))
        self._layout.addStretch(1)

    def _clear(self) -> None:
        """Remove what is shown, keeping the trailer and the extensions for the next file."""
        while (item := self._layout.takeAt(0)) is not None:
            widget = item.widget()
            if widget in (self.trailer, self.extensions):
                cast(QWidget, widget).hide()
            elif widget is not None:
                widget.deleteLater()

    def _add_heading(self, title: str) -> None:
        heading = QLabel(self.tr(title).upper(), self._content)
        font = heading.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 0.85)
        heading.setFont(font)
        heading.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        heading.setContentsMargins(0, 10, 0, 0)
        self._layout.addWidget(heading)

    def _selectable(self, text: str, *, muted: bool) -> QLabel:
        """Return a label that wraps long text and lets it be selected and copied."""
        label = QLabel(text, self._content)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if muted:
            label.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        return label

    def _rows(self, rows: tuple[tuple[str, str], ...], label_width: int) -> QWidget:
        """Return the labelled values of a section: the label muted, the value beside it."""
        grid = QWidget(self._content)
        layout = QGridLayout(grid)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(3)
        layout.setColumnStretch(1, 1)
        layout.setColumnMinimumWidth(0, label_width)
        for index, (label, value) in enumerate(rows):
            name = self._selectable(self.tr(label), muted=True)
            name.setWordWrap(False)
            layout.addWidget(name, index, 0, Qt.AlignmentFlag.AlignTop)
            layout.addWidget(self._selectable(value, muted=False), index, 1)
        return grid

    def _show_block(self, block: QWidget, lines: int) -> None:
        """Show the trailer or the extensions with room for their lines, at most ten of them."""
        shown = min(lines, _MAX_BLOCK_LINES)
        block.setFixedHeight((shown + 1) * self.fontMetrics().lineSpacing() + 8)
        self._layout.addWidget(block)
        block.show()

    def extension_menu(self, item: QListWidgetItem) -> QMenu:
        """Return the context menu of the vendor extension ``item``."""
        menu = QMenu(self)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        save = menu.addAction(load_icon("save-as"), self.tr("Save &as..."))
        save.triggered.connect(lambda: self.extension_save_requested.emit(item.text()))
        return menu

    def _show_extension_menu(self, position: QPoint) -> None:
        item = self.extensions.itemAt(position) or self.extensions.currentItem()
        if item is not None:
            self.extension_menu(item).popup(self.extensions.viewport().mapToGlobal(position))
