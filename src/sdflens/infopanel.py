# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Header metadata of the loaded file: table widget and text export."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from sdfio import SdfFile

__all__ = ["InfoPanel", "format_metadata", "metadata_rows", "trailer_text"]


def metadata_rows(sdf: SdfFile) -> list[tuple[str, str]]:
    """Return the header fields as ``(name, value)`` rows, named as in the standard.

    Lengths are in meters, like in ``sdfio info``.
    """
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


def trailer_text(sdf: SdfFile) -> str:
    """Return the trailer (record 3) as text."""
    trailer = sdf.trailer
    # The trailer isn't validated on read, so it may not decode cleanly.
    return trailer if isinstance(trailer, str) else trailer.decode("ascii", errors="replace")


def format_metadata(sdf: SdfFile) -> str:
    """Return the header fields and the trailer as text, in the layout of ``sdfio info``."""
    rows = metadata_rows(sdf)
    width = max(len(name) for name, _ in rows)
    lines = [f"{name:<{width}} = {value}" for name, value in rows]
    text = trailer_text(sdf)
    if text.strip():
        lines.append(f"{'Trailer':<{width}} =")
        lines.extend(text.splitlines())
    return "\n".join(lines) + "\n"


class InfoPanel(QWidget):
    """Read-only table of the header fields with the trailer below it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty panel."""
        super().__init__(parent)
        self.table = QTableWidget(0, 2, self)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header = self.table.horizontalHeader()
        header.hide()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        self.trailer = QPlainTextEdit(self)
        self.trailer.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        layout.addWidget(QLabel(self.tr("Trailer"), self))
        layout.addWidget(self.trailer, 1)

    def set_sdf(self, sdf: SdfFile | None) -> None:
        """Show the header and trailer of ``sdf``, or nothing."""
        rows = metadata_rows(sdf) if sdf is not None else []
        self.table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(value))
        self.table.setFixedHeight(
            self.table.verticalHeader().length() + 2 * self.table.frameWidth()
        )
        self.trailer.setPlainText(trailer_text(sdf) if sdf is not None else "")
