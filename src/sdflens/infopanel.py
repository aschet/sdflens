# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Header metadata of the loaded file: panel widget and text export."""

from __future__ import annotations

from datetime import datetime
from typing import cast

from PySide6.QtCore import QCoreApplication, QDateTime, QLocale, QPoint, Qt, Signal
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
    QSizePolicy,
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
    "trailer_text",
]

#: Narrowest width in characters; longer values wrap.
_MIN_COLUMNS = 28

#: The trailer and the list of extensions may shrink to this many lines to fit the panel.
_MIN_BLOCK_LINES = 2


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


def _translated(text: str) -> str:
    """Return ``text`` as the panel shows it, in the language of the system."""
    return QCoreApplication.translate("InfoPanel", text)


def format_metadata(file: SurfaceFile) -> str:
    """Return the information of ``file`` as text, as the panel shows it.

    The sections are those of the panel, each with its title in capitals and its labelled values
    below it, followed by the trailer of an SDF file or the vendor extensions of an x3p file.
    """
    sections = describe_file(file, _format_date)
    width = max(
        (len(_translated(name)) for section in sections for name, _ in section.rows), default=0
    )
    blocks: list[list[str]] = []
    for section in sections:
        lines = [_translated(section.title).upper()]
        if section.text:
            lines.append(section.text)
        lines += [f"{_translated(name):<{width}}  {value}" for name, value in section.rows]
        blocks.append(lines)
    trailer = trailer_text(file).strip() if isinstance(file, SdfFile) else ""
    if trailer:
        blocks.append([_translated("Trailer").upper(), *trailer.splitlines()])
    extensions = extension_names(file)
    if extensions:
        blocks.append([_translated("Vendor extensions").upper(), *extensions])
    return "\n\n".join("\n".join(lines) for lines in blocks) + "\n"


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
                metrics.horizontalAdvance(_translated(name))
                for section in self.sections
                for name, _ in section.rows
            ),
            default=0,
        )
        for section in self.sections:
            self._add_heading(section.title)
            if section.text:
                self._layout.addWidget(self._selectable(section.text))
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
        if not text and not names:
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
        heading = QLabel(_translated(title).upper(), self._content)
        font = heading.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 0.85)
        heading.setFont(font)
        heading.setContentsMargins(0, 10, 0, 0)
        self._layout.addWidget(heading)

    def _selectable(self, text: str) -> QLabel:
        """Return a label that wraps long text and lets it be selected and copied."""
        label = QLabel(text, self._content)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _rows(self, rows: tuple[tuple[str, str], ...], label_width: int) -> QWidget:
        """Return the labelled values of a section: each label with its value beside it."""
        grid = QWidget(self._content)
        layout = QGridLayout(grid)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(3)
        layout.setColumnStretch(1, 1)
        layout.setColumnMinimumWidth(0, label_width)
        for index, (label, value) in enumerate(rows):
            name = self._selectable(_translated(label))
            name.setWordWrap(False)
            layout.addWidget(name, index, 0, Qt.AlignmentFlag.AlignTop)
            layout.addWidget(self._selectable(value), index, 1)
        return grid

    def _show_block(self, block: QWidget, lines: int) -> None:
        """Show the trailer or the extensions: the height the panel has left."""
        spacing = self.fontMetrics().lineSpacing()
        block.setMinimumHeight((min(lines, _MIN_BLOCK_LINES) + 1) * spacing + 8)
        block.setMaximumHeight((lines + 1) * spacing + 8)
        # Its own size hint is ignored: the panel gives it what the sections leave free.
        block.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self._layout.addWidget(block, 1)
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
