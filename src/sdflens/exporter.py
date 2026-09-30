# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""File dialogs and writers for everything the application saves or copies."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import sdfio
from PySide6.QtCore import QDir, QObject, Qt, Signal
from PySide6.QtGui import QBrush, QGuiApplication, QImage, QPainter
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

from .convert import convert_for_export
from .exportdialog import SaveAsDialog
from .infopanel import format_metadata
from .settings import Settings

__all__ = ["Exporter", "compose_screenshot"]


def compose_screenshot(view: QImage, bar: QImage, background: QBrush) -> QImage:
    """Return ``view`` with the color ``bar`` to its right, both in device pixels."""
    # Both grabs share one device pixel ratio; compose them in device pixels.
    view.setDevicePixelRatio(1.0)
    bar.setDevicePixelRatio(1.0)
    canvas = QImage(
        view.width() + bar.width(), max(view.height(), bar.height()), QImage.Format.Format_RGB32
    )
    painter = QPainter(canvas)
    painter.fillRect(canvas.rect(), background)
    painter.drawImage(0, 0, view)
    painter.drawImage(view.width(), 0, bar)
    painter.end()
    return canvas


class Exporter(QObject):
    """Asks where to save, writes the file and reports the outcome.

    ``message(text, timeout_ms)`` is emitted for successes; failures are shown in a message box
    over ``window``. ``source`` arguments name the loaded file, used to suggest file names.
    """

    message = Signal(str, int)

    def __init__(self, window: QWidget, settings: Settings) -> None:
        """Create an exporter whose dialogs are children of ``window``."""
        super().__init__(window)
        self._window = window
        self._settings = settings

    def save_as(self, sdf: sdfio.SdfFile, source: str | None) -> None:
        """Save ``sdf`` in a version, format and data type chosen by the user."""
        last = (self._settings.save_version, self._settings.save_data_type)
        dialog = SaveAsDialog(sdf, self._window, last)
        if dialog.exec() != SaveAsDialog.DialogCode.Accepted:
            return
        stem = Path(source).stem if source else "surface"
        path = self._ask_save_path(
            self.tr("Save as"),
            f"{stem}-{dialog.dialect}.sdf",
            self.tr("Surface data file (*.sdf);;All files (*)"),
        )
        if not path:
            return
        if not Path(path).suffix:
            path += ".sdf"
        self._settings.save_version = dialog.version
        self._settings.save_data_type = dialog.data_type_name
        self.write_export(sdf, path, dialog.dialect, dialog.data_type, dialog.file_format)

    def write_export(
        self,
        sdf: sdfio.SdfFile,
        path: str,
        dialect: sdfio.SdfDialect,
        data_type: sdfio.DataType,
        file_format: sdfio.FileFormat,
    ) -> None:
        """Save ``sdf`` to ``path`` converted to the given version, data type and format."""
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        error: Exception | None = None
        try:
            convert_for_export(sdf, dialect, data_type).save(path, format=file_format)
        except (sdfio.SdfError, OSError) as exc:
            error = exc
        finally:
            QApplication.restoreOverrideCursor()
        if error is not None:
            QMessageBox.critical(self._window, self.tr("Cannot save file"), f"{path}\n\n{error}")
        else:
            self.message.emit(self.tr("Saved {path}").format(path=path), 5000)

    def save_screenshot(self, source: str | None, capture: Callable[[], QImage | None]) -> None:
        """Save the image returned by ``capture`` as a PNG file."""
        path = self._ask_save_path(
            self.tr("Save screenshot"),
            f"{Path(source).stem}.png" if source else "screenshot.png",
            self.tr("PNG image (*.png)"),
        )
        if not path:
            return
        if Path(path).suffix.lower() != ".png":
            path += ".png"
        image = capture()
        if image is not None and not image.save(path):
            QMessageBox.critical(self._window, self.tr("Cannot save screenshot"), path)

    def copy_screenshot(self, image: QImage | None) -> None:
        """Put ``image`` on the clipboard."""
        if image is not None:
            QGuiApplication.clipboard().setImage(image)
            self.message.emit(self.tr("Screenshot copied to clipboard"), 3000)

    def export_metadata(self, sdf: sdfio.SdfFile, source: str | None) -> None:
        """Save the header fields and trailer of ``sdf`` as a text file."""
        path = self._ask_save_path(
            self.tr("Export information"),
            f"{Path(source).stem}.txt" if source else "information.txt",
            self.tr("Text files (*.txt);;All files (*)"),
        )
        if not path:
            return
        try:
            Path(path).write_text(format_metadata(sdf), encoding="utf-8")
        except OSError as error:
            QMessageBox.critical(
                self._window, self.tr("Cannot export information"), f"{path}\n\n{error}"
            )

    def copy_metadata(self, sdf: sdfio.SdfFile) -> None:
        """Put the header fields and trailer of ``sdf`` on the clipboard."""
        QGuiApplication.clipboard().setText(format_metadata(sdf))
        self.message.emit(self.tr("Information copied to clipboard"), 3000)

    def _ask_save_path(self, title: str, name: str, name_filter: str) -> str:
        """Ask for a file to save to, or return an empty string; remembers the folder."""
        path, _ = QFileDialog.getSaveFileName(
            self._window,
            title,
            QDir(self._settings.dialog_directory()).filePath(name),
            name_filter,
        )
        if not path:
            return ""
        self._settings.remember_directory(path)
        return QDir.toNativeSeparators(path)
