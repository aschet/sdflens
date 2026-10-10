# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""File dialogs and writers for everything the application saves or copies."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import sdfio
import x3pio
from PySide6.QtCore import QDir, QObject, Qt, Signal
from PySide6.QtGui import QBrush, QGuiApplication, QImage, QPainter
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget

from .convert import convert_file
from .exportdialog import SaveOptionsDialog
from .infopanel import format_metadata
from .settings import Settings
from .surfacefile import FILE_ERRORS, SurfaceFile, is_grid

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

    def save_as(self, file: SurfaceFile, source: str | None, layer: int = 0) -> None:
        """Ask for a file name and format, then for the options of the format, and save ``file``.

        An SDF file holds one grid, so of several layers the one asked for, counted from 0, is
        saved as SDF; an x3p file keeps all of them.

        The format is chosen by the file type of the dialog, which is SDF unless the data is a
        point cloud, or by the extension that is typed. Only a grid of heights can be saved as
        SDF.
        """
        sdf_filter = self.tr("Surface data file (*.sdf)")
        x3p_filter = self.tr("x3p file (*.x3p)")
        can_sdf = is_grid(file)
        default_filter = sdf_filter if can_sdf else x3p_filter
        stem = Path(source).stem if source else "surface"
        suffix = ".sdf" if default_filter == sdf_filter else ".x3p"
        path, chosen = self._ask_save_as(
            stem + suffix,
            [(sdf_filter, ".sdf"), (x3p_filter, ".x3p")] if can_sdf else [(x3p_filter, ".x3p")],
            default_filter,
        )
        if not path:
            return
        self._settings.remember_directory(path)
        path = QDir.toNativeSeparators(path)
        typed = Path(path).suffix.lower()
        is_x3p = typed == ".x3p" or (typed != ".sdf" and chosen == x3p_filter)
        if not is_x3p and not can_sdf:
            QMessageBox.critical(
                self._window,
                self.tr("Cannot save file"),
                self.tr("Only a grid of heights can be saved as an SDF file, not this data."),
            )
            return
        if not typed:
            path += ".x3p" if is_x3p else ".sdf"

        settings = self._settings
        last = (
            (settings.save_x3p_version, settings.save_x3p_encoding, settings.save_x3p_data_type)
            if is_x3p
            else (
                settings.save_sdf_version,
                settings.save_sdf_encoding,
                settings.save_sdf_data_type,
            )
        )
        dialog = SaveOptionsDialog(file, is_x3p, self._window, last, layer)
        if dialog.exec() != SaveOptionsDialog.DialogCode.Accepted:
            return
        if is_x3p:
            settings.save_x3p_version = dialog.version_name
            settings.save_x3p_encoding = dialog.encoding_name
            settings.save_x3p_data_type = dialog.data_type_name
        else:
            settings.save_sdf_version = dialog.version_name
            settings.save_sdf_encoding = dialog.encoding_name
            settings.save_sdf_data_type = dialog.data_type_name
        self.write_export(file, path, dialog.dialect, dialog.data_type, dialog.file_format, layer)

    def write_export(
        self,
        file: SurfaceFile,
        path: str,
        dialect: sdfio.SdfDialect | x3pio.Revision,
        data_type: sdfio.DataType | x3pio.DataType,
        file_format: sdfio.FileFormat | x3pio.DataStorage,
        layer: int = 0,
    ) -> None:
        """Save ``file`` to ``path`` converted to the given version, data type and format.

        ``layer`` is the layer that is saved as an SDF file, counted from 0.
        """
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        error: Exception | None = None
        dropped = False
        try:
            converted = convert_file(file, dialect, data_type, layer)
            if isinstance(converted, sdfio.SdfFile):
                converted.save(path, format=cast(sdfio.FileFormat, file_format))
            else:
                converted.save(path, storage=cast(x3pio.DataStorage, file_format))
                dropped = (
                    isinstance(file, x3pio.X3pFile)
                    and bool(file.extensions)
                    and not converted.extensions
                )
        except FILE_ERRORS as exc:
            error = exc
        finally:
            QApplication.restoreOverrideCursor()
        if error is not None:
            QMessageBox.critical(self._window, self.tr("Cannot save file"), f"{path}\n\n{error}")
        elif dropped:
            self.message.emit(
                self.tr("Saved {path}; the vendor extensions could not be kept").format(path=path),
                8000,
            )
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

    def export_metadata(self, file: SurfaceFile, source: str | None) -> None:
        """Save the header fields and trailer or vendor extensions of ``file`` as a text file."""
        path = self._ask_save_path(
            self.tr("Export information"),
            f"{Path(source).stem}.txt" if source else "information.txt",
            self.tr("Text files (*.txt);;All files (*)"),
        )
        if not path:
            return
        try:
            Path(path).write_text(format_metadata(file), encoding="utf-8")
        except OSError as error:
            QMessageBox.critical(
                self._window, self.tr("Cannot export information"), f"{path}\n\n{error}"
            )

    def copy_metadata(self, file: SurfaceFile) -> None:
        """Put the header fields and trailer or vendor extensions of ``file`` on the clipboard."""
        QGuiApplication.clipboard().setText(format_metadata(file))
        self.message.emit(self.tr("Information copied to clipboard"), 3000)

    def export_extension(self, name: str, content: bytes | None) -> None:
        """Save the file of the vendor extension ``name`` under the last part of its ID."""
        if content is None:
            self.message.emit(self.tr("The vendor extension has no file"), 5000)
            return
        base = name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "extension"
        path = self._ask_save_path(self.tr("Save vendor extension"), base, self.tr("All files (*)"))
        if not path:
            return
        try:
            Path(path).write_bytes(content)
        except OSError as error:
            QMessageBox.critical(
                self._window, self.tr("Cannot save vendor extension"), f"{path}\n\n{error}"
            )
        else:
            self.message.emit(self.tr("Saved {path}").format(path=path), 5000)

    def _ask_save_as(
        self, name: str, filters: list[tuple[str, str]], default_filter: str
    ) -> tuple[str, str]:
        """Ask for a file to save to, with a file type for each ``(filter, suffix)``.

        Qt's own dialog is used, because it changes the extension of the name when another file
        type is chosen, so that the name and the type do not contradict each other. Returns the
        path and the chosen filter, or empty strings if the dialog is cancelled.
        """
        dialog = QFileDialog(self._window, self.tr("Save as"), self._settings.dialog_directory())
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        # The native dialogs of some platforms keep the old extension when the type is changed.
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
        dialog.setNameFilters([text for text, _suffix in filters])
        dialog.selectNameFilter(default_filter)
        dialog.selectFile(name)
        if dialog.exec() != QFileDialog.DialogCode.Accepted or not dialog.selectedFiles():
            return "", ""
        return dialog.selectedFiles()[0], dialog.selectedNameFilter()

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
