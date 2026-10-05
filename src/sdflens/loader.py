# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Background loading of surface files: parsing and mesh building run off the GUI thread."""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QObject, QRunnable, QThreadPool, Signal

from .surface import NoMeasuredPointsError
from .surfacefile import FILE_ERRORS, build_mesh, build_model, read_surface_file

__all__ = ["Loader"]


class _Signals(QObject):
    loaded = Signal(str, object, object, object)
    failed = Signal(str, str)


class _Task(QRunnable):
    """Reads one file; touches no Qt objects other than emitting queued signals."""

    def __init__(self, path: str, signals: _Signals) -> None:
        super().__init__()
        self._path = path
        self._signals = signals

    def run(self) -> None:
        # Deliberately narrow: only expected failures get a message, anything else is a bug.
        try:
            file = read_surface_file(self._path)
            model = build_model(file)
            mesh = build_mesh(model)
        except NoMeasuredPointsError:
            message = QCoreApplication.translate("Loader", "The file contains no measured points")
            self._signals.failed.emit(self._path, message)
        except FILE_ERRORS as error:
            self._signals.failed.emit(self._path, str(error))
        else:
            self._signals.loaded.emit(self._path, file, model, mesh)


class Loader(QObject):
    """Loads one SDF or x3p file at a time on a worker thread.

    ``loaded(path, file, model, mesh)`` and ``failed(path, message)`` are delivered on the thread
    that owns the loader, so their slots may touch widgets and GL resources.
    """

    loaded = Signal(str, object, object, object)
    failed = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        """Create an idle loader."""
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._signals = _Signals(self)
        self._signals.loaded.connect(self._on_loaded)
        self._signals.failed.connect(self._on_failed)
        self._busy = False

    @property
    def busy(self) -> bool:
        """Whether a load is in progress."""
        return self._busy

    def load(self, path: str) -> bool:
        """Start loading ``path``; returns ``False`` if a load is already in progress."""
        if self._busy:
            return False
        self._busy = True
        self._pool.start(_Task(path, self._signals))
        return True

    def _on_loaded(self, path: str, file: object, model: object, mesh: object) -> None:
        self._busy = False
        self.loaded.emit(path, file, model, mesh)

    def _on_failed(self, path: str, message: str) -> None:
        self._busy = False
        self.failed.emit(path, message)
