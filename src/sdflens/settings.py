# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Persistent user settings with a typed attribute per key."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast, overload

from PySide6.QtCore import QByteArray, QSettings, QStandardPaths

from .colormap import DEFAULT_COLORMAP

__all__ = ["Settings"]


class _Setting[T]:
    """A setting whose type is that of its default value."""

    def __init__(self, key: str, default: T) -> None:
        self._key = key
        self._default = default

    @overload
    def __get__(self, obj: None, _owner: type) -> _Setting[T]: ...
    @overload
    def __get__(self, obj: Settings, _owner: type) -> T: ...
    def __get__(self, obj: Settings | None, _owner: type) -> _Setting[T] | T:
        if obj is None:
            return self
        return cast(T, obj._store.value(self._key, self._default, type=type(self._default)))

    def __set__(self, obj: Settings, value: T) -> None:
        obj._store.setValue(self._key, value)


class Settings:
    """The window state and the last choices of the user, kept between sessions."""

    view = _Setting("view", "3d")
    colormap = _Setting("colormap", DEFAULT_COLORMAP)
    reverse = _Setting("reverse", False)
    render_mode = _Setting("renderMode", "surface")
    save_version = _Setting("saveVersion", "")
    save_data_type = _Setting("saveDataType", "")
    last_directory = _Setting("lastDirectory", "")

    def __init__(self) -> None:
        """Open the settings of the application."""
        self._store = QSettings()

    @property
    def geometry(self) -> QByteArray | None:
        """The saved window geometry, if any."""
        value: Any = self._store.value("geometry")
        return value if isinstance(value, QByteArray) else None

    @geometry.setter
    def geometry(self, value: QByteArray) -> None:
        self._store.setValue("geometry", value)

    def dialog_directory(self) -> str:
        """Return the folder file dialogs start in: the last one used, else the documents."""
        last = self.last_directory
        if last and Path(last).is_dir():
            return last
        return QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)

    def remember_directory(self, path: str) -> None:
        """Start the next file dialog in the folder of ``path``."""
        self.last_directory = str(Path(path).parent)
