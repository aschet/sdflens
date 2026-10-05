# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Dialog choosing the format, version and data type of a saved SDF or x3p file."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import x3pio
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QWidget
from sdfio import DataType, FileFormat, SdfDialect, SdfFile
from sdfio.header import ASCII_PREFIX, BINARY_PREFIX

from .convert import (
    ExportProblem,
    ProblemKind,
    allowed_data_types,
    format_problem,
    not_a_grid,
    x3p_data_types,
    x3p_to_sdf,
)
from .surfacefile import SurfaceFile

__all__ = ["SaveAsDialog", "x3p_code"]

_X3P_CODES = {
    x3pio.X3pDialect.ISO25178_72_2017_DAM1: "DAM1",
    x3pio.X3pDialect.ISO5436_2000: "2000",
}


def x3p_code(dialect: x3pio.X3pDialect) -> str:
    """Return the short name of an x3p dialect, ``DAM1`` or ``2000``."""
    return _X3P_CODES[dialect]


def _magic(dialect: SdfDialect, file_format: FileFormat) -> str:
    """Return the file magic, for example ``bISO-2.0``, of a version in a format."""
    return f"{BINARY_PREFIX if file_format is FileFormat.BINARY else ASCII_PREFIX}{dialect}"


def _x3p_label(dialect: x3pio.X3pDialect, storage: x3pio.DataStorage) -> str:
    """Return the label, for example ``x3p-DAM1 (binary)``, of an x3p dialect and storage."""
    return f"x3p-{x3p_code(dialect)} ({storage.name.lower()})"


@dataclass(frozen=True)
class _Target:
    """A format and version a file can be saved in."""

    dialect: SdfDialect | x3pio.X3pDialect
    file_format: FileFormat | x3pio.DataStorage
    label: str

    @property
    def is_sdf(self) -> bool:
        return isinstance(self.dialect, SdfDialect)


def _targets() -> list[_Target]:
    sdf = [
        _Target(dialect, file_format, _magic(dialect, file_format))
        for file_format in (FileFormat.BINARY, FileFormat.ASCII)
        for dialect in SdfDialect
    ]
    x3p = [
        _Target(dialect, storage, _x3p_label(dialect, storage))
        for dialect in (x3pio.X3pDialect.ISO25178_72_2017_DAM1, x3pio.X3pDialect.ISO5436_2000)
        for storage in (x3pio.DataStorage.BINARY, x3pio.DataStorage.XML)
    ]
    return sdf + x3p


class SaveAsDialog(QDialog):
    """Options for saving the loaded surface as an SDF or an x3p file, preset from that file.

    Every version can be chosen; data types the chosen SDF version does not define are
    disabled. An SDF version the file cannot be saved in has the reason as tooltip, except for a
    file that is not a grid of heights, which cannot be saved as SDF at all and has those
    versions disabled.
    """

    def __init__(
        self,
        file: SurfaceFile,
        parent: QWidget | None = None,
        last: tuple[str, str] | None = None,
    ) -> None:
        """Create the dialog with the options of ``file`` selected.

        ``last`` is the ``(version, data type)`` chosen the previous time, as returned by
        :attr:`version` and :attr:`data_type_name`. It replaces the options of ``file`` if valid.
        """
        super().__init__(parent)
        self._sdf: SdfFile | None = None
        if not not_a_grid(file):
            self._sdf = file if isinstance(file, SdfFile) else x3p_to_sdf(file)
        self.setWindowTitle(self.tr("Save as"))

        self._targets = _targets()
        self._version = QComboBox(self)
        for target in self._targets:
            self._version.addItem(target.label)
        self._data_type = QComboBox(self)
        self._data_family: bool | None = None

        layout = QFormLayout(self)
        layout.addRow(self.tr("Version:"), self._version)
        layout.addRow(self.tr("Data type:"), self._data_type)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        if isinstance(file, SdfFile):
            self._version.setCurrentText(file.header.magic)
            data_type = DataType(file.header.data_type).name.lower()
        else:
            self._version.setCurrentText(_x3p_label(file.header.dialect, file.storage))
            z_type = file.header.z.data_type
            data_type = z_type.name.lower()
        self._update_availability()
        self._select_data_type(data_type)
        if last is not None:
            self._apply_last(*last)
        self._update_availability()
        self._version.currentIndexChanged.connect(lambda _index: self._update_availability())

    @property
    def _target(self) -> _Target:
        return self._targets[self._version.currentIndex()]

    @property
    def version(self) -> str:
        """The chosen version with its format, for example ``bISO-2.0`` or ``x3p-DAM1 (binary)``."""
        return self._version.currentText()

    @property
    def is_x3p(self) -> bool:
        """Whether an x3p file is chosen, not an SDF file."""
        return not self._target.is_sdf

    @property
    def dialect(self) -> SdfDialect | x3pio.X3pDialect:
        """The chosen version without the format, for example ``ISO-2.0``."""
        return self._target.dialect

    @property
    def file_format(self) -> FileFormat | x3pio.DataStorage:
        """The chosen ASCII or binary representation, or the storage of an x3p file."""
        return self._target.file_format

    @property
    def data_type(self) -> DataType | x3pio.DataType:
        """The chosen storage type of the heights."""
        data = self._data_type.currentData()
        return DataType(data) if self._target.is_sdf else x3pio.DataType(data)

    @property
    def data_type_name(self) -> str:
        """The chosen data type as shown in the dialog, for example ``int32``."""
        return self._data_type.currentText()

    def _select_data_type(self, name: str) -> None:
        index = self._data_type.findText(name)
        if index >= 0:
            self._data_type.setCurrentIndex(index)

    def _apply_last(self, version: str, data_type: str) -> None:
        index = self._version.findText(version)
        if index >= 0 and self._problem(self._targets[index]) is None:
            self._version.setCurrentIndex(index)
            self._update_availability()
        self._select_data_type(data_type)

    def _problem(self, target: _Target) -> ExportProblem | None:
        """Return why the file cannot be saved in ``target``, or ``None``."""
        if not isinstance(target.dialect, SdfDialect) or self._sdf is None:
            return None
        file_format = cast(FileFormat, target.file_format)
        return format_problem(self._sdf, target.dialect, file_format)

    def _problem_text(self, problem: ExportProblem | None, dialect: SdfDialect) -> str | None:
        if problem is None:
            return None
        if problem.kind is ProblemKind.BINARY_LIMIT:
            return self.tr(
                "The binary format of {version} stores at most {limit} points per row"
            ).format(version=dialect, limit=problem.limit)
        return self.tr(
            'The trailer must be in the tagged "Name = Value" format for {version}'
        ).format(version=dialect)

    @staticmethod
    def _set_available(combo: QComboBox, index: int, problem: str | None) -> None:
        item = cast(QStandardItemModel, combo.model()).item(index)
        item.setEnabled(problem is None)
        item.setToolTip(problem or "")

    def _fill_data_types(self, is_sdf: bool) -> None:
        """List the data types of SDF or of x3p, keeping the chosen name if it still exists."""
        if self._data_family is is_sdf:
            return
        previous = self._data_type.currentText()
        self._data_family = is_sdf
        self._data_type.clear()
        types = list(DataType) if is_sdf else x3p_data_types()
        for data_type in types:
            self._data_type.addItem(data_type.name.lower(), data_type)
        self._select_data_type(previous)

    def _update_availability(self) -> None:
        """Disable what the file or the chosen version cannot do, and pick a valid data type."""
        versions = cast(QStandardItemModel, self._version.model())
        for index, target in enumerate(self._targets):
            item = versions.item(index)
            if not isinstance(target.dialect, SdfDialect):
                continue
            item.setEnabled(self._sdf is not None)  # only a grid of heights is an SDF file
            problem = self._problem_text(self._problem(target), target.dialect)
            item.setToolTip(problem or "")

        target = self._target
        self._fill_data_types(target.is_sdf)
        if not isinstance(target.dialect, SdfDialect):
            return
        first_valid: int | None = None
        allowed = allowed_data_types(target.dialect)
        for index in range(self._data_type.count()):
            data_type = DataType(self._data_type.itemData(index))
            supported = data_type in allowed
            problem = None
            if not supported:
                problem = self.tr("Not defined for version {version}").format(
                    version=target.dialect
                )
            self._set_available(self._data_type, index, problem)
            if supported and first_valid is None:
                first_valid = index
        if self.data_type not in allowed and first_valid is not None:
            self._data_type.setCurrentIndex(first_valid)
