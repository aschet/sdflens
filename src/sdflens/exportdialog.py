# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Dialog choosing the version, format and data type of a saved SDF file."""

from __future__ import annotations

from typing import cast

from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QWidget
from sdfio import DataType, FileFormat, SdfDialect, SdfFile
from sdfio.header import ASCII_PREFIX, BINARY_PREFIX

from .convert import ExportProblem, ProblemKind, allowed_data_types, format_problem

__all__ = ["SaveAsDialog"]


def _magic(dialect: SdfDialect, file_format: FileFormat) -> str:
    """Return the file magic, for example ``bISO-2.0``, of a version in a format."""
    return f"{BINARY_PREFIX if file_format is FileFormat.BINARY else ASCII_PREFIX}{dialect}"


class SaveAsDialog(QDialog):
    """Options for saving the loaded surface as an SDF file, preset from that file.

    Every version can be chosen; data types the chosen version does not define are disabled.
    A version the file cannot be saved in has the reason as tooltip.
    """

    def __init__(
        self,
        sdf: SdfFile,
        parent: QWidget | None = None,
        last: tuple[str, str] | None = None,
    ) -> None:
        """Create the dialog with the options of ``sdf`` preselected.

        ``last`` is the ``(version, data type)`` chosen the previous time, as returned by
        :attr:`version` and :attr:`data_type_name`. It replaces the options of ``sdf`` if valid.
        """
        super().__init__(parent)
        self._sdf = sdf
        self.setWindowTitle(self.tr("Save as"))

        self._choices = [
            (dialect, file_format)
            for file_format in (FileFormat.BINARY, FileFormat.ASCII)
            for dialect in SdfDialect
        ]
        self._version = QComboBox(self)
        for dialect, file_format in self._choices:
            self._version.addItem(_magic(dialect, file_format))
        self._data_type = QComboBox(self)
        for data_type in DataType:
            self._data_type.addItem(data_type.name.lower(), data_type)

        layout = QFormLayout(self)
        layout.addRow(self.tr("Version:"), self._version)
        layout.addRow(self.tr("Data type:"), self._data_type)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        header = sdf.header
        self._version.setCurrentText(header.magic)
        self._select_data_type(DataType(header.data_type).name.lower())
        if last is not None:
            self._apply_last(*last)
        self._update_availability()
        self._version.currentIndexChanged.connect(lambda _index: self._update_availability())

    @property
    def version(self) -> str:
        """The chosen file magic, for example ``bISO-2.0``."""
        return self._version.currentText()

    @property
    def dialect(self) -> SdfDialect:
        """The chosen version without the format, for example ``ISO-2.0``."""
        return self._choices[self._version.currentIndex()][0]

    @property
    def file_format(self) -> FileFormat:
        """The chosen ASCII or binary representation."""
        return self._choices[self._version.currentIndex()][1]

    @property
    def data_type(self) -> DataType:
        """The chosen data area storage type."""
        return DataType(self._data_type.currentData())

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
        if index >= 0:
            dialect, file_format = self._choices[index]
            if format_problem(self._sdf, dialect, file_format) is None:
                self._version.setCurrentIndex(index)
        self._select_data_type(data_type)

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

    def _update_availability(self) -> None:
        """Disable the data types the chosen version does not define, and pick a valid one."""
        versions = cast(QStandardItemModel, self._version.model())
        for index, (dialect, file_format) in enumerate(self._choices):
            problem = self._problem_text(format_problem(self._sdf, dialect, file_format), dialect)
            versions.item(index).setToolTip(problem or "")

        dialect = self.dialect
        first_valid: int | None = None
        allowed = allowed_data_types(dialect)
        for index in range(self._data_type.count()):
            data_type = DataType(self._data_type.itemData(index))
            supported = data_type in allowed
            problem = None
            if not supported:
                problem = self.tr("Not defined for version {version}").format(version=dialect)
            self._set_available(self._data_type, index, problem)
            if supported and first_valid is None:
                first_valid = index
        if self.data_type not in allowed and first_valid is not None:
            self._data_type.setCurrentIndex(first_valid)
