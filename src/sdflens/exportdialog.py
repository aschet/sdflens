# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Dialog choosing the version, encoding and data type of a saved SDF or x3p file."""

from __future__ import annotations

from typing import cast

import x3pio
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLayout,
    QWidget,
)
from sdfio import DataType, FileFormat, SdfDialect, SdfFile

from .convert import (
    ExportProblem,
    ProblemKind,
    addable_z_offset,
    allowed_data_types,
    extensions_fit,
    format_problem,
    x3p_data_types,
    x3p_to_sdf,
)
from .surfacefile import SurfaceFile, is_grid

__all__ = ["SaveOptionsDialog"]

_X3P_REVISIONS = (x3pio.Revision.ISO25178_72_2017_DAM1, x3pio.Revision.ISO5436_2000)
_NEWEST_SDF = str(SdfDialect.ISO_2_0)
_NEWEST_X3P = x3pio.Revision.ISO25178_72_2017_DAM1.value


class SaveOptionsDialog(QDialog):
    """Version, encoding and data type of the SDF or x3p file that the user is about to save.

    The format has been chosen before, in the file dialog. The options are preset from the
    loaded file where it has the same format, otherwise the newest version in binary encoding
    and the widest data type. What the chosen options cannot do is said in a note below them, and
    the dialog cannot be accepted while the file cannot be saved with them.
    """

    def __init__(
        self,
        file: SurfaceFile,
        is_x3p: bool,
        parent: QWidget | None = None,
        last: tuple[str, str, str] | None = None,
        layer: int = 0,
    ) -> None:
        """Create the dialog for saving ``file`` in the format of ``is_x3p``.

        ``layer`` is the layer that is saved as an SDF file, counted from 0.

        ``last`` is the ``(version, encoding, data type)`` chosen the previous time in this format,
        as shown by :attr:`version_name`, :attr:`encoding_name` and :attr:`data_type_name`. It
        replaces the options of ``file`` where it is valid.
        """
        super().__init__(parent)
        self._file = file
        self._is_x3p = is_x3p
        self._layer = layer
        self._sdf: SdfFile | None = None
        if not is_x3p and is_grid(file):
            self._sdf = file if isinstance(file, SdfFile) else x3p_to_sdf(file, layer)
        # An SDF file has no coordinate system, so the z offset of an x3p file is lost unless it
        # is added to the heights; an x3p file keeps its placement.
        offset = 0.0 if is_x3p else addable_z_offset(file)
        self.setWindowTitle(self.tr("Save as x3p") if is_x3p else self.tr("Save as SDF"))

        self._version = QComboBox(self)
        self._encoding = QComboBox(self)
        self._data_type = QComboBox(self)
        if is_x3p:
            for x3p_revision in _X3P_REVISIONS:
                self._version.addItem(x3p_revision.value, x3p_revision)
            self._encoding.addItem(self.tr("Binary"), x3pio.DataStorage.BINARY)
            self._encoding.addItem(self.tr("XML"), x3pio.DataStorage.XML)
            for x3p_type in x3p_data_types():
                self._data_type.addItem(x3p_type.name.lower(), x3p_type)
        else:
            for sdf_dialect in SdfDialect:
                self._version.addItem(str(sdf_dialect), sdf_dialect)
            self._encoding.addItem(self.tr("Binary"), FileFormat.BINARY)
            self._encoding.addItem(self.tr("ASCII"), FileFormat.ASCII)
            for sdf_type in DataType:
                self._data_type.addItem(sdf_type.name.lower(), sdf_type)

        # An x3p file keeps all its layers unless the user asks for the one that is shown.
        layers = len(file.layers) if isinstance(file, x3pio.X3pFile) else 1
        self._single_layer = QCheckBox(self.tr("Save only the current layer"), self)
        self._single_layer.setVisible(is_x3p and layers > 1)
        self._z_offset = QCheckBox(self.tr("Add the z offset to the heights"), self)
        self._z_offset.setToolTip(
            self.tr("SDF has no coordinate system, so the z offset of the placement is lost.")
        )
        self._z_offset.setVisible(offset != 0.0)

        self._note = QLabel(self)
        self._note.setWordWrap(True)
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout = QFormLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)  # follows the note
        layout.addRow(self.tr("Version:"), self._version)
        layout.addRow(self.tr("Encoding:"), self._encoding)
        layout.addRow(self.tr("Data type:"), self._data_type)
        layout.addRow(self._single_layer)
        layout.addRow(self._z_offset)
        layout.addRow(self._note)
        layout.addRow(self._buttons)

        self._preset()
        if last is not None:
            self._apply_last(*last)
        self._update()
        self._version.currentIndexChanged.connect(lambda _index: self._update())
        self._encoding.currentIndexChanged.connect(lambda _index: self._update())
        self._z_offset.toggled.connect(lambda _checked: self._update())

    @property
    def dialect(self) -> SdfDialect | x3pio.Revision:
        """The chosen version of the standard."""
        data = self._version.currentData()
        return x3pio.Revision(data) if self._is_x3p else SdfDialect(data)

    @property
    def file_format(self) -> FileFormat | x3pio.DataStorage:
        """The chosen encoding: binary or ASCII for SDF, binary or XML for x3p."""
        data = self._encoding.currentData()
        return x3pio.DataStorage(data) if self._is_x3p else FileFormat(data)

    @property
    def data_type(self) -> DataType | x3pio.DataType:
        """The chosen storage type of the heights."""
        data = self._data_type.currentData()
        return x3pio.DataType(data) if self._is_x3p else DataType(data)

    @property
    def single_layer(self) -> bool:
        """Whether only the layer that is shown is saved, as an x3p file of layers can be."""
        return self._single_layer.isVisibleTo(self) and self._single_layer.isChecked()

    @property
    def apply_z_offset(self) -> bool:
        """Whether the z offset of the placement is added to the heights of an SDF file."""
        return self._z_offset.isVisibleTo(self) and self._z_offset.isChecked()

    @property
    def version_name(self) -> str:
        """The chosen version as shown in the dialog, for example ``ISO-2.0``."""
        return self._version.currentText()

    @property
    def encoding_name(self) -> str:
        """The chosen encoding as shown in the dialog, for example ``Binary``."""
        return self._encoding.currentText()

    @property
    def data_type_name(self) -> str:
        """The chosen data type as shown in the dialog, for example ``int32``."""
        return self._data_type.currentText()

    def _preset(self) -> None:
        """Select the options of the loaded file where it has the format that is saved."""
        file = self._file
        if isinstance(file, SdfFile) and not self._is_x3p:
            self._select(self._version, str(SdfDialect(file.header.dialect)))
            self._encoding.setCurrentIndex(0 if file.header.binary else 1)
            self._select(self._data_type, DataType(file.header.data_type).name.lower())
        elif isinstance(file, x3pio.X3pFile) and self._is_x3p:
            self._select(self._version, file.revision.value)
            self._encoding.setCurrentIndex(0 if file.storage is x3pio.DataStorage.BINARY else 1)
            self._select(self._data_type, file.header.z.data_type.name.lower())
        else:  # another format: the newest version in binary encoding, in the widest type
            self._select(self._version, _NEWEST_SDF if not self._is_x3p else _NEWEST_X3P)
            self._select(self._data_type, "float64" if self._is_x3p else "binary64")

    @staticmethod
    def _select(combo: QComboBox, text: str) -> None:
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _apply_last(self, version: str, encoding: str, data_type: str) -> None:
        self._select(self._version, version)
        self._select(self._encoding, encoding)
        self._select(self._data_type, data_type)

    def _problem(self) -> ExportProblem | None:
        """Return why the file cannot be saved with the chosen version and encoding."""
        dialect, file_format = self.dialect, self.file_format
        if not isinstance(dialect, SdfDialect) or self._sdf is None:
            return None
        return format_problem(self._sdf, dialect, cast(FileFormat, file_format))

    def _problem_text(self, problem: ExportProblem) -> str:
        dialect = self.version_name
        if problem.kind is ProblemKind.BINARY_LIMIT:
            return self.tr(
                "The binary encoding of {version} stores at most {limit} points per row."
            ).format(version=dialect, limit=problem.limit)
        return self.tr(
            'The trailer must be in the tagged "Name = Value" format for {version}.'
        ).format(version=dialect)

    def _consequences(self) -> str:
        """Return what saving in the chosen version loses or changes, or an empty string."""
        file, dialect = self._file, self.dialect
        if isinstance(file, SdfFile):
            return "" if not self._is_x3p else self.tr("The trailer is saved as the comment.")
        if not self._is_x3p:
            note = self.tr(
                "The metadata is saved in the trailer. Offsets, the rotation and the vendor "
                "extensions are not kept."
            )
            if self.apply_z_offset:
                note = self.tr(
                    "The metadata is saved in the trailer. The z offset is added to the "
                    "heights. The x and y offsets, the rotation and the vendor extensions are "
                    "not kept."
                )
            if len(file.layers) > 1:
                note += " " + self.tr("Only the current layer is saved.")
            return note
        if (
            isinstance(dialect, x3pio.Revision)
            and file.extensions
            and not extensions_fit(file, dialect)
        ):
            return self.tr("The vendor extensions cannot be kept in this version.")
        return ""

    def _update(self) -> None:
        """Disable the data types the version does not define, and say what is lost or wrong."""
        dialect = self.dialect
        first_valid: int | None = None
        if isinstance(dialect, SdfDialect):
            allowed = allowed_data_types(dialect)
            model = cast(QStandardItemModel, self._data_type.model())
            for index in range(self._data_type.count()):
                supported = DataType(self._data_type.itemData(index)) in allowed
                model.item(index).setEnabled(supported)
                if supported and first_valid is None:
                    first_valid = index
            if self.data_type not in allowed and first_valid is not None:
                self._data_type.setCurrentIndex(first_valid)

        problem = self._problem()
        self._note.setText(self._problem_text(problem) if problem else self._consequences())
        self._note.setVisible(bool(self._note.text()))
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(problem is None)
