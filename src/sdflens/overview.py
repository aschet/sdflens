# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A small map of the surface that shows where the profile lies while its number changes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QLineF, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from .colormap import apply_colormap
from .surface import SurfaceModel
from .units import format_length

__all__ = ["ProfileOverview", "map_size"]

_BOX = (160, 120)  # the largest size of the map in pixels
_MAX_ELONGATION = 4.0  # a longer surface is shown less long, so that the line can be seen
_CAPTION_WIDTH = 76  # the width that the caption needs, so that the widget keeps its width
_PADDING = 6
_HIDE_DELAY = 1200  # milliseconds after the last change
_LINE_COLOR = QColor(255, 154, 31)


def map_size(
    columns: int, rows: int, x_scale: float, y_scale: float, box: tuple[int, int] = _BOX
) -> tuple[int, int]:
    """Return the width and the height in pixels of the map of a surface.

    The map has the aspect ratio of the surface, limited to ``_MAX_ELONGATION`` to one.

    >>> map_size(400, 300, 1e-6, 1e-6)
    (160, 120)
    >>> map_size(400, 72, 5e-6, 5e-3)
    (30, 120)
    """
    aspect = (columns * x_scale) / (rows * y_scale)
    aspect = min(max(aspect, 1.0 / _MAX_ELONGATION), _MAX_ELONGATION)
    width = max(round(min(box[0], box[1] * aspect)), 1)
    return width, max(round(width / aspect), 1)


def _reduced(data: NDArray[np.float64], width: int, height: int) -> NDArray[np.float64]:
    """Return the averages of blocks of ``data``, about as many as ``width`` by ``height``."""
    rows, columns = data.shape
    block_y, block_x = max(rows // height, 1), max(columns // width, 1)
    out_rows, out_columns = -(-rows // block_y), -(-columns // block_x)
    padded = np.full((out_rows * block_y, out_columns * block_x), np.nan)
    padded[:rows, :columns] = data
    blocks = padded.reshape(out_rows, block_y, out_columns, block_x)
    valid = np.isfinite(blocks)
    counts = valid.sum(axis=(1, 3))
    sums = np.where(valid, blocks, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


class ProfileOverview(QWidget):
    """Shows the surface in the colors of the colormap and a line where the profile lies.

    The map appears when :meth:`show_profile` is called, stays while it is held, and goes a
    moment after the last call. It takes no mouse input.
    """

    def __init__(self, parent: QWidget) -> None:
        """Create the hidden map over ``parent``."""
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()
        self._image = QImage()
        self._model: SurfaceModel | None = None
        self._lut: NDArray[np.uint8] | None = None
        self._along_x = True
        self._index = 0
        self._caption = ""
        self._held = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_HIDE_DELAY)
        self._timer.timeout.connect(self.hide)

    def set_model(self, model: SurfaceModel | None) -> None:
        """Set the surface to show; ``None`` hides the map."""
        self._model = model
        self._image = QImage()
        self.hide()

    def set_lut(self, lut: NDArray[np.uint8]) -> None:
        """Set the ``(256, 3)`` color lookup table."""
        self._lut = lut
        self._image = QImage()

    def hold(self) -> None:
        """Keep the map shown, as long as the slider is being dragged."""
        self._held = True
        self._timer.stop()

    def release(self) -> None:
        """Let the map go after its delay."""
        self._held = False
        if not self.isHidden():
            self._timer.start()

    def show_profile(self, index: int, *, along_x: bool) -> None:
        """Mark profile number ``index``, counted from 0, along x (a row) or y (a column)."""
        model = self._model
        if model is None or self._lut is None or model.num_profiles < 2 or model.num_points < 2:
            return
        width, height = map_size(model.num_points, model.num_profiles, model.x_scale, model.y_scale)
        if self._image.isNull() or self._image.size().width() == 0:
            self._build_image(model, width, height)
        self._along_x, self._index = along_x, index
        # A row lies at a position along y, a column at one along x.
        name, step = ("y", model.y_scale) if along_x else ("x", model.x_scale)
        self._caption = f"{name} = {format_length(index * step)}"
        self.setFixedSize(
            max(width, _CAPTION_WIDTH) + 2 * _PADDING,
            height + 2 * _PADDING + self._caption_height(),
        )
        self.reposition()
        self.show()
        self.raise_()
        self.update()
        if self._held:
            self._timer.stop()
        else:
            self._timer.start()

    def _caption_height(self) -> int:
        return self.fontMetrics().height() + 2

    def reposition(self) -> None:
        """Put the map at the top right of its parent."""
        parent = self.parentWidget()
        if parent is not None:
            self.move(parent.width() - self.width() - 12, 12)

    def _build_image(self, model: SurfaceModel, width: int, height: int) -> None:
        if self._lut is None:
            return
        # y increases with the row index, so row 0 belongs at the bottom of the image.
        data = _reduced(model.data[::-1], width, height)
        rgba = apply_colormap(data, self._lut, model.value_range.lo, model.value_range.hi)
        rows, columns = rgba.shape[:2]
        self._image = QImage(
            rgba.tobytes(), columns, rows, columns * 4, QImage.Format.Format_RGBA8888
        ).copy()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Draw the map with the profile and the caption."""
        model = self._model
        if model is None or self._image.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        window = self.palette().window().color()
        window.setAlpha(235)
        painter.setPen(QPen(self.palette().mid().color(), 1.0))
        painter.setBrush(window)
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)

        width, height = map_size(model.num_points, model.num_profiles, model.x_scale, model.y_scale)
        left = (self.width() - width) // 2
        target = QRect(left, _PADDING, width, height)
        painter.drawImage(target, self._image)
        if self._along_x:
            y = target.bottom() + 1 - (self._index + 0.5) / model.num_profiles * height
            line = QLineF(target.left(), y, target.right() + 1, y)
        else:
            x = target.left() + (self._index + 0.5) / model.num_points * width
            line = QLineF(x, target.top(), x, target.bottom() + 1)
        painter.setPen(
            QPen(QColor(0, 0, 0, 140), 3.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap)
        )
        painter.drawLine(line)
        painter.setPen(QPen(_LINE_COLOR, 1.75, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
        painter.drawLine(line)
        painter.setPen(self.palette().windowText().color())
        caption = QRect(0, height + _PADDING + 2, self.width(), self._caption_height())
        painter.drawText(caption, Qt.AlignmentFlag.AlignCenter, self._caption)
