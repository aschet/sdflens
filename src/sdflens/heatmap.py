# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""2D depth view: colormapped image of the height data with rulers and pan/zoom."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QEvent, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from .camera import Tool
from .colormap import apply_colormap
from .surface import SurfaceModel
from .units import format_length, format_tick, nice_ticks, unit_for
from .view2d import Bounds, ViewTransform2D

__all__ = ["HeatmapView"]

_LEFT = 64
_TOP = 8
_RIGHT = 8
_BOTTOM = 28
_ZOOM_STEP = 1.2
_TICK_LENGTH = 4
_CHUNK_POINTS = 4_000_000  # points averaged at once, which bounds the temporary memory


def _block_size(device_pixels_per_point: float, count: int) -> int:
    """Return the power of two of points to average along an axis of ``count`` points."""
    if device_pixels_per_point >= 0.5:
        return 1
    return min(1 << int(math.log2(1.0 / device_pixels_per_point)), 1 << (count.bit_length() - 1))


def _reduce(data: NDArray[np.float64], block_y: int, block_x: int) -> NDArray[np.float64]:
    """Average blocks of measured points; a block without any measured point is NaN."""
    rows, cols = data.shape
    out_rows, out_cols = -(-rows // block_y), -(-cols // block_x)
    result = np.empty((out_rows, out_cols))
    per_chunk = max(1, _CHUNK_POINTS // (block_y * out_cols * block_x))
    for first in range(0, out_rows, per_chunk):
        slab = data[first * block_y : (first + per_chunk) * block_y]
        padded = np.full((-(-slab.shape[0] // block_y) * block_y, out_cols * block_x), np.nan)
        padded[: slab.shape[0], :cols] = slab
        blocks = padded.reshape(-1, block_y, out_cols, block_x)
        valid = np.isfinite(blocks)
        counts = valid.sum(axis=(1, 3))
        sums = np.where(valid, blocks, 0.0).sum(axis=(1, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            result[first : first + len(counts)] = np.where(counts > 0, sums / counts, np.nan)
    return result


class HeatmapView(QWidget):
    """2D colormapped view; left/middle drag pans, wheel zooms about the cursor."""

    #: Text describing the point under the cursor, empty when outside the data.
    hovered = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty view."""
        super().__init__(parent)
        self._model: SurfaceModel | None = None
        self._lut: NDArray[np.uint8] | None = None
        self._image: QImage | None = None
        self._levels: dict[tuple[int, int], QImage] = {}
        self._transform = ViewTransform2D()
        self._fitted = True
        self._tool = Tool.ROTATE
        self._drag_tool: Tool | None = None
        self._last_pos = QPointF()
        self.setMouseTracking(True)

    def set_model(self, model: SurfaceModel | None) -> None:
        """Show ``model`` (or nothing) fitted into the view."""
        self._model = model
        self._rebuild_image()
        self.home()

    def set_lut(self, lut: NDArray[np.uint8]) -> None:
        """Set the ``(256, 3)`` color lookup table."""
        self._lut = lut
        self._rebuild_image()
        self.update()

    def set_tool(self, tool: Tool) -> None:
        """Set the action of the left mouse button (rotate is treated as pan in 2D)."""
        self._tool = tool

    def home(self) -> None:
        """Fit the whole image into the view."""
        self._fitted = True
        self._fit()
        self.update()

    def zoom_in(self) -> None:
        """Zoom in one step about the view center."""
        self._zoom_about_center(_ZOOM_STEP)

    def zoom_out(self) -> None:
        """Zoom out one step about the view center."""
        self._zoom_about_center(1.0 / _ZOOM_STEP)

    def grab_image(self) -> QImage:
        """Return the rendered view."""
        return self.grab().toImage()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Draw the image and the rulers."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        plot = self._plot_rect()
        model = self._model
        if model is None or self._image is None or plot.width() < 10 or plot.height() < 10:
            return

        painter.save()
        painter.setClipRect(plot)
        painter.translate(plot.topLeft())
        bounds = self._bounds(model)
        left, top = self._transform.world_to_screen(bounds.x0, bounds.y1)
        right, bottom = self._transform.world_to_screen(bounds.x1, bounds.y0)
        # Points smaller than half a device pixel are averaged, separately along each axis.
        point_x = self._transform.scale * model.x_scale * self.devicePixelRatioF()
        point_y = self._transform.scale * model.pixel_size_y * self.devicePixelRatioF()
        block_x = _block_size(point_x, model.num_points)
        block_y = _block_size(point_y, model.num_profiles)
        image = self._level(block_y, block_x)
        source = QRectF(0.0, 0.0, model.num_points / block_x, model.num_profiles / block_y)
        smooth = point_x * block_x < 1.0 and point_y * block_y < 1.0
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)
        painter.drawImage(QRectF(left, top, right - left, bottom - top), image, source)
        painter.restore()

        self._draw_rulers(painter, plot)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Keep the image fitted after a resize unless the user has navigated."""
        if self._fitted:
            self._fit()
        else:
            plot = self._plot_rect()
            self._transform.viewport_width = max(plot.width(), 1)
            self._transform.viewport_height = max(plot.height(), 1)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Start a pan or zoom drag."""
        button = event.button()
        if button == Qt.MouseButton.LeftButton:
            self._drag_tool = Tool.ZOOM if self._tool is Tool.ZOOM else Tool.PAN
        elif button == Qt.MouseButton.MiddleButton:
            self._drag_tool = Tool.PAN
        elif button == Qt.MouseButton.RightButton:
            self._drag_tool = Tool.ZOOM
        else:
            return
        self._last_pos = event.position()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Pan/zoom while dragging, otherwise report the value under the cursor."""
        pos = event.position()
        if self._drag_tool is None:
            self.hovered.emit(self._describe(pos))
            return
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self._last_pos = pos
        self._fitted = False
        if self._drag_tool is Tool.PAN:
            self._transform.pan(dx, dy)
        else:
            origin = self._plot_rect().topLeft()
            self._transform.zoom_at(
                math.exp(-dy * 0.01),
                self._last_pos.x() - origin.x(),
                self._last_pos.y() - origin.y(),
            )
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """End the drag."""
        self._drag_tool = None

    def leaveEvent(self, event: QEvent) -> None:
        """Clear the hover readout."""
        self.hovered.emit("")

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom about the cursor."""
        origin = self._plot_rect().topLeft()
        pos = event.position()
        self._fitted = False
        self._transform.zoom_at(
            _ZOOM_STEP ** (event.angleDelta().y() / 120.0),
            pos.x() - origin.x(),
            pos.y() - origin.y(),
        )
        self.update()

    def _plot_rect(self) -> QRect:
        return QRect(_LEFT, _TOP, self.width() - _LEFT - _RIGHT, self.height() - _TOP - _BOTTOM)

    @staticmethod
    def _bounds(model: SurfaceModel) -> Bounds:
        """Metric rectangle covered by the pixels (each centered on its coordinate)."""
        xs = model.x_scale
        ys = model.pixel_size_y
        return Bounds(
            -0.5 * xs, -0.5 * ys, (model.num_points - 0.5) * xs, (model.num_profiles - 0.5) * ys
        )

    def _fit(self) -> None:
        plot = self._plot_rect()
        if self._model is None:
            return
        self._transform.fit(plot.width(), plot.height(), self._bounds(self._model))

    def _zoom_about_center(self, factor: float) -> None:
        plot = self._plot_rect()
        self._fitted = False
        self._transform.zoom_at(factor, plot.width() / 2.0, plot.height() / 2.0)
        self.update()

    def _rebuild_image(self) -> None:
        model = self._model
        self._levels = {}
        if model is None or self._lut is None:
            self._image = None
            return
        self._image = self._level(1, 1)

    def _level(self, block_y: int, block_x: int) -> QImage:
        """Return the colormapped image of the data averaged in blocks, built on first use."""
        cached = self._levels.get((block_y, block_x))
        if cached is not None:
            return cached
        model, lut = self._model, self._lut
        if model is None or lut is None:
            return QImage()
        # y increases with the row index, so row 0 belongs at the bottom of the image.
        data = model.data[::-1]
        if block_y > 1 or block_x > 1:
            data = _reduce(data, block_y, block_x)
        rgba = apply_colormap(data, lut, model.value_range.lo, model.value_range.hi)
        height, width = rgba.shape[:2]
        image = QImage(
            rgba.tobytes(), width, height, width * 4, QImage.Format.Format_RGBA8888
        ).copy()
        self._levels[(block_y, block_x)] = image
        return image

    def _describe(self, pos: QPointF) -> str:
        model = self._model
        if model is None:
            return ""
        origin = self._plot_rect().topLeft()
        x, y = self._transform.screen_to_world(pos.x() - origin.x(), pos.y() - origin.y())
        col = math.floor(x / model.x_scale + 0.5)
        row = math.floor(y / model.pixel_size_y + 0.5)
        if not (0 <= col < model.num_points and 0 <= row < model.num_profiles):
            return ""
        value = float(model.data[row, col])
        z_text = format_length(value) if math.isfinite(value) else self.tr("not measured")
        return self.tr("x = {x}, y = {y}, z = {z} (profile {profile}, point {point})").format(
            x=format_length(col * model.x_scale),
            y=format_length(row * model.y_scale),
            z=z_text,
            profile=row + 1,
            point=col + 1,
        )

    def _draw_rulers(self, painter: QPainter, plot: QRect) -> None:
        transform = self._transform
        painter.setPen(self.palette().windowText().color())
        painter.drawRect(plot)
        metrics = painter.fontMetrics()

        x_lo, y_hi = transform.screen_to_world(0.0, 0.0)
        x_hi, y_lo = transform.screen_to_world(plot.width(), plot.height())
        x_factor, x_symbol = unit_for(x_hi - x_lo)
        y_factor, y_symbol = unit_for(y_hi - y_lo)
        model = self._model
        x_min = 0.0
        x_max = (model.num_points - 1) * model.x_scale if model else 0.0
        y_min = 0.0
        y_max = (model.num_profiles - 1) * model.pixel_size_y if model else 0.0

        for tick in nice_ticks(max(x_lo, x_min), min(x_hi, x_max), max(2, plot.width() // 90)):
            sx, _ = transform.world_to_screen(tick, 0.0)
            px = plot.left() + int(sx)
            painter.drawLine(px, plot.bottom(), px, plot.bottom() + _TICK_LENGTH)
            label = format_tick(tick, x_factor)
            painter.drawText(
                px - metrics.horizontalAdvance(label) // 2,
                plot.bottom() + _TICK_LENGTH + metrics.ascent() + 1,
                label,
            )
        for tick in nice_ticks(max(y_lo, y_min), min(y_hi, y_max), max(2, plot.height() // 50)):
            _, sy = transform.world_to_screen(0.0, tick)
            py = plot.top() + int(sy)
            painter.drawLine(plot.left() - _TICK_LENGTH, py, plot.left(), py)
            label = format_tick(tick, y_factor)
            painter.drawText(
                plot.left() - _TICK_LENGTH - 3 - metrics.horizontalAdvance(label),
                py + metrics.ascent() // 2,
                label,
            )
        painter.drawText(
            plot.right() - metrics.horizontalAdvance(f"x [{x_symbol}]"),
            plot.bottom() + _BOTTOM - 3,
            f"x [{x_symbol}]",
        )
        painter.drawText(4, plot.top() + metrics.ascent(), f"y [{y_symbol}]")
