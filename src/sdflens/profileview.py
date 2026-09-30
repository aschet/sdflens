# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Profile view: the height along one profile (a row) or column of the surface."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QEvent, QLineF, QPointF, QRect, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPolygonF,
    QStandardItemModel,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .camera import Tool
from .surface import SurfaceModel
from .units import format_length, format_tick, nice_ticks, unit_for

__all__ = ["ProfileView", "envelope"]

_LEFT = 64
_TOP = 8
_RIGHT = 8
_BOTTOM = 28
_TICK_LENGTH = 4
_ZOOM_STEP = 1.2
_MIN_POINTS = 3  # the closest zoom shows this many points
_MARKER_SPACING = 6.0  # pixels between points from which the points get markers
_LINE_COLOR = QColor(47, 111, 176)


def envelope(
    values: NDArray[np.float64], first: float, last: float, columns: int
) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]:
    """Return the columns holding measured points, with the lowest and highest value of each.

    The index interval ``[first, last)`` is split into ``columns`` equal parts; a part covers the
    points whose index lies inside it. Parts without a measured point are left out.

    >>> columns, low, high = envelope(np.arange(10.0), 0.0, 10.0, 5)
    >>> columns.tolist(), low.tolist(), high.tolist()
    ([0, 1, 2, 3, 4], [0.0, 2.0, 4.0, 6.0, 8.0], [1.0, 3.0, 5.0, 7.0, 9.0])
    """
    edges = np.clip(np.ceil(np.linspace(first, last, columns + 1)), 0, len(values)).astype(np.intp)
    filled = np.flatnonzero(edges[1:] > edges[:-1])
    if filled.size == 0:
        return np.empty(0, np.intp), np.empty(0), np.empty(0)
    offsets = edges[:-1][filled] - edges[0]
    segment = values[edges[0] : edges[-1]]
    low = np.fmin.reduceat(segment, offsets)
    high = np.fmax.reduceat(segment, offsets)
    measured = np.isfinite(low)
    return filled[measured], low[measured], high[measured]


class _Plot(QWidget):
    """Height against position for one series; drag pans and the wheel zooms along x."""

    #: Index of the point under the cursor, ``-1`` when there is none.
    point_hovered = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._values: NDArray[np.float64] = np.empty(0)
        self._step = 1.0
        self._axis = "x"
        self._x0 = 0.0
        self._span = 1.0
        self._tool = Tool.PAN
        self._drag_tool: Tool | None = None
        self._last_pos = QPointF()
        self.setMouseTracking(True)

    def set_series(self, values: NDArray[np.float64], step: float, axis: str) -> None:
        """Show ``values`` spaced ``step`` meters apart; a same-sized series keeps the zoom."""
        same = len(values) == len(self._values) and step == self._step
        self._values, self._step, self._axis = values, step, axis
        if same:
            self.update()
        else:
            self.home()

    def value_at(self, index: int) -> float:
        """Return the value of point ``index``."""
        return float(self._values[index])

    def set_tool(self, tool: Tool) -> None:
        """Set the action of the left mouse button; rotating pans."""
        self._tool = Tool.ZOOM if tool is Tool.ZOOM else Tool.PAN

    def home(self) -> None:
        """Show all points."""
        low, high = self._full()
        self._x0, self._span = low, high - low
        self.update()

    def zoom_about_center(self, factor: float) -> None:
        """Zoom about the horizontal center of the plot."""
        self._zoom_at(factor, self._plot_rect().width() / 2.0)

    def _full(self) -> tuple[float, float]:
        """Return the metric interval of all points, each taking one step."""
        count = max(len(self._values), 1)
        return -0.5 * self._step, (count - 0.5) * self._step

    def _plot_rect(self) -> QRect:
        return QRect(_LEFT, _TOP, self.width() - _LEFT - _RIGHT, self.height() - _TOP - _BOTTOM)

    def _zoom_at(self, factor: float, plot_x: float) -> None:
        """Zoom by ``factor`` keeping the position at ``plot_x`` pixels from the plot's left."""
        width = max(self._plot_rect().width(), 1)
        low, high = self._full()
        anchor = self._x0 + plot_x / width * self._span
        span = min(max(self._span / factor, min(_MIN_POINTS * self._step, high - low)), high - low)
        self._x0 = min(max(anchor - plot_x / width * span, low), high - span)
        self._span = span
        self.update()

    def _index_at(self, pos: QPointF) -> int:
        plot = self._plot_rect()
        if not plot.contains(pos.toPoint()) or plot.width() < 1:
            return -1
        x = self._x0 + (pos.x() - plot.left()) / plot.width() * self._span
        index = round(x / self._step)
        return index if 0 <= index < len(self._values) else -1

    def paintEvent(self, event: QPaintEvent) -> None:
        """Draw the series and the rulers."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        plot = self._plot_rect()
        if len(self._values) == 0 or plot.width() < 10 or plot.height() < 10:
            return

        x_lo, x_hi = self._x0, self._x0 + self._span
        first, last = x_lo / self._step, x_hi / self._step
        start, stop = max(0, math.ceil(first)), min(len(self._values), math.floor(last) + 1)
        visible = self._values[start:stop]
        z_lo, z_hi = self._z_range(visible)

        painter.save()
        painter.setClipRect(plot)
        painter.setPen(_LINE_COLOR)
        if (last - first) / plot.width() >= 1.0:
            self._draw_envelope(painter, plot, first, last, z_lo, z_hi)
        else:
            self._draw_points(painter, plot, start, visible, z_lo, z_hi)
        painter.restore()
        self._draw_rulers(painter, plot, x_lo, x_hi, z_lo, z_hi)

    @staticmethod
    def _z_range(visible: NDArray[np.float64]) -> tuple[float, float]:
        """Return the height interval of the visible points with a margin; never empty."""
        low = float(np.fmin.reduce(visible)) if visible.size else math.nan
        high = float(np.fmax.reduce(visible)) if visible.size else math.nan
        if not math.isfinite(low):
            return 0.0, 1.0
        pad = 0.05 * (high - low) if high > low else (0.5 * abs(low) or 1e-9)
        return low - pad, high + pad

    def _draw_envelope(
        self, painter: QPainter, plot: QRect, first: float, last: float, z_lo: float, z_hi: float
    ) -> None:
        columns, low, high = envelope(self._values, first, last, plot.width())
        bottom, factor = plot.top() + plot.height(), plot.height() / (z_hi - z_lo)
        lines = [
            QLineF(
                plot.left() + column + 0.5,
                bottom - (top - z_lo) * factor,
                plot.left() + column + 0.5,
                bottom - (base - z_lo) * factor,
            )
            for column, base, top in zip(columns.tolist(), low.tolist(), high.tolist(), strict=True)
        ]
        painter.drawLines(lines)

    def _draw_points(
        self,
        painter: QPainter,
        plot: QRect,
        start: int,
        visible: NDArray[np.float64],
        z_lo: float,
        z_hi: float,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        indices = np.arange(start, start + len(visible))
        px = plot.left() + (indices * self._step - self._x0) / self._span * plot.width()
        py = plot.top() + plot.height() - (visible - z_lo) / (z_hi - z_lo) * plot.height()
        edges = np.diff(np.concatenate(([0], np.isfinite(visible), [0])).astype(int))
        for begin, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            points = [
                QPointF(x, y)
                for x, y in zip(px[begin:end].tolist(), py[begin:end].tolist(), strict=True)
            ]
            painter.drawPolyline(QPolygonF(points))
            if self._step / self._span * plot.width() >= _MARKER_SPACING:
                for point in points:
                    painter.drawEllipse(point, 2.0, 2.0)

    def _draw_rulers(
        self, painter: QPainter, plot: QRect, x_lo: float, x_hi: float, z_lo: float, z_hi: float
    ) -> None:
        painter.setPen(self.palette().windowText().color())
        painter.drawRect(plot)
        metrics = painter.fontMetrics()
        x_factor, x_symbol = unit_for(x_hi - x_lo)
        z_factor, z_symbol = unit_for(z_hi - z_lo)
        x_max = (len(self._values) - 1) * self._step
        bottom = plot.top() + plot.height()

        for tick in nice_ticks(max(x_lo, 0.0), min(x_hi, x_max), max(2, plot.width() // 90)):
            px = plot.left() + int((tick - x_lo) / (x_hi - x_lo) * plot.width())
            painter.drawLine(px, bottom, px, bottom + _TICK_LENGTH)
            label = format_tick(tick, x_factor)
            painter.drawText(
                px - metrics.horizontalAdvance(label) // 2,
                bottom + _TICK_LENGTH + metrics.ascent() + 1,
                label,
            )
        for tick in nice_ticks(z_lo, z_hi, max(2, plot.height() // 50)):
            py = bottom - int((tick - z_lo) / (z_hi - z_lo) * plot.height())
            painter.drawLine(plot.left() - _TICK_LENGTH, py, plot.left(), py)
            label = format_tick(tick, z_factor)
            painter.drawText(
                plot.left() - _TICK_LENGTH - 3 - metrics.horizontalAdvance(label),
                py + metrics.ascent() // 2,
                label,
            )
        caption = f"{self._axis} [{x_symbol}]"
        painter.drawText(
            plot.right() - metrics.horizontalAdvance(caption), bottom + _BOTTOM - 3, caption
        )
        painter.drawText(4, plot.top() + metrics.ascent(), f"z [{z_symbol}]")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Start a pan or zoom drag."""
        button = event.button()
        if button == Qt.MouseButton.LeftButton:
            self._drag_tool = self._tool
        elif button == Qt.MouseButton.MiddleButton:
            self._drag_tool = Tool.PAN
        elif button == Qt.MouseButton.RightButton:
            self._drag_tool = Tool.ZOOM
        else:
            return
        self._last_pos = event.position()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Pan or zoom while dragging, otherwise report the point under the cursor."""
        pos = event.position()
        if self._drag_tool is None:
            self.point_hovered.emit(self._index_at(pos))
            return
        plot = self._plot_rect()
        dx, dy = pos.x() - self._last_pos.x(), pos.y() - self._last_pos.y()
        self._last_pos = pos
        if self._drag_tool is Tool.PAN:
            low, high = self._full()
            shift = dx / max(plot.width(), 1) * self._span
            self._x0 = min(max(self._x0 - shift, low), high - self._span)
            self.update()
        else:
            self._zoom_at(math.exp(-dy * 0.01), pos.x() - plot.left())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """End the drag."""
        self._drag_tool = None

    def leaveEvent(self, event: QEvent) -> None:
        """Clear the hover readout."""
        self.point_hovered.emit(-1)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom about the cursor."""
        self._zoom_at(_ZOOM_STEP ** (event.angleDelta().y() / 120.0), event.position().x() - _LEFT)


class ProfileView(QWidget):
    """Height along a chosen profile of the surface, along x (a row) or along y (a column)."""

    #: Text describing the point under the cursor, empty when there is none.
    hovered = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty view."""
        super().__init__(parent)
        self._model: SurfaceModel | None = None
        self._values: NDArray[np.float64] = np.empty(0)
        self._step = 1.0

        self._direction = QComboBox(self)
        self._direction.addItems([self.tr("Along x"), self.tr("Along y")])
        self._label = QLabel(self)
        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._spin = QSpinBox(self)
        self._plot = _Plot(self)

        controls = QHBoxLayout()
        controls.setContentsMargins(6, 4, 6, 0)
        controls.addWidget(self._direction)
        controls.addWidget(self._label)
        controls.addWidget(self._slider, 1)
        controls.addWidget(self._spin)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self._plot, 1)

        self._direction.currentIndexChanged.connect(lambda _index: self._configure(1))
        self._slider.valueChanged.connect(self._spin.setValue)
        self._spin.valueChanged.connect(self._slider.setValue)
        self._spin.valueChanged.connect(lambda _value: self._show())
        self._plot.point_hovered.connect(self._on_point_hovered)
        self._configure(1)

    def set_model(self, model: SurfaceModel | None) -> None:
        """Show the first profile of ``model`` (or nothing), along x."""
        self._model = model
        self._direction.blockSignals(True)
        self._direction.setCurrentIndex(0)
        self._direction.blockSignals(False)
        can_turn = model is not None and model.num_profiles > 1
        cast(QStandardItemModel, self._direction.model()).item(1).setEnabled(can_turn)
        self._configure(1)

    def set_lut(self, lut: NDArray[np.uint8]) -> None:
        """Accept a color table; the profile is drawn in one color."""

    def set_tool(self, tool: Tool) -> None:
        """Set the action of the left mouse button; rotating pans."""
        self._plot.set_tool(tool)

    def home(self) -> None:
        """Show all points of the profile."""
        self._plot.home()

    def zoom_in(self) -> None:
        """Zoom in one step."""
        self._plot.zoom_about_center(_ZOOM_STEP)

    def zoom_out(self) -> None:
        """Zoom out one step."""
        self._plot.zoom_about_center(1.0 / _ZOOM_STEP)

    def grab_image(self) -> QImage:
        """Return the rendered plot."""
        return self._plot.grab().toImage()

    @property
    def _along_x(self) -> bool:
        return self._direction.currentIndex() == 0

    def _configure(self, index: int) -> None:
        """Set the selectable range for the current direction and select profile ``index``."""
        model = self._model
        count = 0
        if model is not None:
            count = model.num_profiles if self._along_x else model.num_points
        self._label.setText(self.tr("Profile:") if self._along_x else self.tr("Column:"))
        for widget in (self._slider, self._spin):
            widget.blockSignals(True)
            widget.setRange(1, max(count, 1))
            widget.setValue(min(max(index, 1), max(count, 1)))
            widget.blockSignals(False)
        self._spin.setSuffix(self.tr(" of {count}").format(count=count))
        self._spin.setEnabled(count > 1)
        self._slider.setEnabled(count > 1)
        self._show()

    def _show(self) -> None:
        model = self._model
        if model is None:
            self._values, self._step = np.empty(0), 1.0
        elif self._along_x:
            self._values, self._step = model.data[self._spin.value() - 1], model.x_scale
        else:
            column = np.ascontiguousarray(model.data[:, self._spin.value() - 1])
            self._values, self._step = column, model.pixel_size_y
        self._plot.set_series(self._values, self._step, "x" if self._along_x else "y")

    def _on_point_hovered(self, index: int) -> None:
        if index < 0 or self._model is None:
            self.hovered.emit("")
            return
        value = self._plot.value_at(index)
        z_text = format_length(value) if math.isfinite(value) else self.tr("not measured")
        position = format_length(index * self._step)
        if self._along_x:
            text = self.tr("x = {position}, z = {z} (profile {number}, point {point})")
        else:
            text = self.tr("y = {position}, z = {z} (column {number}, point {point})")
        self.hovered.emit(
            text.format(position=position, z=z_text, number=self._spin.value(), point=index + 1)
        )
