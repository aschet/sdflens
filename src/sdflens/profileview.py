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
    QPen,
    QPolygonF,
    QResizeEvent,
    QStandardItemModel,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .camera import Tool
from .icons import load_icon
from .overview import ProfileOverview
from .pointcloud import PointCloudModel
from .surface import SurfaceModel
from .units import format_length, format_tick, nice_ticks, unit_for

__all__ = ["ProfileView", "connect", "envelope", "position_envelope"]

_LEFT = 64
_TOP = 8
_RIGHT = 8
_BOTTOM = 28
_TICK_LENGTH = 4
_ZOOM_STEP = 1.2
_MIN_POINTS = 3  # the closest zoom shows this many points
_MARKER_RADIUS = 4.5  # of the marker on the point under the cursor
_LINE_COLOR = QColor(47, 111, 176)
_MARKER_COLOR = QColor(230, 120, 20)


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


def connect(
    columns: NDArray[np.intp], low: NDArray[np.float64], high: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``low`` and ``high`` extended so that each column touches the next adjacent one.

    >>> columns = np.array([0, 1, 3])
    >>> low, high = connect(columns, np.array([0.0, 4.0, 9.0]), np.array([1.0, 6.0, 9.0]))
    >>> low.tolist(), high.tolist()
    ([0.0, 4.0, 9.0], [4.0, 6.0, 9.0])
    """
    low, high = low.copy(), high.copy()
    adjacent = np.diff(columns) == 1
    above = adjacent & (high[:-1] < low[1:])
    below = adjacent & (low[:-1] > high[1:])
    high[:-1] = np.where(above, low[1:], high[:-1])
    low[:-1] = np.where(below, high[1:], low[:-1])
    return low, high


def length_along(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the distance of each point of a profile from the first, along the profile.

    ``points`` are the ``(N, 3)`` points in the order of the profile. The distance is measured in
    the plane of ``x`` and ``y`` and summed over the segments, so a diagonal profile has its true
    length and a profile along an axis starts at 0.

    >>> length_along(np.array([[0.0, 0.0, 1.0], [3.0, 4.0, 1.0], [6.0, 8.0, 1.0]])).tolist()
    [0.0, 5.0, 10.0]
    """
    steps = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    return np.concatenate(([0.0], np.cumsum(steps)))


def position_envelope(
    values: NDArray[np.float64],
    positions: NDArray[np.float64],
    first: float,
    last: float,
    columns: int,
) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.float64]]:
    """Return the columns holding measured points of a profile with irregular positions.

    Like :func:`envelope`, but the interval ``[first, last)`` of positions is split into
    ``columns`` equal parts, and a part covers the points whose position lies inside it. The
    positions have to be in increasing order.

    >>> columns, low, high = position_envelope(
    ...     np.array([1.0, 2.0, 5.0, 7.0]), np.array([0.0, 1.0, 9.0, 9.5]), 0.0, 10.0, 2
    ... )
    >>> columns.tolist(), low.tolist(), high.tolist()
    ([0, 1], [1.0, 5.0], [2.0, 7.0])
    """
    start, stop = np.searchsorted(positions, [first, last])
    slot = np.floor((positions[start:stop] - first) / (last - first) * columns).astype(np.intp)
    segment = values[start:stop]
    keep = (slot >= 0) & (slot < columns) & np.isfinite(segment)
    slot, segment = slot[keep], segment[keep]
    if slot.size == 0:
        return np.empty(0, np.intp), np.empty(0), np.empty(0)
    begins = np.flatnonzero(np.diff(slot, prepend=-1))
    return slot[begins], np.fmin.reduceat(segment, begins), np.fmax.reduceat(segment, begins)


class _Plot(QWidget):
    """Height against position for one series; drag pans and the wheel zooms along x."""

    #: Index of the point under the cursor, ``-1`` when there is none.
    point_hovered = Signal(int)
    #: The plot has a new size.
    resized = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._values: NDArray[np.float64] = np.empty(0)
        self._step = 1.0
        #: The positions of the points of an irregular profile, in increasing order.
        self._positions: NDArray[np.float64] | None = None
        self._x0 = 0.0
        self._span = 1.0
        self._tool = Tool.PAN
        self._drag_tool: Tool | None = None
        self._last_pos = QPointF()
        self._hover = -1  # index of the point under the cursor
        self.setMouseTracking(True)

    def set_series(
        self,
        values: NDArray[np.float64],
        step: float,
        positions: NDArray[np.float64] | None = None,
    ) -> None:
        """Show ``values`` spaced ``step`` meters apart, or at ``positions`` if they are given.

        A series of the same size keeps the zoom.
        """
        same = (
            len(values) == len(self._values)
            and step == self._step
            and (positions is None) == (self._positions is None)
        )
        self._values, self._step, self._positions = values, step, positions
        self._hover = -1
        if same:
            self.update()
        else:
            self.home()

    def value_at(self, index: int) -> float:
        """Return the value of point ``index``."""
        return float(self._values[index])

    def position_at(self, index: int) -> float:
        """Return the position along the profile of point ``index`` in meters."""
        return float(self._positions[index]) if self._positions is not None else index * self._step

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
        if self._positions is not None and len(self._positions) > 1:
            return self._positions[0] - 0.5 * self._step, self._positions[-1] + 0.5 * self._step
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
        if self._positions is not None:
            # The nearest point of an irregular profile.
            above = int(np.searchsorted(self._positions, x))
            below = above - 1
            nearer_below = below >= 0 and (
                above >= len(self._values)
                or x - self._positions[below] <= self._positions[above] - x
            )
            index = below if nearer_below else above
        else:
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
        if self._positions is not None:
            # The points in view, with their neighbours so that a line reaches the edge.
            start = max(0, int(np.searchsorted(self._positions, x_lo)) - 1)
            stop = min(len(self._values), int(np.searchsorted(self._positions, x_hi)) + 1)
            per_pixel = (stop - start) / plot.width()
            first, last = x_lo, x_hi
        else:
            first, last = x_lo / self._step, x_hi / self._step
            start, stop = max(0, math.ceil(first)), min(len(self._values), math.floor(last) + 1)
            per_pixel = (last - first) / plot.width()
        visible = self._values[start:stop]
        z_lo, z_hi = self._z_range(visible)

        painter.save()
        painter.setClipRect(plot)
        # The square cap draws a column that has a single point as a dot of one pixel.
        painter.setPen(QPen(_LINE_COLOR, 1.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap))
        if per_pixel >= 1.0:
            self._draw_envelope(painter, plot, first, last, z_lo, z_hi)
        else:
            self._draw_points(painter, plot, start, visible, z_lo, z_hi)
        self._draw_hover(painter, plot, z_lo, z_hi)
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
        if self._positions is not None:
            columns, low, high = position_envelope(
                self._values, self._positions, first, last, plot.width()
            )
        else:
            columns, low, high = envelope(self._values, first, last, plot.width())
        low, high = connect(columns, low, high)
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
        if self._positions is not None:
            where = self._positions[start : start + len(visible)]
        else:
            where = np.arange(start, start + len(visible), dtype=np.float64) * self._step
        px = plot.left() + (where - self._x0) / self._span * plot.width()
        py = plot.top() + plot.height() - (visible - z_lo) / (z_hi - z_lo) * plot.height()
        edges = np.diff(np.concatenate(([0], np.isfinite(visible), [0])).astype(int))
        for begin, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            points = [
                QPointF(x, y)
                for x, y in zip(px[begin:end].tolist(), py[begin:end].tolist(), strict=True)
            ]
            painter.drawPolyline(QPolygonF(points))

    def _draw_hover(self, painter: QPainter, plot: QRect, z_lo: float, z_hi: float) -> None:
        """Mark the point under the cursor, if it is measured."""
        if not 0 <= self._hover < len(self._values) or not math.isfinite(self._values[self._hover]):
            return
        x = plot.left() + (self.position_at(self._hover) - self._x0) / self._span * plot.width()
        y = plot.bottom() + 1 - (self._values[self._hover] - z_lo) / (z_hi - z_lo) * plot.height()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(self.palette().window().color(), 1.5))
        painter.setBrush(_MARKER_COLOR)
        painter.drawEllipse(QPointF(x, y), _MARKER_RADIUS, _MARKER_RADIUS)

    def _draw_rulers(
        self, painter: QPainter, plot: QRect, x_lo: float, x_hi: float, z_lo: float, z_hi: float
    ) -> None:
        painter.setPen(self.palette().windowText().color())
        painter.drawRect(plot)
        metrics = painter.fontMetrics()
        x_factor, x_symbol = unit_for(x_hi - x_lo)
        z_factor, z_symbol = unit_for(z_hi - z_lo)
        if self._positions is not None:
            x_min, x_max = float(self._positions[0]), float(self._positions[-1])
        else:
            x_min, x_max = 0.0, (len(self._values) - 1) * self._step
        bottom = plot.top() + plot.height()

        for tick in nice_ticks(max(x_lo, x_min), min(x_hi, x_max), max(2, plot.width() // 90 + 1)):
            px = plot.left() + int((tick - x_lo) / (x_hi - x_lo) * plot.width())
            painter.drawLine(px, bottom, px, bottom + _TICK_LENGTH)
            label = format_tick(tick, x_factor)
            painter.drawText(
                px - metrics.horizontalAdvance(label) // 2,
                bottom + _TICK_LENGTH + metrics.ascent() + 1,
                label,
            )
        for tick in nice_ticks(z_lo, z_hi, max(2, plot.height() // 50 + 1)):
            py = bottom - int((tick - z_lo) / (z_hi - z_lo) * plot.height())
            painter.drawLine(plot.left() - _TICK_LENGTH, py, plot.left(), py)
            label = format_tick(tick, z_factor)
            painter.drawText(
                plot.left() - _TICK_LENGTH - 3 - metrics.horizontalAdvance(label),
                py + metrics.ascent() // 2,
                label,
            )
        caption = f"l [{x_symbol}]"
        painter.drawText(
            plot.right() - metrics.horizontalAdvance(caption), bottom + _BOTTOM - 3, caption
        )
        painter.drawText(4, plot.top() + metrics.ascent(), f"z [{z_symbol}]")

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Tell that the size changed."""
        super().resizeEvent(event)
        self.resized.emit()

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
        self._set_hover(-1)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Pan or zoom while dragging, otherwise report the point under the cursor."""
        pos = event.position()
        if self._drag_tool is None:
            self._set_hover(self._index_at(pos))
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
        """Clear the hover marker and readout."""
        self._set_hover(-1)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom about the cursor."""
        self._zoom_at(_ZOOM_STEP ** (event.angleDelta().y() / 120.0), event.position().x() - _LEFT)
        self._set_hover(self._index_at(event.position()))

    def _set_hover(self, index: int) -> None:
        if index != self._hover:
            self._hover = index
            self.update()
        self.point_hovered.emit(index)


class ProfileView(QWidget):
    """Height along a chosen profile of the surface, along x (a row) or along y (a column)."""

    #: Text describing the point under the cursor, empty when there is none.
    hovered = Signal(str)
    #: The button to save the shown profile was clicked.
    save_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty view."""
        super().__init__(parent)
        self._model: SurfaceModel | None = None
        #: The positions, the heights and the axis of an irregular profile, in place of a model.
        self._irregular: tuple[NDArray[np.float64], NDArray[np.float64], str] | None = None
        self._values: NDArray[np.float64] = np.empty(0)
        self._step = 1.0

        self._direction = QComboBox(self)
        self._direction.addItems([self.tr("Along x"), self.tr("Along y")])
        self._label = QLabel(self)
        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._spin = QSpinBox(self)
        self._save = QPushButton(load_icon("save-as"), "", self)
        self._save.setToolTip(self.tr("Save profile as..."))
        self._plot = _Plot(self)
        self._overview = ProfileOverview(self._plot)

        controls = QHBoxLayout()
        controls.setContentsMargins(6, 4, 6, 0)
        controls.addWidget(self._direction)
        controls.addWidget(self._label)
        controls.addWidget(self._slider, 1)
        controls.addWidget(self._spin)
        controls.addWidget(self._save)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self._plot, 1)

        self._direction.currentIndexChanged.connect(lambda _index: self._configure(1))
        self._slider.valueChanged.connect(self._spin.setValue)
        self._spin.valueChanged.connect(self._slider.setValue)
        self._spin.valueChanged.connect(lambda _value: self._show())
        self._plot.point_hovered.connect(self._on_point_hovered)
        self._plot.resized.connect(self._overview.reposition)
        self._spin.valueChanged.connect(lambda _value: self._show_overview())
        self._direction.currentIndexChanged.connect(lambda _index: self._show_overview())
        self._slider.sliderPressed.connect(self._hold_overview)
        self._slider.sliderReleased.connect(self._overview.release)
        self._save.clicked.connect(self.save_requested)
        self._configure(1)

    def set_model(
        self, model: SurfaceModel | PointCloudModel | None, *, keep_selection: bool = False
    ) -> None:
        """Show the first profile of ``model`` (or nothing), along x.

        A profile whose points are not evenly spaced is shown at their positions; other points
        are not shown. With ``keep_selection``, ``model`` has the shape of the one before, and
        the direction and the number of the profile stay.
        """
        self._irregular = None
        if isinstance(model, PointCloudModel):
            if model.profile_axis is not None:
                column = 0 if model.profile_axis == "x" else 1
                order = np.argsort(model.points[:, column], kind="stable")
                along = model.points[order]
                self._irregular = (length_along(along), along[:, 2], model.profile_axis)
            model = None
        self._model = model
        self._overview.set_model(model)
        if keep_selection:
            self._show()
            return
        # A single column is a profile along y; a row cannot be chosen from it.
        along_y = model is not None and model.num_points < 2 <= model.num_profiles
        self._direction.blockSignals(True)
        self._direction.setCurrentIndex(1 if along_y else 0)
        self._direction.blockSignals(False)
        items = cast(QStandardItemModel, self._direction.model())
        items.item(0).setEnabled(not along_y)
        items.item(1).setEnabled(model is not None and model.num_profiles > 1)
        self._direction.setEnabled(self._irregular is None)
        self._configure(1)

    def set_lut(self, lut: NDArray[np.uint8]) -> None:
        """Set the color table of the overview map; the profile is drawn in one color."""
        self._overview.set_lut(lut)

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

    def current_profile(self) -> tuple[NDArray[np.float64], float, bool, int]:
        """Return the shown profile: heights, step, whether along x, and its number from 1."""
        return self._values.copy(), self._step, self._along_x, self._spin.value()

    @property
    def _along_x(self) -> bool:
        if self._irregular is not None:
            return self._irregular[2] == "x"
        return self._direction.currentIndex() == 0

    def _configure(self, index: int) -> None:
        """Set the selectable range for the current direction and select profile ``index``."""
        model = self._model
        count = 0
        if model is not None:
            count = model.num_profiles if self._along_x else model.num_points
        elif self._irregular is not None:
            count = 1
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

    def _hold_overview(self) -> None:
        self._overview.hold()
        self._show_overview()

    def _show_overview(self) -> None:
        """Show where the profile lies on the surface."""
        self._overview.show_profile(self._spin.value() - 1, along_x=self._along_x)

    def _show(self) -> None:
        model = self._model
        if self._irregular is not None:
            positions, self._values, _axis = self._irregular
            # The mean distance between the points stands for the step.
            self._step = float(positions[-1] - positions[0]) / max(len(positions) - 1, 1) or 1.0
            self._save.setEnabled(False)  # an irregular profile cannot be saved as a profile
            self._plot.set_series(self._values, self._step, positions)
            return
        if model is None:
            self._values, self._step = np.empty(0), 1.0
        elif self._along_x:
            self._values, self._step = model.data[self._spin.value() - 1], model.x_scale
        else:
            column = np.ascontiguousarray(model.data[:, self._spin.value() - 1])
            self._values, self._step = column, model.pixel_size_y
        self._save.setEnabled(model is not None)
        self._plot.set_series(self._values, self._step)

    def _on_point_hovered(self, index: int) -> None:
        if index < 0 or (self._model is None and self._irregular is None):
            self.hovered.emit("")
            return
        value = self._plot.value_at(index)
        z_text = format_length(value) if math.isfinite(value) else self.tr("not measured")
        position = format_length(self._plot.position_at(index))
        if self._irregular is not None:
            text = self.tr("l = {position}, z = {z}")
        elif self._along_x:
            text = self.tr("l = {position}, z = {z} (profile {number}, point {point})")
        else:
            text = self.tr("l = {position}, z = {z} (column {number}, point {point})")
        self.hovered.emit(
            text.format(position=position, z=z_text, number=self._spin.value(), point=index + 1)
        )
