# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Color scale bar with unit-labeled ticks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QImage, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .surface import ValueRange
from .units import format_tick, nice_ticks, unit_for

__all__ = ["ColorBar"]

_MARGIN_TOP = 28
_MARGIN_BOTTOM = 14
_BAR_LEFT = 10
_BAR_WIDTH = 20
_TICK_LENGTH = 5
_SWATCH = 12


class ColorBar(QWidget):
    """Vertical color scale showing the height range and a legend for non-measured points."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty color bar."""
        super().__init__(parent)
        self._image: QImage | None = None
        self._range: ValueRange | None = None
        self._has_invalid = False
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        """Return a width fitting the tick labels and the legend in the current font."""
        metrics = self.fontMetrics()
        ticks = _BAR_WIDTH + _TICK_LENGTH + 3 + metrics.horizontalAdvance("-0.000")
        legend = _SWATCH + 5 + metrics.horizontalAdvance(self.tr("not measured"))
        return QSize(_BAR_LEFT + max(ticks, legend) + _BAR_LEFT, 0)

    def set_lut(self, lut: NDArray[np.uint8]) -> None:
        """Set the ``(256, 3)`` color lookup table."""
        top_first = np.ascontiguousarray(lut[::-1])
        self._image = QImage(
            top_first.tobytes(), 1, len(lut), 3, QImage.Format.Format_RGB888
        ).copy()
        self.update()

    def set_range(self, value_range: ValueRange | None, has_invalid: bool = False) -> None:
        """Set the displayed height range and whether non-measured points exist."""
        self._range = value_range
        self._has_invalid = has_invalid
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Draw the gradient, ticks and legend."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        text_color = self.palette().windowText().color()
        painter.setPen(text_color)

        legend_height = _SWATCH + 8 if self._has_invalid else 0
        bar_top = _MARGIN_TOP
        bar_height = self.height() - _MARGIN_TOP - _MARGIN_BOTTOM - legend_height
        if self._image is None or self._range is None or bar_height < 20:
            return
        bar = QRectF(_BAR_LEFT, bar_top, _BAR_WIDTH, bar_height)
        painter.drawImage(bar, self._image)
        painter.setPen(QPen(text_color, 1.0))
        painter.drawRect(bar)

        lo, hi = self._range.lo, self._range.hi
        factor, symbol = unit_for(hi - lo)
        painter.drawText(QRectF(0, 4, self.width(), 18), Qt.AlignmentFlag.AlignCenter, symbol)
        target = max(2, int(bar_height // 40))
        metrics = painter.fontMetrics()
        for tick in nice_ticks(lo, hi, target):
            y = bar_top + (hi - tick) / (hi - lo) * bar_height
            x = _BAR_LEFT + _BAR_WIDTH
            painter.drawLine(int(x), int(y), int(x + _TICK_LENGTH), int(y))
            painter.drawText(
                int(x + _TICK_LENGTH + 3),
                int(y + metrics.ascent() / 2 - 1),
                format_tick(tick, factor),
            )

        if self._has_invalid:
            swatch = QRectF(_BAR_LEFT, self.height() - _MARGIN_BOTTOM - _SWATCH, _SWATCH, _SWATCH)
            painter.fillRect(swatch, self.palette().window())
            painter.drawRect(swatch)
            painter.drawText(
                int(swatch.right() + 5), int(swatch.bottom() - 1), self.tr("not measured")
            )
