# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A mouse drag in a view: the tool that the pressed button stands for and the movement."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt

from .camera import Tool

__all__ = ["Drag", "zoom_factor"]

_ZOOM_PER_PIXEL = 0.01


def zoom_factor(dy: float) -> float:
    """Return the zoom factor of a drag by ``dy`` pixels; dragging up zooms in."""
    return math.exp(-dy * _ZOOM_PER_PIXEL)


class Drag:
    """The state of a drag: the tool in use, and where the mouse was at the last event."""

    def __init__(self) -> None:
        """Start without a drag."""
        self.tool: Tool | None = None
        self._last = QPointF()

    def press(self, button: Qt.MouseButton, left: Tool, pos: QPointF) -> bool:
        """Start a drag with the tool of ``button`` at ``pos``.

        The middle button pans and the right button zooms; the left button uses ``left``.

        :returns: Whether ``button`` starts a drag.
        """
        if button == Qt.MouseButton.LeftButton:
            self.tool = left
        elif button == Qt.MouseButton.MiddleButton:
            self.tool = Tool.PAN
        elif button == Qt.MouseButton.RightButton:
            self.tool = Tool.ZOOM
        else:
            return False
        self._last = pos
        return True

    def move(self, pos: QPointF) -> tuple[float, float]:
        """Return how far the mouse moved in x and y since the last event, now at ``pos``."""
        dx, dy = pos.x() - self._last.x(), pos.y() - self._last.y()
        self._last = pos
        return dx, dy

    def release(self) -> None:
        """End the drag."""
        self.tool = None
