# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pan/zoom transform between metric surface coordinates and the 2D viewport."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Bounds", "ViewTransform2D"]

_MIN_ZOOM = 0.05
_MAX_ZOOM = 5000.0


@dataclass(frozen=True)
class Bounds:
    """Rectangle in meters, with y pointing up."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        """Extent along x."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Extent along y."""
        return self.y1 - self.y0


class ViewTransform2D:
    """Maps meters (y up) to viewport pixels (y down) with pan and zoom about a point.

    >>> t = ViewTransform2D()
    >>> t.fit(200, 100, Bounds(0.0, 0.0, 2.0, 1.0), margin=1.0)
    >>> t.world_to_screen(1.0, 0.5)
    (100.0, 50.0)
    >>> t.screen_to_world(100.0, 50.0)
    (1.0, 0.5)
    """

    def __init__(self) -> None:
        """Create an identity-like transform; call :meth:`fit` before use."""
        self.scale = 1.0  # pixels per meter
        self.center_x = 0.0
        self.center_y = 0.0
        self.viewport_width = 1.0
        self.viewport_height = 1.0
        self._fit_scale = 1.0

    def fit(self, width: float, height: float, bounds: Bounds, margin: float = 0.95) -> None:
        """Show ``bounds`` centered and as large as fits into a ``width`` x ``height`` viewport."""
        self.viewport_width = max(width, 1.0)
        self.viewport_height = max(height, 1.0)
        span_x = max(bounds.width, 1e-30)
        span_y = max(bounds.height, 1e-30)
        self.scale = min(self.viewport_width / span_x, self.viewport_height / span_y) * margin
        self._fit_scale = self.scale
        self.center_x = 0.5 * (bounds.x0 + bounds.x1)
        self.center_y = 0.5 * (bounds.y0 + bounds.y1)

    def world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Convert meters to viewport pixels."""
        return (
            self.viewport_width / 2.0 + (x - self.center_x) * self.scale,
            self.viewport_height / 2.0 - (y - self.center_y) * self.scale,
        )

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        """Convert viewport pixels to meters."""
        return (
            self.center_x + (sx - self.viewport_width / 2.0) / self.scale,
            self.center_y - (sy - self.viewport_height / 2.0) / self.scale,
        )

    def pan(self, dx: float, dy: float) -> None:
        """Drag the content by ``dx``, ``dy`` pixels."""
        self.center_x -= dx / self.scale
        self.center_y += dy / self.scale

    def zoom_at(self, factor: float, sx: float, sy: float) -> None:
        """Zoom by ``factor`` keeping the point under pixel ``sx``, ``sy`` fixed."""
        if factor <= 0 or not math.isfinite(factor):
            return
        anchor = self.screen_to_world(sx, sy)
        self.scale = min(
            max(self.scale * factor, self._fit_scale * _MIN_ZOOM), self._fit_scale * _MAX_ZOOM
        )
        self.center_x = anchor[0] - (sx - self.viewport_width / 2.0) / self.scale
        self.center_y = anchor[1] + (sy - self.viewport_height / 2.0) / self.scale
