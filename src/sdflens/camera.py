# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Orbit camera for the 3D view (right-handed, z up), computed with numpy only."""

from __future__ import annotations

import math
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

__all__ = ["Camera", "Tool"]

_FOV_DEGREES = 40.0
_HOME_AZIMUTH = -65.0
_HOME_ELEVATION = 30.0
_MAX_ELEVATION = 89.0
_ROTATE_DEGREES_PER_PIXEL = 0.4
_MIN_DISTANCE_FACTOR = 0.05
_MAX_DISTANCE_FACTOR = 50.0


class Tool(Enum):
    """Action bound to a left mouse drag."""

    ROTATE = auto()
    PAN = auto()
    ZOOM = auto()


class Camera:
    """Turntable camera orbiting a target, looking at a scene of a given bounding radius.

    Azimuth ``-90`` looks along +y with x pointing right; elevation is above the xy plane.

    >>> camera = Camera()
    >>> camera.fit(1.0)
    >>> view, proj = camera.matrices(1.0)
    >>> view.shape, proj.shape
    ((4, 4), (4, 4))
    """

    def __init__(self) -> None:
        """Create a camera at the home view of a unit-radius scene."""
        self.radius = 1.0
        self.azimuth = _HOME_AZIMUTH
        self.elevation = _HOME_ELEVATION
        self.distance = 1.0
        self.target = np.zeros(3)
        self.fit(1.0)

    def fit(self, radius: float) -> None:
        """Reset to the home view showing a bounding sphere of ``radius``."""
        self.radius = max(radius, 1e-9)
        self.azimuth = _HOME_AZIMUTH
        self.elevation = _HOME_ELEVATION
        self.target = np.zeros(3)
        self.distance = self.radius / math.sin(math.radians(_FOV_DEGREES) / 2.0) * 1.1

    def set_radius(self, radius: float) -> None:
        """Change the scene radius, keeping the apparent scene size unless it must grow."""
        radius = max(radius, 1e-9)
        self.distance *= radius / self.radius
        self.radius = radius

    def rotate(self, dx: float, dy: float) -> None:
        """Rotate about the target by a mouse movement in pixels."""
        self.azimuth -= dx * _ROTATE_DEGREES_PER_PIXEL
        self.elevation = min(
            max(self.elevation + dy * _ROTATE_DEGREES_PER_PIXEL, -_MAX_ELEVATION), _MAX_ELEVATION
        )

    def set_view(self, azimuth: float, elevation: float) -> None:
        """Look at the target from a direction, keeping the distance; elevation is limited."""
        self.azimuth = azimuth
        self.elevation = min(max(elevation, -_MAX_ELEVATION), _MAX_ELEVATION)

    def pan(self, dx: float, dy: float, viewport_height: float) -> None:
        """Move the target in the view plane by a mouse movement in pixels."""
        right, up, _ = self._axes()
        per_pixel = 2.0 * self.distance * math.tan(math.radians(_FOV_DEGREES) / 2.0)
        per_pixel /= max(viewport_height, 1.0)
        self.target = self.target - right * dx * per_pixel + up * dy * per_pixel

    def zoom(self, factor: float) -> None:
        """Move towards (``factor`` > 1) or away from (< 1) the target."""
        if factor <= 0:
            return
        self.distance = min(
            max(self.distance / factor, self.radius * _MIN_DISTANCE_FACTOR),
            self.radius * _MAX_DISTANCE_FACTOR,
        )

    def eye(self) -> NDArray[np.float64]:
        """Return the camera position."""
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        direction = np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
        )
        return self.target + direction * self.distance

    def matrices(self, aspect: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the row-major ``(view, projection)`` matrices for a viewport aspect ratio."""
        right, up, forward = self._axes()
        eye = self.eye()
        view = np.eye(4)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -view[:3, :3] @ eye

        near = max(self.distance - 1.5 * self.radius, self.distance * 0.01)
        far = self.distance + 1.5 * self.radius
        focal = 1.0 / math.tan(math.radians(_FOV_DEGREES) / 2.0)
        proj = np.zeros((4, 4))
        proj[0, 0] = focal / max(aspect, 1e-6)
        proj[1, 1] = focal
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2.0 * far * near / (far - near)
        proj[3, 2] = -1.0
        return view, proj

    def _axes(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return the (right, up, forward) unit vectors of the view."""
        forward = self.target - self.eye()
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return right, up, forward
