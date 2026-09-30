# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The interface the 2D and the 3D view share."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PySide6.QtGui import QImage

from .camera import Tool

__all__ = ["Viewport"]


class Viewport(Protocol):
    """A view of the surface that the main window can drive without knowing its kind."""

    def set_lut(self, lut: NDArray[np.uint8]) -> None:
        """Set the ``(256, 3)`` color lookup table."""

    def set_tool(self, tool: Tool) -> None:
        """Set the action of the left mouse button."""

    def home(self) -> None:
        """Fit the whole surface into the view."""

    def zoom_in(self) -> None:
        """Zoom in one step."""

    def zoom_out(self) -> None:
        """Zoom out one step."""

    def grab_image(self) -> QImage:
        """Return the rendered view."""
