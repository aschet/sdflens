# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Render the installer banner and dialog bitmaps from the application icon SVG.

Usage: ``python packaging/make_installer_art.py <output folder>``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect, QRectF
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

SOURCE = Path(__file__).parent.parent / "src" / "sdflens" / "icons" / "app.svg"
PANEL = QColor("#3b5573")  # lighter than the icon tile so that the tile stands out
WHITE = QColor("white")


def new_image(width: int, height: int) -> QImage:
    """Return a white RGB image, as the installer bitmaps are opaque."""
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(WHITE)
    return image


def banner(renderer: QSvgRenderer) -> QImage:
    """Return the 493 x 58 top banner: the icon at the right, the title text is drawn on top."""
    image = new_image(493, 58)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter, QRectF(493 - 50, 5, 48, 48))
    painter.end()
    return image


def dialog(renderer: QSvgRenderer) -> QImage:
    """Return the 493 x 312 bitmap of the welcome and exit pages: a dark panel at the left."""
    image = new_image(493, 312)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.fillRect(QRect(0, 0, 164, 312), PANEL)
    renderer.render(painter, QRectF(22, 96, 120, 120))
    painter.end()
    return image


def main() -> int:
    """Write ``banner.bmp`` and ``dialog.bmp`` to the folder given on the command line."""
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    _app = QGuiApplication(sys.argv[:1])
    renderer = QSvgRenderer(str(SOURCE))
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    for name, image in (("banner.bmp", banner(renderer)), ("dialog.bmp", dialog(renderer))):
        if not image.save(str(target / name)):
            print(f"Cannot write {name}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
