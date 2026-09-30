# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Render the application icon SVG into a multi-size Windows ``.ico`` file.

Usage: ``python packaging/make_icon.py <output.ico>``
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

SIZES = (16, 24, 32, 48, 64, 128, 256)
SOURCE = Path(__file__).parent.parent / "src" / "sdflens" / "icons" / "app.svg"


def render_png(renderer: QSvgRenderer, size: int) -> bytes:
    """Return the icon rendered at ``size`` x ``size`` pixels as PNG data."""
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(data.data())


def main() -> int:
    """Write the ``.ico`` file given on the command line."""
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    _app = QGuiApplication(sys.argv[:1])
    renderer = QSvgRenderer(str(SOURCE))
    images = [render_png(renderer, size) for size in SIZES]

    # ICONDIR followed by one ICONDIRENTRY per PNG-compressed image (supported since Vista).
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = len(header) + 16 * len(images)
    entries = b""
    for size, png in zip(SIZES, images, strict=True):
        dimension = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dimension, dimension, 0, 0, 1, 32, len(png), offset)
        offset += len(png)

    target = Path(sys.argv[1])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(header + entries + b"".join(images))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
