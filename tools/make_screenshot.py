# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regenerate the README screenshot of samples/microlens-array.sdf in docs/.

Usage: ``python tools/make_screenshot.py``; needs a desktop session with OpenGL 3.3, not the
offscreen platform. The settings of the user are left alone.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEventLoop, QSettings, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from sdflens.mainwindow import MainWindow

ROOT = Path(__file__).parent.parent
SAMPLE = ROOT / "samples" / "microlens-array.sdf"
OUTPUT = ROOT / "docs" / "screenshot.png"
SIZE = (1280, 800)
COLORMAP = "Viridis"


def settle(loop: QEventLoop, milliseconds: int = 500) -> None:
    """Run the event loop for a while, so that the views draw what they were told to show."""
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def main() -> int:
    """Load the sample in a window of fixed size and save a picture of the 3D view."""
    surface_format = QSurfaceFormat()
    surface_format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)  # Wayland defaults to ES
    surface_format.setVersion(3, 3)
    surface_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface_format.setDepthBufferSize(24)
    surface_format.setSamples(4)
    QSurfaceFormat.setDefaultFormat(surface_format)

    QApplication(sys.argv[:1])  # must exist before any widget
    QCoreApplication.setOrganizationName("sdflens-screenshot")
    QCoreApplication.setApplicationName("sdflens-screenshot")
    settings_dir = tempfile.mkdtemp()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, settings_dir)

    window = MainWindow()
    window.resize(*SIZE)
    window._colormap_combo.setCurrentText(COLORMAP)
    window.show()
    loop = QEventLoop()
    window._loader.loaded.connect(lambda *_: loop.quit())
    window.load_file(str(SAMPLE))
    loop.exec()
    window._z_bar._auto.click()  # files open at true scale, which shows the lenses too flat
    window._zoom_in_action.trigger()
    settle(loop)  # let the GL view draw the loaded surface
    window.statusBar().clearMessage()
    OUTPUT.parent.mkdir(exist_ok=True)
    if not window.grab().save(str(OUTPUT)):
        print(f"Could not write {OUTPUT}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
