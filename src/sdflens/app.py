# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Command line entry point of the viewer."""

from __future__ import annotations

import argparse
import ctypes
import sys
from collections.abc import Sequence

from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from . import __version__
from .icons import app_icon
from .mainwindow import MainWindow

__all__ = ["main"]


def _set_windows_app_id() -> None:
    """Give the process its own taskbar identity so Windows shows its icon, not Python's."""
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("sdflens.sdflens")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdflens", description="View ISO 25178 SDF and x3p surface data files in 2D and 3D."
    )
    parser.add_argument("path", nargs="?", help="SDF or x3p file to open")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the viewer.

    :param argv: Arguments to parse; defaults to :data:`sys.argv`.
    :returns: The process exit status.
    """
    args = _build_parser().parse_args(argv)
    _set_windows_app_id()

    # The default format must be set before the QApplication exists.
    surface_format = QSurfaceFormat()
    surface_format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)  # Wayland defaults to ES
    surface_format.setVersion(3, 3)
    surface_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface_format.setDepthBufferSize(24)
    surface_format.setSamples(4)
    QSurfaceFormat.setDefaultFormat(surface_format)

    app = QApplication(sys.argv[:1])
    app.setOrganizationName("sdflens")
    app.setApplicationName("sdflens")
    app.setApplicationDisplayName("sdflens")
    app.setApplicationVersion(__version__)
    app.setWindowIcon(app_icon())

    window = MainWindow()
    window.show()
    if args.path:
        window.load_file(args.path)
    return app.exec()
