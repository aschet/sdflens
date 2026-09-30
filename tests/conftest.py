# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
import sdfio
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from helpers import make_ramp


@pytest.fixture(scope="session")
def qapp(tmp_path_factory: pytest.TempPathFactory) -> QApplication:
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path_factory.mktemp("settings")),
    )
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication([])
    app.setOrganizationName("sdflens-tests")
    app.setApplicationName("sdflens-tests")
    return app


@pytest.fixture
def ramp_path(tmp_path: Path) -> Path:
    path = tmp_path / "ramp.sdf"
    sdfio.write(path, make_ramp(), x_scale=1e-6, y_scale=2e-6, z_scale=1e-12)
    return path
