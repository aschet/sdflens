# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# PyInstaller spec of the Windows app; built by build_windows.ps1 (SPECPATH is this folder).
from pathlib import Path

root = Path(SPECPATH).parent
package = root / "src" / "sdflens"

# The About dialog shows the notices file that build_windows.ps1 generates before this build.
datas = [
    (str(package / "icons"), "sdflens/icons"),
    (str(root / "build" / "THIRD-PARTY-NOTICES.txt"), "sdflens"),
]

analysis = Analysis(
    [str(root / "packaging" / "launcher.py")],
    pathex=[str(root / "src")],
    datas=datas,
    # The SVG icons need Qt's SVG plugins, which nothing imports directly.
    hiddenimports=["PySide6.QtSvg"],
)

exe = EXE(
    PYZ(analysis.pure),
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="sdflens",
    console=False,
    icon=str(root / "build" / "sdflens.ico"),
)

COLLECT(exe, analysis.binaries, analysis.datas, name="sdflens")
