# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Entry point of the frozen Windows build (PyInstaller cannot start a package with ``-m``)."""

from sdflens.app import main

raise SystemExit(main())
