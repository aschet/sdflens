# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write the third-party license notices of the runtime dependencies: make_notices.py <output>."""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

from dependencies import runtime_distributions

_RULE = "=" * 78
_COLORMAP_NOTICES = Path(__file__).resolve().parents[1] / "src" / "sdflens" / "colormap_notices.txt"


def section(dist: metadata.Distribution) -> str:
    """Return the notice of one distribution: its license terms and the license files it ships."""
    info = dist.metadata
    expression = info.get("License-Expression") or info.get("License") or "unknown"
    # Dual-licensed packages (Qt) are used under their GPL option, so the reference to their
    # commercial terms that they ship as a license file does not apply.
    gpl_choice = " OR " in expression and "GPL-3.0" in expression
    urls = dict(url.split(", ", 1) for url in info.get_all("Project-URL") or [])
    source = urls.get("Source") or urls.get("Repository") or urls.get("Homepage")

    lines = [_RULE, f"{info['Name']} {dist.version}", _RULE, "", f"License: {expression}"]
    if source:
        lines.append(f"Source: {source}")
    if gpl_choice:
        lines.append("Used under the terms of the GNU GPL version 3 (see the LICENSE file).")
    texts = [
        f"--- {'/'.join(file.parts[2:])} ---\n{file.read_text(encoding='utf-8').strip()}"
        for file in dist.files or []
        if file.parts[1:2] == ("licenses",)
        and not (gpl_choice and file.name.startswith("LicenseRef-"))
    ]
    if not texts and not gpl_choice:
        print(f"warning: {info['Name']} ships no license file", file=sys.stderr)
    return "\n\n".join(["\n".join(lines), *texts])


def main(output: str) -> None:
    """Write the notices of all runtime dependencies of sdflens and of its colormaps."""
    sections = [section(dist) for dist in runtime_distributions("sdflens")]
    colormaps = _COLORMAP_NOTICES.read_text(encoding="utf-8").strip()
    sections.append(f"{_RULE}\nColormaps\n{_RULE}\n\n{colormaps}")
    Path(output).write_text("\n\n\n".join(sections) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
