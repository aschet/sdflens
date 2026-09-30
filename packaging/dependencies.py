# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The installed runtime dependencies of a package, from the metadata pip recorded."""

from __future__ import annotations

from importlib import metadata

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

__all__ = ["direct_dependencies", "runtime_distributions"]


def direct_dependencies(name: str) -> list[metadata.Distribution]:
    """Return the installed distributions that ``name`` requires, without optional extras."""
    found: list[metadata.Distribution] = []
    for text in metadata.requires(name) or []:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        found.append(metadata.distribution(requirement.name))
    return found


def runtime_distributions(name: str) -> list[metadata.Distribution]:
    """Return the installed distributions that ``name`` depends on, directly or not."""
    found: dict[str, metadata.Distribution] = {}
    pending = [name]
    while pending:
        for dist in direct_dependencies(pending.pop()):
            key = canonicalize_name(dist.metadata["Name"])
            if key not in found:
                found[key] = dist
                pending.append(dist.metadata["Name"])
    return [found[key] for key in sorted(found)]
