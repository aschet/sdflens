# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write a CycloneDX SBOM of sdflens and its runtime dependencies: make_sbom.py <output>."""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

from cyclonedx.contrib.license.factories import LicenseFactory
from cyclonedx.model.bom import Bom
from cyclonedx.model.component import Component, ComponentType
from cyclonedx.output import make_outputter
from cyclonedx.schema import OutputFormat, SchemaVersion
from cyclonedx.validation import make_schemabased_validator
from dependencies import direct_dependencies, runtime_distributions
from packageurl import PackageURL

_ROOT = "sdflens"


def component(dist: metadata.Distribution, kind: ComponentType) -> Component:
    """Return the SBOM component of an installed distribution, licensed as its metadata says."""
    name = dist.metadata["Name"]
    expression = dist.metadata.get("License-Expression") or dist.metadata.get("License")
    purl = PackageURL("pypi", name=name.lower(), version=dist.version)
    return Component(
        type=kind,
        name=name,
        version=dist.version,
        bom_ref=str(purl),
        purl=purl,
        licenses=[LicenseFactory().make_from_string(expression)] if expression else [],
    )


def build_bom() -> Bom:
    """Describe sdflens as an application with its runtime dependency graph."""
    bom = Bom()
    root = component(metadata.distribution(_ROOT), ComponentType.APPLICATION)
    bom.metadata.component = root
    dists = runtime_distributions(_ROOT)
    components = {dist.metadata["Name"]: component(dist, ComponentType.LIBRARY) for dist in dists}
    for item in components.values():
        bom.components.add(item)
    for name, owner in {_ROOT: root, **components}.items():
        bom.register_dependency(
            owner, [components[dep.metadata["Name"]] for dep in direct_dependencies(name)]
        )
    return bom


def main(output: str) -> None:
    """Write the validated SBOM as CycloneDX 1.6 JSON to ``output``."""
    text = make_outputter(build_bom(), OutputFormat.JSON, SchemaVersion.V1_6).output_as_string(
        indent=2
    )
    error = make_schemabased_validator(OutputFormat.JSON, SchemaVersion.V1_6).validate_str(text)
    if error is not None:
        raise SystemExit(f"The generated SBOM is invalid: {error}")
    Path(output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
