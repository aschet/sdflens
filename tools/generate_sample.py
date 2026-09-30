# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regenerate samples/microlens-array.sdf, a synthetic micro-lens array with a few defects.

Usage: ``python tools/generate_sample.py``
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sdfio

OUTPUT = Path(__file__).parent.parent / "samples" / "microlens-array.sdf"
TRAILER = (
    "Description = Synthetic micro-lens array\r\n"
    "LensPitch = 100 um\r\n"
    "Defects = missing lens, scratch, dropouts\r\n"
)

PITCH = 1e-6  # sampling interval in meters
ROWS, COLS = 300, 400
LENS_PITCH = 100e-6
LENS_RADIUS = 45e-6  # aperture radius of one lens
CURVATURE = 250e-6  # radius of curvature of the lens surface


def sag(r: np.ndarray) -> np.ndarray:
    """Return the height of a spherical surface above its apex plane at distance ``r``."""
    return CURVATURE - np.sqrt(CURVATURE**2 - np.minimum(r, LENS_RADIUS) ** 2)


def build() -> np.ndarray:
    """Return the heights in meters of a 4 x 3 lens array with a missing lens and defects."""
    y, x = np.mgrid[0:ROWS, 0:COLS] * PITCH
    rng = np.random.default_rng(25178)

    local_x = (x % LENS_PITCH) - 0.5 * LENS_PITCH
    local_y = (y % LENS_PITCH) - 0.5 * LENS_PITCH
    r = np.hypot(local_x, local_y)
    z = np.where(r < LENS_RADIUS, sag(np.array(LENS_RADIUS)) - sag(r), 0.0)

    lens_col, lens_row = (x // LENS_PITCH).astype(int), (y // LENS_PITCH).astype(int)
    z[(lens_col == 2) & (lens_row == 1)] = 0.0  # a missing lens
    z -= 0.35e-6 * np.exp(-((y - 0.6 * x - 60e-6) ** 2) / (2 * (1.5e-6) ** 2)) * (z > 0)  # scratch
    z += 15e-9 * rng.standard_normal(z.shape)  # roughness

    z[np.hypot(x - 170e-6, y - 50e-6) < 8e-6] = np.nan  # specular dropout
    z[rng.random(z.shape) < 0.0002] = np.nan  # random dropouts
    return z


def main() -> None:
    """Write the sample file with fixed dates, so that regenerating it changes nothing."""
    date = datetime(2026, 9, 30, tzinfo=UTC)
    metadata = sdfio.SdfMetadata(
        manufacturer_id="sdflens",
        data_type=sdfio.DataType.INT32,
        create_date=date,
        mod_date=date,
    )
    OUTPUT.parent.mkdir(exist_ok=True)
    sdfio.write(
        OUTPUT,
        build(),
        x_scale=PITCH,
        y_scale=PITCH,
        z_scale=1e-10,
        metadata=metadata,
        trailer=TRAILER,
    )


if __name__ == "__main__":
    main()
