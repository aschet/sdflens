# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Colormaps and mapping of height values, including non-measured points, to colors."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .colormap_data import TABLES

__all__ = ["DEFAULT_COLORMAP", "apply_colormap", "build_lut", "colormap_names"]

LUT_SIZE = 256

_GRAYSCALE = "Grayscale"

DEFAULT_COLORMAP = "Viridis"


def colormap_names() -> list[str]:
    """Return the names of the available colormaps."""
    return [*TABLES, _GRAYSCALE]


def build_lut(name: str, reverse: bool = False) -> NDArray[np.uint8]:
    """Return a ``(256, 3)`` uint8 lookup table for the colormap ``name``.

    :raises KeyError: If ``name`` is not one of :func:`colormap_names`.

    >>> build_lut("Grayscale").shape
    (256, 3)
    >>> build_lut("Grayscale", reverse=True)[0].tolist()
    [255, 255, 255]
    """
    if name == _GRAYSCALE:
        lut = np.repeat(np.arange(LUT_SIZE, dtype=np.uint8)[:, np.newaxis], 3, axis=1)
    else:
        lut = np.frombuffer(bytes.fromhex(TABLES[name]), dtype=np.uint8)
        lut = lut.reshape(LUT_SIZE, 3)
    return lut[::-1].copy() if reverse else lut.copy()


def apply_colormap(
    data: NDArray[np.float64], lut: NDArray[np.uint8], lo: float, hi: float
) -> NDArray[np.uint8]:
    """Map ``data`` to RGBA using ``lut`` over ``[lo, hi]``; ``NaN`` points are transparent.

    >>> rgba = apply_colormap(np.array([[0.0, np.nan, 1.0]]), build_lut("Grayscale"), 0.0, 1.0)
    >>> rgba[0].tolist()
    [[0, 0, 0, 255], [0, 0, 0, 0], [255, 255, 255, 255]]
    """
    valid = np.isfinite(data)
    span = hi - lo
    fraction = np.where(valid, (data - lo) / span if span > 0 else 0.0, 0.0)
    indices = np.rint(np.clip(fraction, 0.0, 1.0) * (len(lut) - 1)).astype(np.intp)
    rgba = np.empty((*data.shape, 4), dtype=np.uint8)
    rgba[..., :3] = lut[indices]
    rgba[..., 3] = np.where(valid, 255, 0)
    rgba[~valid, :3] = 0
    return rgba
