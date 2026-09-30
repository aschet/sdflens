# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Z scale (height exaggeration) factor limits and slider mapping."""

from __future__ import annotations

import math

__all__ = [
    "SLIDER_MAX",
    "Z_FACTOR_MAX",
    "Z_FACTOR_MIN",
    "clamp_z_factor",
    "factor_to_slider",
    "slider_to_factor",
]

#: Height exaggeration relative to the true metric scale (1.0 = physically true).
Z_FACTOR_MIN = 0.01
Z_FACTOR_MAX = 10000.0
SLIDER_MAX = 1000

_LOG_MIN = math.log10(Z_FACTOR_MIN)
_LOG_MAX = math.log10(Z_FACTOR_MAX)


def clamp_z_factor(factor: float) -> float:
    """Clamp ``factor`` to the supported range; non-finite values give the true scale.

    >>> clamp_z_factor(1e9)
    10000.0
    >>> clamp_z_factor(float("nan"))
    1.0
    """
    if not math.isfinite(factor):
        return 1.0
    return min(max(factor, Z_FACTOR_MIN), Z_FACTOR_MAX)


def factor_to_slider(factor: float) -> int:
    """Map a z factor to a logarithmic slider position in ``[0, SLIDER_MAX]``.

    >>> factor_to_slider(Z_FACTOR_MIN), factor_to_slider(Z_FACTOR_MAX)
    (0, 1000)
    """
    position = (math.log10(clamp_z_factor(factor)) - _LOG_MIN) / (_LOG_MAX - _LOG_MIN)
    return round(position * SLIDER_MAX)


def slider_to_factor(position: int) -> float:
    """Map a slider position back to a z factor (inverse of :func:`factor_to_slider`).

    >>> slider_to_factor(0)
    0.01
    """
    fraction = min(max(position, 0), SLIDER_MAX) / SLIDER_MAX
    return clamp_z_factor(10.0 ** (_LOG_MIN + fraction * (_LOG_MAX - _LOG_MIN)))
