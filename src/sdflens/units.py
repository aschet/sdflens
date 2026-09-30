# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SI unit prefixes, tick generation and number formatting for lengths in meters."""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["format_length", "format_tick", "format_values", "nice_ticks", "unit_for"]

_UNITS = (
    (1e-12, "pm"),
    (1e-9, "nm"),
    (1e-6, "\u00b5m"),
    (1e-3, "mm"),
    (1e-2, "cm"),
    (1.0, "m"),
)


def unit_for(magnitude: float) -> tuple[float, str]:
    """Return the ``(factor, symbol)`` of the largest SI length unit not exceeding ``magnitude``.

    Zero, negative and non-finite magnitudes fall back to meters.

    >>> unit_for(2.5e-6)
    (1e-06, 'µm')
    >>> unit_for(float("nan"))
    (1.0, 'm')
    """
    if not math.isfinite(magnitude) or magnitude <= 0:
        return 1.0, "m"
    chosen = _UNITS[0]
    for unit in _UNITS:
        if magnitude >= unit[0] * 0.999:
            chosen = unit
    return chosen


def format_length(value: float, digits: int = 4) -> str:
    """Format a length in meters with an automatically chosen SI prefix.

    >>> format_length(1.5e-6)
    '1.5 µm'
    >>> format_length(float("nan"))
    'n/a'
    """
    if not math.isfinite(value):
        return "n/a"
    factor, symbol = unit_for(abs(value))
    return f"{value / factor:.{digits}g} {symbol}"


def format_values(values: Sequence[float], digits: int = 4) -> tuple[list[str], str]:
    """Format lengths in metres in one shared SI unit, chosen for the largest magnitude.

    >>> format_values([4e-6, 6e-6])
    (['4', '6'], 'µm')
    >>> format_values([0.0, 1.9e-8])
    (['0', '19'], 'nm')
    """
    finite = [abs(value) for value in values if math.isfinite(value)]
    factor, symbol = unit_for(max(finite, default=0.0))
    texts = [f"{value / factor:.{digits}g}" if math.isfinite(value) else "n/a" for value in values]
    return texts, symbol


def format_tick(value: float, factor: float) -> str:
    """Format ``value`` divided by the unit ``factor`` without floating point noise.

    >>> format_tick(3e-7, 1e-9)
    '300'
    """
    return f"{value / factor:.6g}"


def nice_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    """Return "nice" (1, 2 or 5 times a power of ten) tick positions within ``[lo, hi]``.

    >>> nice_ticks(0.0, 1.0, 6)
    [0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0]
    """
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return []
    if hi < lo:
        lo, hi = hi, lo
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / max(target - 1, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    step = magnitude * 10.0
    for multiple in (1.0, 2.0, 5.0, 10.0):
        if raw <= multiple * magnitude:
            step = multiple * magnitude
            break
    first = math.ceil(lo / step - 1e-9)
    last = math.floor(hi / step + 1e-9)
    return [k * step for k in range(first, last + 1)]
