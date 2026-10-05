# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Toolbar controlling the height exaggeration of the 3D view."""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QSizePolicy,
    QSlider,
    QToolBar,
    QWidget,
)

from .icons import themed_svg
from .surface import AUTO_HEIGHT_FRACTION, AUTO_Z_FACTOR_MAX
from .zscale import (
    SLIDER_MAX,
    Z_FACTOR_MAX,
    Z_FACTOR_MIN,
    clamp_z_factor,
    factor_to_slider,
    slider_to_factor,
)

__all__ = ["ZScaleBar"]


class ZScaleBar(QToolBar):
    """Slider, spin box and presets for the height exaggeration.

    ``factor_changed(factor)`` is emitted whenever the factor changes, by the user or by
    :meth:`set_factor`. ``auto_requested()`` is emitted by the *Auto* button; only the owner
    knows the surface, so it answers with :meth:`set_factor`.
    """

    factor_changed = Signal(float)
    auto_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the toolbar at a factor of 1."""
        super().__init__(parent)
        self.setWindowTitle(self.tr("Z scale"))
        self.setObjectName("zScaleToolBar")
        self.setMovable(False)
        self._factor = 1.0
        self._updating = False

        self._icon = QSvgWidget(self)
        self._icon.load(themed_svg("z-scale"))
        self._icon.setFixedSize(20, 20)
        self._icon.setToolTip(self.tr("Z scale (height exaggeration)"))
        self._slider = QSlider(Qt.Orientation.Horizontal, self)
        self._slider.setRange(0, SLIDER_MAX)
        self._slider.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._slider.valueChanged.connect(self._on_slider)
        self._spin = QDoubleSpinBox(self)
        self._spin.setRange(Z_FACTOR_MIN, Z_FACTOR_MAX)
        self._spin.setDecimals(2)
        self._spin.setSuffix(" \u00d7")
        self._spin.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
        self._spin.valueChanged.connect(self._on_user)
        self._auto = QAction(self.tr("Auto"), self)
        self._auto.setToolTip(
            self.tr(
                "Fit the typical height to {percent}% of the extent, at most {limit}\u00d7"
            ).format(percent=round(AUTO_HEIGHT_FRACTION * 100), limit=round(AUTO_Z_FACTOR_MAX))
        )
        self._auto.triggered.connect(self.auto_requested)
        true_scale = QAction("1\u00d7", self)
        true_scale.setToolTip(self.tr("True metric scale"))
        true_scale.triggered.connect(lambda: self.set_factor(1.0))
        for widget in (self._icon, self._slider, self._spin):
            self.addWidget(widget)
        self.addAction(self._auto)
        self.addAction(true_scale)

    def changeEvent(self, event: QEvent) -> None:
        """Recolor the icon when the palette changes."""
        if event.type() == QEvent.Type.PaletteChange:
            self._icon.load(themed_svg("z-scale"))
        super().changeEvent(event)

    @property
    def factor(self) -> float:
        """The current height exaggeration."""
        return self._factor

    def set_factor(self, factor: float) -> None:
        """Show and announce ``factor``, limited to the supported range."""
        self._factor = clamp_z_factor(factor)
        self._updating = True
        self._slider.setValue(factor_to_slider(self._factor))
        self._spin.setValue(self._factor)
        self._updating = False
        self.factor_changed.emit(self._factor)

    def set_auto_available(self, available: bool) -> None:
        """Enable the *Auto* button, which needs a loaded surface."""
        self._auto.setEnabled(available)

    def _on_slider(self, position: int) -> None:
        self._on_user(slider_to_factor(position))

    def _on_user(self, factor: float) -> None:
        if not self._updating:
            self.set_factor(factor)
