# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The look of the window: the colors of sdflens web, light or dark as the system is."""

from __future__ import annotations

from typing import NamedTuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

__all__ = ["apply_theme", "stylesheet"]


class _Tokens(NamedTuple):
    stage: str  # behind the picture
    surface: str  # bars, panels and inputs
    text: str
    muted: str
    border: str
    accent: str
    accent_text: str
    hover: str  # accent at low opacity
    disabled: str


_LIGHT = _Tokens(
    stage="#eceef2",
    surface="#ffffff",
    text="#1c2026",
    muted="#5b6573",
    border="#d5dae1",
    accent="#2f6fb0",
    accent_text="#ffffff",
    hover="rgba(47, 111, 176, 0.10)",
    disabled="#a3abb5",
)

_DARK = _Tokens(
    stage="#1a1b1f",
    surface="#202227",
    text="#e7e9ec",
    muted="#9ba4af",
    border="#353840",
    accent="#5aa2e8",
    accent_text="#0d1117",
    hover="rgba(90, 162, 232, 0.16)",
    disabled="#5f6670",
)


def _tokens(scheme: Qt.ColorScheme) -> _Tokens:
    return _DARK if scheme == Qt.ColorScheme.Dark else _LIGHT


def _palette(tokens: _Tokens) -> QPalette:
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: tokens.stage,
        QPalette.ColorRole.WindowText: tokens.text,
        QPalette.ColorRole.Base: tokens.surface,
        QPalette.ColorRole.AlternateBase: tokens.stage,
        QPalette.ColorRole.Text: tokens.text,
        QPalette.ColorRole.Button: tokens.surface,
        QPalette.ColorRole.ButtonText: tokens.text,
        QPalette.ColorRole.ToolTipBase: tokens.surface,
        QPalette.ColorRole.ToolTipText: tokens.text,
        QPalette.ColorRole.PlaceholderText: tokens.muted,
        QPalette.ColorRole.Highlight: tokens.accent,
        QPalette.ColorRole.HighlightedText: tokens.accent_text,
        QPalette.ColorRole.Link: tokens.accent,
        QPalette.ColorRole.Mid: tokens.border,
        QPalette.ColorRole.Midlight: tokens.border,
        QPalette.ColorRole.Dark: tokens.border,
        QPalette.ColorRole.Light: tokens.surface,
        QPalette.ColorRole.Shadow: tokens.border,
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(tokens.disabled))
    return palette


def stylesheet(scheme: Qt.ColorScheme = Qt.ColorScheme.Light) -> str:
    """Return the style sheet for the bars, menus, panels and inputs."""
    t = _tokens(scheme)
    return f"""
QToolBar {{
    background: {t.surface};
    border: 0;
    border-bottom: 1px solid {t.border};
    padding: 4px 8px;
    spacing: 4px;
}}
QToolBar::separator {{
    background: {t.border};
    width: 1px;
    margin: 6px 6px;
}}
QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 5px;
    color: {t.text};
}}
QToolButton:hover {{
    background: {t.hover};
}}
QToolButton:checked {{
    background: {t.hover};
    border-color: {t.accent};
    color: {t.accent};
}}
QToolButton:disabled {{
    background: transparent;
    border-color: transparent;
}}
QToolBar QLabel {{
    padding: 0 2px;
}}
QMenuBar {{
    background: {t.surface};
    border-bottom: 1px solid {t.border};
    padding: 2px 6px;
}}
QMenuBar::item {{
    background: transparent;
    padding: 4px 10px;
    border-radius: 6px;
}}
QMenuBar::item:selected {{
    background: {t.hover};
}}
QMenu {{
    background: {t.surface};
    border: 1px solid {t.border};
    border-radius: 8px;
    padding: 4px;
}}
QMenu::item {{
    padding: 5px 22px 5px 12px;
    border-radius: 6px;
}}
QMenu::item:selected {{
    background: {t.hover};
    color: {t.text};
}}
QMenu::item:disabled {{
    color: {t.disabled};
    background: transparent;
}}
QMenu::separator {{
    height: 1px;
    background: {t.border};
    margin: 4px 6px;
}}
QPushButton {{
    background: {t.surface};
    border: 1px solid {t.border};
    border-radius: 8px;
    padding: 5px 14px;
}}
QPushButton:hover {{
    background: {t.hover};
}}
QPushButton:default, QPushButton:pressed {{
    border-color: {t.accent};
}}
QPushButton:disabled {{
    color: {t.disabled};
    background: transparent;
}}
QComboBox, QLineEdit {{
    background: {t.stage};
    border: 1px solid {t.border};
    border-radius: 8px;
    padding: 4px 8px;
    selection-background-color: {t.accent};
    selection-color: {t.accent_text};
}}
QComboBox:focus, QLineEdit:focus {{
    border-color: {t.accent};
}}
QComboBox::drop-down {{
    border: 0;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {t.surface};
    border: 1px solid {t.border};
    selection-background-color: {t.hover};
    selection-color: {t.text};
    outline: 0;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {t.border};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {t.accent};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {t.accent};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:disabled {{
    background: {t.disabled};
}}
QStatusBar {{
    background: {t.surface};
    border-top: 1px solid {t.border};
    color: {t.muted};
}}
QStatusBar::item {{
    border: 0;
}}
QScrollArea, QScrollArea > QWidget > QWidget {{
    background: {t.surface};
    border: 0;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {t.border};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {t.muted};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}
#infoBody {{
    background: {t.surface};
    border-left: 1px solid {t.border};
}}
#infoBody QPushButton {{
    background: {t.stage};
}}
QPlainTextEdit, QListWidget {{
    background: {t.stage};
    border: 1px solid {t.border};
    border-radius: 8px;
}}
QToolTip {{
    background: {t.surface};
    color: {t.text};
    border: 1px solid {t.border};
    padding: 4px 6px;
}}
"""


def apply_theme(app: QApplication) -> None:
    """Style ``app`` and keep it in step with the light or dark setting of the system."""
    app.setStyle("Fusion")

    def update(scheme: Qt.ColorScheme | None = None) -> None:
        scheme = scheme or app.styleHints().colorScheme()
        app.setPalette(_palette(_tokens(scheme)))
        app.setStyleSheet(stylesheet(scheme))

    update()
    app.styleHints().colorSchemeChanged.connect(update)
