# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Main window: toolbars, 2D/3D/profile views, color bar, information panel and files."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import x3pio
from PySide6.QtCore import QDir, Qt
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .aboutdialog import AboutDialog
from .camera import Tool
from .colorbar import ColorBar
from .colormap import build_lut, colormap_names
from .convert import profile_file
from .exporter import Exporter, compose_screenshot
from .glwidget import RenderMode, SurfaceView
from .heatmap import HeatmapView
from .icons import app_icon, load_icon
from .infopanel import InfoPanel
from .loader import Loader
from .mesh import SurfaceMesh
from .pointcloud import PointCloudModel
from .profileview import ProfileView
from .settings import Settings
from .surface import Model3D, SurfaceModel
from .surfacefile import SurfaceFile, layer_count
from .units import format_values
from .viewport import Viewport
from .zscalebar import ZScaleBar

__all__ = ["MainWindow"]

_VIEW_3D = "3d"
_VIEW_2D = "2d"
_VIEW_PROFILE = "profile"


class MainWindow(QMainWindow):
    """The sdflens application window."""

    def __init__(self) -> None:
        """Create the window with no file loaded."""
        super().__init__()
        self._settings = Settings()
        self._file: SurfaceFile | None = None
        self._model: SurfaceModel | PointCloudModel | None = None
        self._view_preference = _VIEW_3D
        self._path: str | None = None
        self._layer = 0  # the layer that is shown, counted from 0
        self._pending_layer: int | None = None  # a layer to show once the one being built is done

        self._surface_view = SurfaceView(self)
        self._heatmap_view = HeatmapView(self)
        self._profile_view = ProfileView(self)
        self._views: tuple[Viewport, ...] = (
            self._surface_view,
            self._heatmap_view,
            self._profile_view,
        )
        self._stack = QStackedWidget(self)
        self._stack.addWidget(self._surface_view)
        self._stack.addWidget(self._heatmap_view)
        self._stack.addWidget(self._profile_view)
        self._colorbar = ColorBar(self)
        self._info_panel = InfoPanel(self)

        self._loader = Loader(self)
        self._loader.loaded.connect(self._on_loaded)
        self._loader.failed.connect(self._on_failed)
        self._loader.layer_loaded.connect(self._on_layer_loaded)
        self._loader.layer_failed.connect(self._on_layer_failed)
        self._exporter = Exporter(self, self._settings)
        self._exporter.message.connect(self.statusBar().showMessage)
        self._heatmap_view.hovered.connect(self._on_hovered)
        self._surface_view.picked.connect(self._on_hovered)
        self._profile_view.hovered.connect(self._on_hovered)
        self._profile_view.save_requested.connect(self._save_profile)

        self._create_actions()
        self._create_central()
        self._create_menus()
        self._create_toolbars()
        self._stats_label = QLabel(self)
        self.statusBar().addPermanentWidget(self._stats_label)

        self._colormap_combo.setCurrentText(self._settings.colormap)
        self._reverse_action.setChecked(self._settings.reverse)
        self._apply_colormap()
        self._set_view_mode(self._settings.view)
        self._set_render_mode(self._stored_render_mode())
        self._z_bar.set_factor(1.0)
        self._update_actions()

        geometry = self._settings.geometry
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(1200, 800)
        self.setAcceptDrops(True)
        self.setWindowIcon(app_icon())
        self._update_title()
        self.statusBar().showMessage(self.tr("Open a surface file to start"))

    def load_file(self, path: str) -> None:
        """Load ``path`` in the background and show it when done."""
        path = QDir.toNativeSeparators(path)
        if not self._loader.load(path):
            return
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        self.statusBar().showMessage(self.tr("Loading {path} ...").format(path=path))
        self._update_actions()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept dragged local files."""
        if event.mimeData().hasUrls() and not self._loader.busy:
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        """Load the first dropped local file."""
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.load_file(url.toLocalFile())
                return

    def closeEvent(self, event: QCloseEvent) -> None:
        """Remember the window and view settings."""
        self._settings.geometry = self.saveGeometry()
        self._settings.view = self._view_preference
        self._settings.colormap = self._colormap_combo.currentText()
        self._settings.reverse = self._reverse_action.isChecked()
        self._settings.render_mode = self._render_mode.value
        self._surface_view.release_gl()
        super().closeEvent(event)

    def createPopupMenu(self) -> QMenu | None:
        """Offer no context menu, so toolbars cannot be hidden."""
        return None

    def _create_actions(self) -> None:
        self._open_action = self._action(self.tr("&Open..."), "open", QKeySequence.StandardKey.Open)
        self._open_action.triggered.connect(self._open_dialog)
        self._save_as_action = self._action(
            self.tr("Save &as..."), "save-as", QKeySequence.StandardKey.Save
        )
        self._save_as_action.setShortcuts(
            [
                QKeySequence(QKeySequence.StandardKey.Save),
                QKeySequence(QKeySequence.StandardKey.SaveAs),
            ]
        )
        self._save_as_action.triggered.connect(self._save_as)
        self._screenshot_action = self._action(
            self.tr("&Save..."),
            "screenshot",
            self.tr("Ctrl+Alt+S", "shortcut"),
            self.tr("Save screenshot"),
        )
        self._screenshot_action.triggered.connect(self._save_screenshot)
        self._copy_screenshot_action = self._action(
            self.tr("&Copy"), "copy", QKeySequence.StandardKey.Copy, self.tr("Copy screenshot")
        )
        self._copy_screenshot_action.triggered.connect(self._copy_screenshot)
        self._export_action = self._action(
            self.tr("&Export..."),
            "export",
            self.tr("Ctrl+E", "shortcut"),
            self.tr("Export information"),
        )
        self._export_action.triggered.connect(self._export_metadata)
        self._copy_metadata_action = self._action(
            self.tr("&Copy"),
            "copy",
            self.tr("Ctrl+Shift+C", "shortcut"),
            self.tr("Copy information"),
        )
        self._copy_metadata_action.triggered.connect(self._copy_metadata)
        self._quit_action = self._action(self.tr("E&xit"), None, QKeySequence.StandardKey.Quit)
        if not self._quit_action.shortcuts():
            # Qt defines no standard quit key on Windows; Alt+F4 is its convention.
            self._quit_action.setShortcut(QKeySequence(self.tr("Alt+F4", "shortcut")))
        self._quit_action.triggered.connect(self.close)

        self._home_action = self._action(self.tr("&Home"), "home", self.tr("Home", "shortcut"))
        self._home_action.triggered.connect(lambda: self._current().home())
        self._zoom_in_action = self._action(
            self.tr("Zoom &in"), "zoom-in", QKeySequence.StandardKey.ZoomIn
        )
        self._zoom_in_action.triggered.connect(lambda: self._current().zoom_in())
        self._zoom_out_action = self._action(
            self.tr("Zoom &out"), "zoom-out", QKeySequence.StandardKey.ZoomOut
        )
        self._zoom_out_action.triggered.connect(lambda: self._current().zoom_out())

        tools = QActionGroup(self)
        self._rotate_action = self._tool_action(
            tools, self.tr("&Rotate"), "rotate", self.tr("R", "shortcut"), Tool.ROTATE
        )
        self._pan_action = self._tool_action(
            tools, self.tr("&Pan"), "pan", self.tr("P", "shortcut"), Tool.PAN
        )
        self._zoom_action = self._tool_action(
            tools, self.tr("&Zoom"), "zoom", self.tr("Z", "shortcut"), Tool.ZOOM
        )
        self._rotate_action.setChecked(True)

        modes = QActionGroup(self)
        self._view_3d_action = self._mode_action(
            modes, self.tr("&3D view"), "view-3d", self.tr("Ctrl+3", "shortcut"), _VIEW_3D
        )
        self._view_2d_action = self._mode_action(
            modes, self.tr("&2D view"), "view-2d", self.tr("Ctrl+2", "shortcut"), _VIEW_2D
        )
        self._view_profile_action = self._mode_action(
            modes,
            self.tr("&Profile view"),
            "view-profile",
            self.tr("Ctrl+1", "shortcut"),
            _VIEW_PROFILE,
        )

        self._render_mode = RenderMode.SURFACE
        render_modes = QActionGroup(self)
        self._render_actions: dict[RenderMode, QAction] = {}
        for mode, text in (
            (RenderMode.SURFACE, self.tr("&Surface")),
            (RenderMode.WIREFRAME, self.tr("&Wireframe")),
            (RenderMode.POINTS, self.tr("&Points")),
        ):
            action = self._action(text, None)
            action.setCheckable(True)
            render_modes.addAction(action)
            action.triggered.connect(lambda _checked=False, m=mode: self._set_render_mode(m))
            self._render_actions[mode] = action

        self._reverse_action = self._action(
            self.tr("Reverse &colormap"), "reverse", self.tr("Ctrl+R", "shortcut")
        )
        self._reverse_action.setCheckable(True)
        self._reverse_action.toggled.connect(self._apply_colormap)
        self._about_action = self._action(
            self.tr("&About..."), None, QKeySequence.StandardKey.HelpContents
        )
        self._about_action.setMenuRole(QAction.MenuRole.AboutRole)
        self._about_action.triggered.connect(self._show_about)

        self._colormap_combo = QComboBox(self)
        self._colormap_combo.addItems(colormap_names())
        self._colormap_combo.currentTextChanged.connect(self._apply_colormap)

        # The layers of a file are numbered; the format gives them no names.
        self._layer_spin = QSpinBox(self)
        self._layer_spin.setRange(1, 1)
        self._layer_spin.setToolTip(self.tr("Layer to show"))
        self._layer_spin.valueChanged.connect(self._on_layer_chosen)
        self._previous_layer_action = self._action(
            self.tr("Pre&vious layer"), None, self.tr("Ctrl+Up", "shortcut")
        )
        self._previous_layer_action.triggered.connect(
            lambda: self._layer_spin.setValue(self._layer_spin.value() - 1)
        )
        self._next_layer_action = self._action(
            self.tr("&Next layer"), None, self.tr("Ctrl+Down", "shortcut")
        )
        self._next_layer_action.triggered.connect(
            lambda: self._layer_spin.setValue(self._layer_spin.value() + 1)
        )

    def _action(
        self,
        text: str,
        icon: str | None,
        shortcut: str | QKeySequence.StandardKey | None = None,
        tooltip: str | None = None,
    ) -> QAction:
        """Create an action; its tooltip is ``tooltip`` (default: the text) plus the shortcut."""
        action = QAction(load_icon(icon), text, self) if icon else QAction(text, self)
        if isinstance(shortcut, QKeySequence.StandardKey):
            action.setShortcuts(shortcut)
        elif shortcut:
            action.setShortcut(QKeySequence(shortcut))
        name = tooltip or action.toolTip()
        keys = action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
        action.setToolTip(f"{name} ({keys})" if keys else name)
        return action

    def _tool_action(
        self, group: QActionGroup, text: str, icon: str, shortcut: str, tool: Tool
    ) -> QAction:
        action = self._action(text, icon, shortcut)
        action.setCheckable(True)
        group.addAction(action)
        action.triggered.connect(lambda: self._set_tool(tool))
        return action

    def _mode_action(
        self, group: QActionGroup, text: str, icon: str, shortcut: str, mode: str
    ) -> QAction:
        action = self._action(text, icon, shortcut)
        action.setCheckable(True)
        group.addAction(action)
        action.triggered.connect(lambda: self._choose_view(mode))
        return action

    def _action_button(self, action: QAction, parent: QWidget) -> QPushButton:
        button = QPushButton(action.icon(), action.iconText(), parent)
        button.setToolTip(action.toolTip())
        button.clicked.connect(action.trigger)
        button.setEnabled(action.isEnabled())
        action.changed.connect(lambda: button.setEnabled(action.isEnabled()))
        return button

    def _create_central(self) -> None:
        surface = QWidget(self)
        surface_layout = QHBoxLayout(surface)
        surface_layout.addWidget(self._stack, 1)
        surface_layout.addWidget(self._colorbar)
        self.setCentralWidget(surface)

        info = QWidget(self)
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(0, 0, 0, 0)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        for action in (self._export_action, self._copy_metadata_action):
            buttons.addWidget(self._action_button(action, info))
        buttons.addStretch(1)
        self._info_panel.extension_save_requested.connect(self._save_extension)
        info_layout.addWidget(self._info_panel, 1)
        info_layout.addLayout(buttons)

        self._info_dock = QDockWidget(self.tr("Information"), self)
        self._info_dock.setObjectName("infoDock")
        self._info_dock.setWidget(info)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._info_dock)
        self._info_toggle_action = self._info_dock.toggleViewAction()
        self._info_toggle_action.setText(self.tr("In&formation"))
        self._info_toggle_action.setShortcut(QKeySequence(self.tr("Ctrl+I", "shortcut")))

    def _create_menus(self) -> None:
        file_menu = self.menuBar().addMenu(self.tr("&File"))
        file_menu.addAction(self._open_action)
        file_menu.addAction(self._save_as_action)
        file_menu.addSeparator()
        screenshot_menu = file_menu.addMenu(self.tr("&Screenshot"))
        screenshot_menu.addAction(self._screenshot_action)
        screenshot_menu.addAction(self._copy_screenshot_action)
        information_menu = file_menu.addMenu(self.tr("&Information"))
        information_menu.addAction(self._export_action)
        information_menu.addAction(self._copy_metadata_action)
        file_menu.addSeparator()
        file_menu.addAction(self._quit_action)

        view_menu = self.menuBar().addMenu(self.tr("&View"))
        view_menu.addAction(self._view_3d_action)
        view_menu.addAction(self._view_2d_action)
        view_menu.addAction(self._view_profile_action)
        view_menu.addAction(self._info_toggle_action)
        view_menu.addSeparator()
        view_menu.addAction(self._previous_layer_action)
        view_menu.addAction(self._next_layer_action)
        view_menu.addSeparator()
        view_menu.addAction(self._home_action)
        view_menu.addAction(self._zoom_in_action)
        view_menu.addAction(self._zoom_out_action)
        view_menu.addSeparator()
        view_menu.addAction(self._rotate_action)
        view_menu.addAction(self._pan_action)
        view_menu.addAction(self._zoom_action)
        view_menu.addSeparator()
        self._render_menu = view_menu.addMenu(self.tr("3D &display"))
        for action in self._render_actions.values():
            self._render_menu.addAction(action)
        view_menu.addAction(self._reverse_action)

        help_menu = self.menuBar().addMenu(self.tr("&Help"))
        help_menu.addAction(self._about_action)

    def _create_toolbars(self) -> None:
        main = QToolBar(self.tr("Main"), self)
        main.setObjectName("mainToolBar")
        main.setMovable(False)
        for action in (
            self._open_action,
            self._save_as_action,
            self._screenshot_action,
            self._copy_screenshot_action,
        ):
            main.addAction(action)
        main.addSeparator()
        for action in (self._view_3d_action, self._view_2d_action, self._view_profile_action):
            main.addAction(action)
        main.addSeparator()
        for action in (self._home_action, self._zoom_in_action, self._zoom_out_action):
            main.addAction(action)
        main.addSeparator()
        for action in (self._rotate_action, self._pan_action, self._zoom_action):
            main.addAction(action)
        main.addSeparator()
        main.addWidget(QLabel(self.tr("Colormap:"), self))
        main.addWidget(self._colormap_combo)
        main.addAction(self._reverse_action)
        # The layer selector is shown only for a file of several layers.
        self._layer_controls = [
            main.addSeparator(),
            main.addWidget(QLabel(self.tr("Layer:"), self)),
            main.addWidget(self._layer_spin),
        ]
        for control in self._layer_controls:
            control.setVisible(False)
        self.addToolBar(main)

        self._z_bar = ZScaleBar(self)
        self._z_bar.factor_changed.connect(self._surface_view.set_z_factor)
        self._z_bar.auto_requested.connect(self._on_z_auto)
        self.addToolBar(self._z_bar)

    def _current_mode(self) -> str:
        current = self._stack.currentWidget()
        if current is self._heatmap_view:
            return _VIEW_2D
        return _VIEW_PROFILE if current is self._profile_view else _VIEW_3D

    def _current(self) -> Viewport:
        return cast(Viewport, self._stack.currentWidget())

    def _set_tool(self, tool: Tool) -> None:
        for view in self._views:
            view.set_tool(tool)

    def _set_render_mode(self, mode: RenderMode) -> None:
        self._render_mode = mode
        self._render_actions[mode].setChecked(True)
        self._surface_view.set_render_mode(mode)

    def _stored_render_mode(self) -> RenderMode:
        try:
            return RenderMode(self._settings.render_mode)
        except ValueError:
            return RenderMode.SURFACE

    @property
    def _is_cloud(self) -> bool:
        """Whether points are shown that are drawn as points only, not as a surface."""
        model = self._model
        return isinstance(model, PointCloudModel) and model.grid is None

    @property
    def _has_grid_views(self) -> bool:
        """Whether the 2D and the profile view apply: to a regular grid, not to points."""
        return self._model is None or isinstance(self._model, SurfaceModel)

    def _choose_view(self, mode: str) -> None:
        """Show the view the user chose and prefer it for the next file."""
        self._view_preference = mode
        self._set_view_mode(mode)

    def _set_view_mode(self, mode: str) -> None:
        widgets: dict[str, QWidget] = {
            _VIEW_2D: self._heatmap_view,
            _VIEW_PROFILE: self._profile_view,
        }
        widget = widgets.get(mode, self._surface_view)
        is_3d = widget is self._surface_view
        was_3d = self._stack.currentWidget() is self._surface_view
        self._stack.setCurrentWidget(widget)
        actions = {
            _VIEW_3D: self._view_3d_action,
            _VIEW_2D: self._view_2d_action,
            _VIEW_PROFILE: self._view_profile_action,
        }
        actions[self._current_mode()].setChecked(True)
        self._colorbar.setVisible(widget is not self._profile_view)
        self._rotate_action.setEnabled(is_3d)
        if is_3d and not was_3d:
            self._rotate_action.setChecked(True)
            self._set_tool(Tool.ROTATE)
        elif not is_3d and self._rotate_action.isChecked():
            self._pan_action.setChecked(True)
            self._set_tool(Tool.PAN)
        self._z_bar.setEnabled(is_3d)
        self._render_menu.setEnabled(is_3d and not self._is_cloud)
        self.statusBar().clearMessage()

    def _apply_colormap(self) -> None:
        lut = build_lut(self._colormap_combo.currentText(), self._reverse_action.isChecked())
        for view in self._views:
            view.set_lut(lut)
        self._colorbar.set_lut(lut)

    def _on_z_auto(self) -> None:
        if self._model is not None:
            self._z_bar.set_factor(self._model.auto_z_factor())

    def _update_actions(self) -> None:
        loaded = self._file is not None
        idle = not self._loader.busy
        self._open_action.setEnabled(idle)
        for action in (
            self._screenshot_action,
            self._copy_screenshot_action,
            self._home_action,
            self._zoom_in_action,
            self._zoom_out_action,
        ):
            action.setEnabled(loaded)
        self._z_bar.set_auto_available(loaded)
        self._save_as_action.setEnabled(loaded)
        self._export_action.setEnabled(loaded)
        self._copy_metadata_action.setEnabled(loaded)
        for action in (self._view_2d_action, self._view_profile_action):
            action.setEnabled(self._has_grid_views)

    def _update_title(self, path: str | None = None) -> None:
        # Qt appends the application display name to every window title.
        self.setWindowTitle(Path(path).name if path else "")

    def _summary(self, model: Model3D, mesh: SurfaceMesh) -> str:
        (x, y), unit = format_values([model.size_x, model.size_y])
        if isinstance(model, PointCloudModel) and model.grid is None:
            return self.tr("size = {x} \u00d7 {y} {unit}, {count} points").format(
                x=x, y=y, unit=unit, count=model.num_points
            )
        if mesh.step > 1:
            return self.tr("size = {x} \u00d7 {y} {unit}, 3D view downsampled by {step}").format(
                x=x, y=y, unit=unit, step=mesh.step
            )
        return self.tr("size = {x} \u00d7 {y} {unit}").format(x=x, y=y, unit=unit)

    def _open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open surface file"),
            self._settings.dialog_directory(),
            self.tr(
                "Surface files (*.sdf *.x3p);;SDF files (*.sdf);;x3p files (*.x3p);;All files (*)"
            ),
        )
        if path:
            self._settings.remember_directory(path)
            self.load_file(path)

    def _save_as(self) -> None:
        if self._file is not None:
            self._exporter.save_as(self._file, self._path, self._layer)

    def _save_profile(self) -> None:
        if self._file is None:
            return
        values, step, along_x, number = self._profile_view.current_profile()
        name = f"{'profile' if along_x else 'column'}_{number}"
        source = Path(self._path or "surface")
        self._exporter.save_as(
            profile_file(self._file, values, step),
            str(source.with_name(f"{source.stem}_{name}{source.suffix}")),
            0,
        )

    def _capture(self) -> QImage | None:
        view_image = self._current().grab_image()
        if view_image.isNull():
            self.statusBar().showMessage(self.tr("The view could not be captured"), 5000)
            return None
        if self._colorbar.isHidden():  # the profile view has no color bar
            return view_image
        bar_image = self._colorbar.grab().toImage()
        return compose_screenshot(view_image, bar_image, self.palette().window())

    def _save_screenshot(self) -> None:
        self._exporter.save_screenshot(self._path, self._capture)

    def _copy_screenshot(self) -> None:
        self._exporter.copy_screenshot(self._capture())

    def _export_metadata(self) -> None:
        if self._file is not None:
            self._exporter.export_metadata(self._file, self._path)

    def _copy_metadata(self) -> None:
        if self._file is not None:
            self._exporter.copy_metadata(self._file)

    def _save_extension(self, name: str) -> None:
        file = self._file
        if isinstance(file, x3pio.X3pFile) and name in file.extensions:
            self._exporter.export_extension(name, file.extensions[name])

    def _show_about(self) -> None:
        AboutDialog(self).exec()

    def _on_loaded(
        self,
        path: str,
        file: SurfaceFile,
        model: SurfaceModel | PointCloudModel,
        mesh: SurfaceMesh,
        checksum_failed: bool,
    ) -> None:
        QApplication.restoreOverrideCursor()
        self._file = file
        self._model = model
        self._path = path
        self._layer = 0
        self._pending_layer = None
        self._configure_layers(layer_count(file))
        self._surface_view.set_surface(model, mesh)
        grid = model if isinstance(model, SurfaceModel) else None
        self._heatmap_view.set_model(grid)
        self._profile_view.set_model(grid)
        self._colorbar.set_range(model.value_range, model.invalid_count > 0)
        self._info_panel.set_file(file)
        self._stats_label.setText(self._stats_text(model, mesh))
        self._z_bar.set_factor(1.0)
        self._set_view_mode(self._view_preference if grid is not None else _VIEW_3D)
        self._update_title(path)
        self._update_actions()
        name = Path(path).name
        if checksum_failed:
            self.statusBar().showMessage(
                self.tr("Loaded {name}: checksum mismatch").format(name=name),
                10000,
            )
        else:
            self.statusBar().showMessage(self.tr("Loaded {name}").format(name=name), 2000)

    def _configure_layers(self, count: int) -> None:
        """Set the layer selector for a file of ``count`` layers, showing it for several."""
        self._layer_spin.blockSignals(True)
        self._layer_spin.setRange(1, max(count, 1))
        self._layer_spin.setValue(1)
        self._layer_spin.setSuffix(self.tr(" of {count}").format(count=count))
        self._layer_spin.blockSignals(False)
        for control in self._layer_controls:
            control.setVisible(count > 1)
        for action in (self._previous_layer_action, self._next_layer_action):
            action.setEnabled(count > 1)

    def _stats_text(self, model: Model3D, mesh: SurfaceMesh) -> str:
        """Return the size and, for a file of several layers, the layer that is shown."""
        text = self._summary(model, mesh)
        if self._file is not None and layer_count(self._file) > 1:
            return self.tr("layer {number} of {count}, {text}").format(
                number=self._layer + 1, count=layer_count(self._file), text=text
            )
        return text

    def _on_layer_chosen(self, number: int) -> None:
        """Show the layer the user chose, counted from 1, or as soon as the one being built is."""
        index = number - 1
        if self._file is None:
            return
        if self._loader.busy:
            self._pending_layer = index
        elif index != self._layer:
            self._loader.load_layer(self._file, index)

    def _on_layer_loaded(
        self, layer: int, model: SurfaceModel | PointCloudModel, mesh: SurfaceMesh
    ) -> None:
        """Show another layer of the file in every view, keeping the zoom and the camera."""
        self._layer = layer
        self._model = model
        self._surface_view.set_surface(model, mesh, keep_view=True)
        grid = model if isinstance(model, SurfaceModel) else None
        self._heatmap_view.set_model(grid, keep_view=True)
        self._profile_view.set_model(grid, keep_selection=True)
        self._colorbar.set_range(model.value_range, model.invalid_count > 0)
        self._stats_label.setText(self._stats_text(model, mesh))
        self._update_actions()
        wanted, self._pending_layer = self._pending_layer, None
        if wanted is not None and wanted != layer and self._file is not None:
            self._loader.load_layer(self._file, wanted)

    def _on_layer_failed(self, layer: int, message: str) -> None:
        """Say why a layer cannot be shown, and go back to the one that is."""
        self._pending_layer = None
        self._layer_spin.blockSignals(True)
        self._layer_spin.setValue(self._layer + 1)
        self._layer_spin.blockSignals(False)
        self.statusBar().showMessage(
            self.tr("Layer {number}: {message}").format(number=layer + 1, message=message), 8000
        )

    def _on_failed(self, path: str, message: str) -> None:
        QApplication.restoreOverrideCursor()
        self._update_actions()
        self.statusBar().clearMessage()
        QMessageBox.critical(self, self.tr("Cannot open file"), f"{path}\n\n{message}")

    def _on_hovered(self, text: str) -> None:
        if text:
            self.statusBar().showMessage(text)
        else:
            self.statusBar().clearMessage()
