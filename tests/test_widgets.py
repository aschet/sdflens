# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import sdfio
import x3pio
from PySide6.QtCore import QDir, QEventLoop, QPoint, QPointF, QSize, QStandardPaths, Qt, QTimer
from PySide6.QtGui import QGuiApplication, QIcon, QImage, QKeySequence, QPalette
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QToolBar,
)

from helpers import COLS, ROWS, make_cloud, make_ramp, make_sdf, make_x3p
from sdflens import app as app_module
from sdflens.aboutdialog import AboutDialog
from sdflens.camera import Tool
from sdflens.colorbar import ColorBar
from sdflens.colormap import build_lut
from sdflens.exportdialog import SaveOptionsDialog
from sdflens.exporter import Exporter
from sdflens.glwidget import RenderMode, SurfaceView, _primitives
from sdflens.heatmap import HeatmapView, _block_size, _reduce
from sdflens.icons import load_icon
from sdflens.infopanel import format_metadata, metadata_rows
from sdflens.loader import Loader
from sdflens.mainwindow import MainWindow
from sdflens.mesh import SurfaceMesh
from sdflens.profileview import ProfileView, envelope
from sdflens.surface import SurfaceModel
from sdflens.zscalebar import ZScaleBar

pytestmark = pytest.mark.usefixtures("qapp")


def _wait_for(*signals: object, timeout_ms: int = 10_000) -> None:
    loop = QEventLoop()
    for signal in signals:
        signal.connect(loop.quit)  # type: ignore[attr-defined]
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()


def test_heatmap_puts_row_zero_at_the_bottom_and_marks_nan() -> None:
    model = SurfaceModel.from_sdf(make_sdf(make_ramp()))
    view = HeatmapView()
    view.set_lut(build_lut("Grayscale"))
    view.set_model(model)
    image = view._image

    assert image is not None
    assert (image.width(), image.height()) == (COLS, ROWS)
    bottom_left = image.pixelColor(0, ROWS - 1)
    top_right = image.pixelColor(COLS - 1, 0)
    assert bottom_left.red() == 0  # lowest value: row 0, column 0
    assert top_right.red() > 200  # highest value: last row, last column
    nan_pixel = image.pixelColor(COLS - 1, ROWS - 1)  # NaN at row 0, last column
    assert nan_pixel.alpha() == 0


def test_heatmap_blocks_are_powers_of_two_and_only_for_small_points() -> None:
    assert _block_size(1.0, 1000) == 1
    assert _block_size(0.5, 1000) == 1
    assert _block_size(0.4, 1000) == 2
    assert _block_size(0.01, 1000) == 64


def test_heatmap_blocks_never_exceed_the_points_of_an_axis() -> None:
    assert _block_size(0.001, 1) == 1  # a profile has a single row
    assert _block_size(0.001, 30) == 16


def test_reduce_averages_measured_points_per_block() -> None:
    data = np.array([[1.0, 3.0, np.nan], [np.nan, np.nan, 5.0]])
    reduced = _reduce(data, 2, 2)

    assert reduced.shape == (1, 2)
    assert reduced[0, 0] == pytest.approx(2.0)  # mean of the measured 1 and 3
    assert reduced[0, 1] == pytest.approx(5.0)  # the padded block keeps its single column
    assert np.isnan(_reduce(np.full((2, 2), np.nan), 2, 2)).all()
    np.testing.assert_array_equal(_reduce(data, 1, 1), data)


def test_heatmap_averages_a_very_wide_image_along_x_only() -> None:
    model = SurfaceModel.from_sdf(make_sdf(np.zeros((3, 70000)), x_scale=1e-7, y_scale=1e-3))
    view = HeatmapView()
    view.set_lut(build_lut("Viridis"))
    view.set_model(model)
    view.resize(700, 400)
    view.grab()

    (block_y, block_x), image = list(view._levels.items())[-1]
    assert block_y == 1
    assert block_x >= 32
    assert image.height() == 3
    assert image.width() == -(-70000 // block_x)


def test_envelope_reports_low_and_high_per_column_and_skips_gaps() -> None:
    values = np.array([1.0, 3.0, np.nan, np.nan, 2.0, 8.0])
    columns, low, high = envelope(values, 0.0, 6.0, 3)

    assert columns.tolist() == [0, 2]  # the middle column has no measured point
    assert low.tolist() == [1.0, 2.0]
    assert high.tolist() == [3.0, 8.0]
    columns, low, high = envelope(values, 0.0, 6.0, 12)  # more columns than points
    assert columns.tolist() == [0, 2, 8, 10]
    assert envelope(values, 10.0, 12.0, 4)[0].size == 0  # right of the data


def test_profile_view_selects_profiles_and_columns() -> None:
    model = SurfaceModel.from_sdf(make_sdf(make_ramp(), x_scale=1e-6, y_scale=2e-6))
    view = ProfileView()
    view.set_model(model)

    assert view._spin.maximum() == ROWS
    np.testing.assert_allclose(view._values, make_ramp()[0], equal_nan=True)
    view._spin.setValue(3)
    np.testing.assert_allclose(view._values, make_ramp()[2])
    view._direction.setCurrentIndex(1)
    assert view._spin.maximum() == COLS
    assert view._spin.value() == 1
    np.testing.assert_allclose(view._values, make_ramp()[:, 0])
    assert view._step == pytest.approx(2e-6)


def test_profile_view_cannot_turn_a_single_profile() -> None:
    view = ProfileView()
    view.set_model(SurfaceModel.from_sdf(make_sdf(np.array([[0.0, 1e-9, 2e-9]]))))

    assert _enabled(view._direction) == ["Along x"]
    assert not view._spin.isEnabled()


def test_profile_view_reports_the_point_under_the_cursor() -> None:
    view = ProfileView()
    view.set_model(SurfaceModel.from_sdf(make_sdf(make_ramp())))
    texts: list[str] = []
    view.hovered.connect(texts.append)
    view._plot.point_hovered.emit(1)
    view._plot.point_hovered.emit(COLS - 1)
    view._plot.point_hovered.emit(-1)

    assert "z = 1 nm" in texts[0]
    assert "profile 1, point 2" in texts[0]
    assert "not measured" in texts[1]
    assert texts[2] == ""


def test_profile_plot_zooms_and_pans_within_the_data() -> None:
    view = ProfileView()
    view.set_model(SurfaceModel.from_sdf(make_sdf(np.zeros((2, 100)))))
    view.resize(500, 300)
    plot = view._plot
    full, step = plot._span, plot._step

    plot.zoom_about_center(2.0)
    assert plot._span == pytest.approx(full / 2)
    plot.zoom_about_center(1e9)
    assert plot._span == pytest.approx(3 * step)  # closest zoom shows three points
    plot.zoom_about_center(1e-9)
    assert plot._span == pytest.approx(full)
    assert plot._x0 == pytest.approx(-0.5 * step)
    view.home()
    assert plot._span == pytest.approx(full)


def _blue_pixels(image: QImage) -> int:
    count = 0
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            if abs(color.red() - 47) < 25 and abs(color.green() - 111) < 25 and color.blue() > 150:
                count += 1
    return count


@pytest.mark.parametrize("count", [5, 5000])
def test_profile_view_draws_points_and_the_envelope(count: int) -> None:
    data = np.random.default_rng(1).random((2, count)) * 1e-6
    view = ProfileView()
    view.set_model(SurfaceModel.from_sdf(make_sdf(data)))
    view.resize(500, 300)

    assert _blue_pixels(view.grab_image()) > 50


def test_profile_mode_hides_the_color_bar_and_captures_only_the_plot(ramp_path: Path) -> None:
    window = MainWindow()
    _load(window, ramp_path)
    window._view_profile_action.trigger()

    assert window._stack.currentWidget() is window._profile_view
    assert window._colorbar.isHidden()
    assert not window._z_bar.isEnabled()
    image = window._capture()
    assert image is not None
    assert image.width() == window._profile_view.grab_image().width()
    window._view_3d_action.trigger()
    assert not window._colorbar.isHidden()


def test_heatmap_hover_reports_value_and_not_measured() -> None:
    model = SurfaceModel.from_sdf(make_sdf(make_ramp()))
    view = HeatmapView()
    view.set_lut(build_lut("Viridis"))
    view.set_model(model)
    view.resize(600, 400)
    origin = view._plot_rect().topLeft()

    def text_at(row: int, col: int) -> str:
        sx, sy = view._transform.world_to_screen(col * model.x_scale, row * model.pixel_size_y)
        return view._describe(QPointF(sx + origin.x(), sy + origin.y()))

    assert "not measured" in text_at(0, COLS - 1)
    assert "not measured" not in text_at(1, 1)
    assert "(profile 2, point 2)" in text_at(1, 1)
    assert text_at(1, 1).startswith("x = 1 \u00b5m, y = 2 \u00b5m, z = ")
    assert view._describe(QPointF(0.0, 0.0)) == ""


def test_colorbar_paints_with_and_without_range() -> None:
    model = SurfaceModel.from_sdf(make_sdf(make_ramp()))
    bar = ColorBar()
    bar.resize(bar.width(), 300)
    bar.grab()
    bar.set_lut(build_lut("Inferno"))
    bar.set_range(model.value_range, has_invalid=True)

    assert not bar.grab().isNull()


def test_metadata_lists_header_fields_only() -> None:
    rows = dict(metadata_rows(make_sdf(make_ramp())))

    assert list(rows) == [
        "Version",
        "ManufacID",
        "CreateDate",
        "ModDate",
        "NumPoints",
        "NumProfiles",
        "Xscale",
        "Yscale",
        "Zscale",
        "Zresolution",
        "DataType",
    ]
    assert rows["Version"] == "bISO-2.0"
    assert rows["NumPoints"] == str(COLS)
    assert rows["Zresolution"] == "-1"


def test_metadata_text_includes_the_trailer(tmp_path: Path) -> None:
    path = tmp_path / "trailer.sdf"
    sdfio.write(path, make_ramp(), x_scale=1e-6, y_scale=1e-6, trailer="OperatorName = Jane\r\n")
    text = format_metadata(sdfio.read(path))

    assert text.startswith("Version")
    assert "Trailer" in text
    assert text.rstrip().endswith("OperatorName = Jane")
    assert "Trailer" not in format_metadata(make_sdf(make_ramp()))


def test_loader_reads_in_the_background(ramp_path: Path) -> None:
    loader = Loader()
    results: list[tuple[str, object, object, object, bool]] = []
    loader.loaded.connect(lambda *args: results.append(args))

    assert loader.load(str(ramp_path))
    assert loader.busy
    assert not loader.load(str(ramp_path))
    _wait_for(loader.loaded)

    assert not loader.busy
    assert len(results) == 1
    assert isinstance(results[0][2], SurfaceModel)
    assert results[0][2].invalid_count == 1
    assert isinstance(results[0][3], SurfaceMesh)
    assert results[0][4] is False


def test_x3p_with_a_wrong_checksum_is_shown_with_a_note(tmp_path: Path) -> None:
    path = tmp_path / "edited.x3p"
    data = make_x3p().dumps()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(path, "w") as target:
        for info in source.infolist():
            content = source.read(info)
            if info.filename == "main.xml":
                content = content.replace(b"Jane Doe", b"Joe Doe")  # changed after it was written
            target.writestr(info, content)
    with pytest.raises(x3pio.X3pChecksumError):
        x3pio.read(path)
    window = MainWindow()
    _load(window, path)

    assert isinstance(window._model, SurfaceModel)
    assert "checksum" in window.statusBar().currentMessage()
    window.close()


def test_loader_reports_missing_file(tmp_path: Path) -> None:
    loader = Loader()
    failures: list[tuple[str, str]] = []
    loader.failed.connect(lambda path, message: failures.append((path, message)))
    loader.load(str(tmp_path / "missing.sdf"))
    _wait_for(loader.failed)

    assert len(failures) == 1
    assert not loader.busy


def _load(window: MainWindow, path: Path) -> None:
    window.load_file(str(path))
    _wait_for(window._loader.loaded)


def test_toolbars_are_fixed_and_cannot_be_hidden() -> None:
    window = MainWindow()
    toolbars = window.findChildren(QToolBar)

    assert len(toolbars) == 2
    assert not any(bar.isMovable() for bar in toolbars)
    assert window._save_as_action in toolbars[0].actions() + toolbars[1].actions()
    assert not window._save_as_action.icon().isNull()
    assert window.createPopupMenu() is None


def test_about_dialog_shows_the_generated_notices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notices = tmp_path / "notices.txt"
    notices.write_text("Licensed under \u00a7 MIT", encoding="utf-8")
    monkeypatch.setattr("sdflens.aboutdialog._NOTICES", notices)

    assert AboutDialog().text.toPlainText() == "Licensed under \u00a7 MIT"


def test_about_dialog_explains_missing_notices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sdflens.aboutdialog._NOTICES", tmp_path / "missing.txt")

    assert "generated" in AboutDialog().text.toPlainText()


def test_main_window_loads_file_and_switches_views(ramp_path: Path) -> None:
    window = MainWindow()
    window.load_file(str(ramp_path))
    assert not window._open_action.isEnabled()  # busy while loading
    _wait_for(window._loader.loaded)

    assert window._model is not None
    assert window._open_action.isEnabled()
    assert window.windowTitle() == "ramp.sdf"
    assert window._info_panel.table.rowCount() == 11
    assert window._stats_label.text() == "size = 5 \u00d7 8 \u00b5m"
    assert window._path == QDir.toNativeSeparators(str(ramp_path))
    assert window.statusBar().currentMessage() == "Loaded ramp.sdf"

    window._view_2d_action.trigger()
    assert window._stack.currentWidget() is window._heatmap_view
    assert not window._z_bar.isEnabled()
    assert not window._rotate_action.isEnabled()
    assert window._pan_action.isChecked()
    window._view_3d_action.trigger()
    assert window._stack.currentWidget() is window._surface_view
    assert window._z_bar.isEnabled()
    assert window._rotate_action.isChecked()  # rotate is the default 3D tool
    assert window._surface_view._tool is Tool.ROTATE


def test_toolbar_tooltips_name_the_action_and_its_shortcut() -> None:
    window = MainWindow()
    for toolbar in window.findChildren(QToolBar):
        for action in toolbar.actions():
            shortcut = action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
            if shortcut:
                assert action.toolTip().endswith(f"({shortcut})"), action.text()
    assert window._open_action.toolTip().startswith("Open (")
    assert window._screenshot_action.toolTip().startswith("Save screenshot (")
    window.close()


def test_x3p_surface_shows_its_metadata_and_vendor_extensions(tmp_path: Path) -> None:
    path = tmp_path / "surface.x3p"
    make_x3p().save(path)
    window = MainWindow()
    _load(window, path)
    panel = window._info_panel
    names = [
        item.text()
        for item in (panel.table.item(row, 0) for row in range(panel.table.rowCount()))
        if item is not None
    ]

    assert "Revision" in names
    assert "MatrixDimension.SizeX" in names
    assert "Creator" in names
    assert not panel.extensions.isHidden()
    assert panel.trailer.isHidden()
    assert [panel.extensions.item(i).text() for i in range(panel.extensions.count())] == [
        "http://www.vendor.com/mypath/a.xml",
        "http://www.vendor.com/image.png",
    ]
    assert window._view_2d_action.isEnabled()
    assert window._view_profile_action.isEnabled()
    assert window._save_as_action.isEnabled()  # an x3p file can be saved as x3p or SDF
    assert window._export_action.isEnabled()
    window.close()


def test_export_and_copy_of_an_x3p_file_act_on_the_information(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "surface.x3p"
    make_x3p().save(path)
    window = MainWindow()
    _load(window, path)
    target = tmp_path / "out.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(target), ""))

    assert window._export_action.isEnabled()
    assert window._copy_metadata_action.isEnabled()
    window._export_action.trigger()
    text = target.read_text(encoding="utf-8")
    assert "FeatureType" in text
    assert "http://www.vendor.com/mypath/a.xml" in text  # the extensions are listed

    QGuiApplication.clipboard().clear()
    window._copy_metadata_action.trigger()
    assert QGuiApplication.clipboard().text() == text
    window.close()


def test_double_click_on_a_vendor_extension_saves_its_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "surface.x3p"
    make_x3p().save(path)
    window = MainWindow()
    _load(window, path)
    asked: list[str] = []
    target = tmp_path / "saved.xml"

    def fake_save(parent: object, title: str, start: str, name_filter: str) -> tuple[str, str]:
        asked.append(Path(start).name)
        return str(target), ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", fake_save)
    extensions = window._info_panel.extensions
    item = extensions.item(0)
    assert item is not None
    assert extensions.selectionMode() is QAbstractItemView.SelectionMode.SingleSelection
    extensions.setCurrentItem(item)  # a click selects the entry, to show that it is clickable
    assert extensions.selectedItems() == [item]
    extensions.itemDoubleClicked.emit(item)

    assert asked == ["a.xml"]  # the last part of the ID
    assert target.read_bytes() == b"<a/>"

    target.unlink()
    asked.clear()
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: ("", ""))
    extensions.itemDoubleClicked.emit(item)  # cancelling the dialog writes nothing
    assert not target.exists()
    window.close()


def test_export_and_copy_of_an_sdf_file_act_on_the_information(ramp_path: Path) -> None:
    window = MainWindow()
    _load(window, ramp_path)

    assert window._export_action.isEnabled()
    assert window._copy_metadata_action.isEnabled()
    window.close()


def test_point_cloud_has_no_2d_or_profile_view(ramp_path: Path, tmp_path: Path) -> None:
    cloud_path = tmp_path / "cloud.x3p"
    x3pio.write_points(cloud_path, make_cloud())
    window = MainWindow()
    window._choose_view("2d")
    _load(window, ramp_path)
    assert window._current_mode() == "2d"

    _load(window, cloud_path)
    assert window._current_mode() == "3d"
    assert not window._view_2d_action.isEnabled()
    assert not window._view_profile_action.isEnabled()
    assert not window._render_menu.isEnabled()
    assert "200 points" in window._stats_label.text()
    assert window._view_preference == "2d"  # the point cloud does not change the preference

    _load(window, ramp_path)  # a grid again: the preferred view and the menu come back
    assert window._current_mode() == "2d"
    assert window._view_2d_action.isEnabled()
    window._choose_view("3d")
    assert window._render_menu.isEnabled()
    assert window._save_as_action.isEnabled()
    window.close()


def test_open_dialog_offers_x3p_files(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake(parent: object, title: str, directory: str, name_filter: str) -> tuple[str, str]:
        seen.append(name_filter)
        return "", ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", fake)
    window = MainWindow()
    window._open_dialog()

    assert "*.sdf *.x3p" in seen[0]
    window.close()


def test_information_dock_shows_the_header_and_can_be_toggled(ramp_path: Path) -> None:
    window = MainWindow()
    window.show()
    assert window._info_dock.isVisible()
    _load(window, ramp_path)

    assert window._info_panel.table.rowCount() == 11
    assert window._export_action.isEnabled()
    window._info_toggle_action.trigger()
    assert not window._info_dock.isVisible()
    window._info_toggle_action.trigger()
    assert window._info_dock.isVisible()
    window.close()


def test_information_table_has_no_header_and_fits_its_rows(ramp_path: Path) -> None:
    window = MainWindow()
    _load(window, ramp_path)
    table = window._info_panel.table

    assert table.horizontalHeader().isHidden()
    assert table.selectionMode() == QAbstractItemView.SelectionMode.NoSelection
    content = table.verticalHeader().length() + 2 * table.frameWidth()
    assert table.sizeHint().height() >= content  # a short table is shown completely
    assert table.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Fixed
    assert table.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff


def test_file_dialogs_start_in_documents_and_remember_the_last_folder(
    ramp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    starts: list[str] = []

    def fake_open(*args: object) -> tuple[str, str]:
        starts.append(str(args[2]))
        return str(ramp_path), ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", fake_open)
    window = MainWindow()
    window._settings.last_directory = ""
    for _ in range(2):
        window._open_dialog()
        _wait_for(window._loader.loaded)

    documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
    assert starts[0] == documents
    assert Path(starts[1]) == ramp_path.parent


def test_render_modes_pick_the_primitives_to_draw() -> None:
    mesh = SurfaceMesh.from_model(SurfaceModel.from_sdf(make_sdf(make_ramp())))

    assert _primitives(mesh, RenderMode.SURFACE)[1] is mesh.triangles
    assert _primitives(mesh, RenderMode.WIREFRAME)[1] is mesh.lines
    assert mesh.lines.size > 0
    points = _primitives(mesh, RenderMode.POINTS)[1]
    assert len(points) == int(mesh.valid.sum())
    assert mesh.valid.ravel()[points].all()


def test_clicking_a_gizmo_axis_aligns_the_view_to_it() -> None:
    model = SurfaceModel.from_sdf(make_sdf(make_ramp()))
    view = SurfaceView()
    view.resize(400, 300)
    view.set_surface(model, SurfaceMesh.from_model(model))
    home = (view._camera.azimuth, view._camera.elevation)

    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPoint(250, 100))
    assert (view._camera.azimuth, view._camera.elevation) == home  # not on the gizmo
    for axis, (azimuth, elevation) in {"x": (0, 0), "y": (90, 0), "z": (-90, 89)}.items():
        view._camera.fit(1.0)
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=view._gizmo_tips()[axis].toPoint())
        assert view._camera.azimuth == pytest.approx(azimuth)
        assert view._camera.elevation == pytest.approx(elevation)


def test_render_mode_is_exclusive_stored_and_only_for_the_3d_view() -> None:
    window = MainWindow()
    window._render_actions[RenderMode.WIREFRAME].trigger()

    assert window._surface_view._render_mode is RenderMode.WIREFRAME
    assert window._render_actions[RenderMode.WIREFRAME].isChecked()
    assert not window._render_actions[RenderMode.SURFACE].isChecked()
    window._view_2d_action.trigger()
    assert not window._render_menu.isEnabled()
    window._view_3d_action.trigger()
    assert window._render_menu.isEnabled()
    window.close()

    restored = MainWindow()
    assert restored._render_mode is RenderMode.WIREFRAME
    restored._render_actions[RenderMode.SURFACE].trigger()
    restored.close()


def test_save_and_copy_screenshot_of_2d_view(
    ramp_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "shot.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(target), ""))
    window = MainWindow()
    _load(window, ramp_path)
    window._view_2d_action.trigger()
    window._save_screenshot()
    window._copy_screenshot()

    assert target.stat().st_size > 0
    assert not QGuiApplication.clipboard().image().isNull()


def test_export_and_copy_metadata(
    ramp_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "meta.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(target), ""))
    window = MainWindow()
    _load(window, ramp_path)
    window._export_metadata()
    window._copy_metadata()

    assert "ManufacID" in target.read_text(encoding="utf-8")
    assert QGuiApplication.clipboard().text() == target.read_text(encoding="utf-8")


def test_main_window_reports_load_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    messages: list[tuple[object, ...]] = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: messages.append(args))
    window = MainWindow()
    window.load_file(str(tmp_path / "missing.sdf"))
    _wait_for(window._loader.failed)

    assert len(messages) == 1
    assert window._model is None
    assert QApplication.overrideCursor() is None


def test_icon_uses_the_text_color_also_when_checked() -> None:
    icon = load_icon("rotate")
    size = QSize(24, 24)

    normal = icon.pixmap(size, QIcon.Mode.Normal, QIcon.State.Off).toImage()
    checked = icon.pixmap(size, QIcon.Mode.Normal, QIcon.State.On).toImage()
    disabled = icon.pixmap(size, QIcon.Mode.Disabled, QIcon.State.Off).toImage()
    text = QGuiApplication.palette().color(QPalette.ColorRole.ButtonText).rgb()

    assert normal == checked
    assert normal != disabled
    assert any(
        normal.pixelColor(x, y).rgb() == text
        for x in range(size.width())
        for y in range(size.height())
    )


def test_z_scale_icon_is_drawn_from_the_svg() -> None:
    icon = ZScaleBar().findChild(QSvgWidget)

    assert icon is not None
    assert icon.renderer().isValid()


def test_z_controls_stay_in_sync() -> None:
    window = MainWindow()
    window._z_bar._spin.setValue(50.0)

    assert window._z_bar.factor == pytest.approx(50.0)
    assert window._surface_view._z_factor == pytest.approx(50.0)
    assert window._z_bar._slider.value() > 0


def test_a_loaded_file_starts_at_true_scale_until_auto_is_requested(ramp_path: Path) -> None:
    window = MainWindow()
    window._z_bar.set_factor(7.0)
    _load(window, ramp_path)

    assert window._z_bar.factor == 1.0
    assert window._model is not None
    window._z_bar._auto.trigger()
    assert window._z_bar.factor == pytest.approx(window._model.auto_z_factor())
    assert window._z_bar.factor > 1.0


def test_command_line_takes_only_an_optional_file_name(capsys: pytest.CaptureFixture[str]) -> None:
    parser = app_module._build_parser()

    assert parser.parse_args([]).path is None
    assert parser.parse_args(["surface.sdf"]).path == "surface.sdf"
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--version"])
    assert exit_info.value.code == 2
    capsys.readouterr()


def test_copy_metadata_contains_header_fields_and_trailer(tmp_path: Path) -> None:
    path = tmp_path / "trailer.sdf"
    sdfio.write(path, make_ramp(), x_scale=1e-6, y_scale=1e-6, trailer="OperatorName = Jane\r\n")
    window = MainWindow()
    _load(window, path)
    window._copy_metadata()
    text = QGuiApplication.clipboard().text()

    assert "ManufacID" in text
    assert "Xscale" in text
    assert text.rstrip().endswith("OperatorName = Jane")
    assert window._info_panel.trailer.toPlainText().strip() == "OperatorName = Jane"


def test_screenshot_includes_the_color_bar(ramp_path: Path) -> None:
    window = MainWindow()
    _load(window, ramp_path)
    window._view_2d_action.trigger()
    image = window._capture()

    assert image is not None
    view_width = window._heatmap_view.grab().width()
    assert image.width() == view_width + window._colorbar.grab().width()
    assert window._colorbar.width() > 0


def test_file_menu_groups_screenshot_and_information_actions() -> None:
    window = MainWindow()
    file_menu = window.menuBar().actions()[0].menu()
    assert isinstance(file_menu, QMenu)
    submenus: dict[str, QMenu] = {}
    for action in file_menu.actions():
        menu = action.menu()
        if isinstance(menu, QMenu):
            submenus[action.text()] = menu

    assert set(submenus) == {"&Screenshot", "&Information"}
    assert [a.text() for a in submenus["&Screenshot"].actions()] == [
        "&Save...",
        "&Copy",
    ]
    assert [a.text() for a in submenus["&Information"].actions()] == [
        "&Export...",
        "&Copy",
    ]
    assert window._quit_action.shortcuts()


def _enabled(combo: QComboBox) -> list[str]:
    return [
        combo.itemText(i)
        for i in range(combo.count())
        if combo.model().flags(combo.model().index(i, 0)) & Qt.ItemFlag.ItemIsEnabled
    ]


def test_options_dialog_presets_from_the_sdf_file_and_disables_unsupported_data_types(
    ramp_path: Path,
) -> None:
    dialog = SaveOptionsDialog(sdfio.read(ramp_path), is_x3p=False)

    assert dialog.version_name == "ISO-2.0"
    assert dialog.encoding_name == "Binary"
    assert dialog.file_format is sdfio.FileFormat.BINARY
    assert dialog.dialect is sdfio.SdfDialect.ISO_2_0
    assert dialog.data_type is sdfio.DataType.BINARY64
    assert len(_enabled(dialog._data_type)) == 5
    dialog._version.setCurrentText("ISO-1.0")
    assert _enabled(dialog._data_type) == ["int16", "int32", "binary64"]
    dialog._data_type.setCurrentText("int8")
    dialog._version.setCurrentText("BCR-1.0")
    assert dialog.data_type.name == "INT8"
    dialog._version.setCurrentText("ISO-1.0")
    assert dialog.data_type.name == "INT16"  # INT8 isn't defined for ISO-1.0


def test_options_dialog_offers_unsigned_data_types_for_bcr_only(ramp_path: Path) -> None:
    dialog = SaveOptionsDialog(sdfio.read(ramp_path), is_x3p=False)
    unsigned = ["uint8", "uint16", "uint32"]

    assert not set(unsigned) & set(_enabled(dialog._data_type))
    dialog._version.setCurrentText("BCR-1.0")
    assert _enabled(dialog._data_type)[:3] == unsigned
    dialog._data_type.setCurrentText("uint16")
    assert dialog.data_type is sdfio.DataType.UINT16
    dialog._version.setCurrentText("ISO-2.0")
    assert dialog.data_type_name not in unsigned


def test_options_dialog_uses_the_last_options_when_valid(ramp_path: Path) -> None:
    dialog = SaveOptionsDialog(
        sdfio.read(ramp_path), is_x3p=False, last=("BCR-1.0", "ASCII", "int16")
    )

    assert dialog.dialect is sdfio.SdfDialect.BCR_1_0
    assert dialog.file_format is sdfio.FileFormat.ASCII
    assert dialog.data_type_name == "int16"


def test_options_dialog_ignores_last_options_it_does_not_know(ramp_path: Path) -> None:
    dialog = SaveOptionsDialog(
        sdfio.read(ramp_path), is_x3p=False, last=("nonsense", "nonsense", "nonsense")
    )

    assert dialog.version_name == "ISO-2.0"
    assert dialog.encoding_name == "Binary"
    assert dialog.data_type_name == "binary64"


def _ok(dialog: SaveOptionsDialog) -> QPushButton:
    button = dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert button is not None
    return button


def test_options_dialog_cannot_be_accepted_for_a_large_grid_in_a_small_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wide.sdf"
    sdfio.write(path, np.zeros((2, 70000)), x_scale=1e-6, y_scale=1e-6)
    dialog = SaveOptionsDialog(sdfio.read(path), is_x3p=False)

    assert _ok(dialog).isEnabled()
    dialog._version.setCurrentText("ISO-1.0")
    # binary 1.0 versions store at most 65535 points per row
    assert not _ok(dialog).isEnabled()
    assert "65535" in dialog._note.text()
    dialog._encoding.setCurrentText("ASCII")
    assert _ok(dialog).isEnabled()
    assert dialog._note.text() == ""


def test_options_dialog_cannot_be_accepted_for_an_untagged_trailer(tmp_path: Path) -> None:
    path = tmp_path / "bcr.sdf"
    metadata = sdfio.SdfMetadata(dialect=sdfio.SdfDialect.BCR_1_0)
    sdfio.write(path, make_ramp(), x_scale=1e-6, y_scale=1e-6, metadata=metadata, trailer="free")
    dialog = SaveOptionsDialog(sdfio.read(path), is_x3p=False)

    assert dialog.version_name == "BCR-1.0"
    assert _ok(dialog).isEnabled()
    dialog._version.setCurrentText("ISO-2.0")
    assert not _ok(dialog).isEnabled()
    assert "tagged" in dialog._note.text()


def test_options_dialog_selects_the_first_enabled_data_type_when_the_version_changes(
    ramp_path: Path,
) -> None:
    dialog = SaveOptionsDialog(sdfio.read(ramp_path), is_x3p=False)
    dialog._data_type.setCurrentText("int8")
    dialog._version.setCurrentText("ISO-1.0")

    assert dialog.data_type_name == "int16"
    assert _enabled(dialog._data_type) == ["int16", "int32", "binary64"]


def test_save_as_remembers_the_chosen_options_per_format(
    ramp_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    _load(window, ramp_path)
    monkeypatch.setattr(
        SaveOptionsDialog, "exec", lambda self: SaveOptionsDialog.DialogCode.Accepted
    )
    monkeypatch.setattr(SaveOptionsDialog, "version_name", property(lambda self: "BCR-1.0"))
    monkeypatch.setattr(SaveOptionsDialog, "encoding_name", property(lambda self: "ASCII"))
    monkeypatch.setattr(SaveOptionsDialog, "data_type_name", property(lambda self: "int16"))
    target = tmp_path / "again.sdf"
    _fake_file_dialog(monkeypatch, target, "Surface data file (*.sdf)")
    window._save_as()

    settings = window._settings
    assert (settings.save_sdf_version, settings.save_sdf_encoding) == ("BCR-1.0", "ASCII")
    assert settings.save_sdf_data_type == "int16"
    assert settings.save_x3p_version == ""  # the other format keeps its own choice
    window.close()


def test_save_as_writes_the_chosen_version_format_and_type(ramp_path: Path, tmp_path: Path) -> None:
    window = MainWindow()
    _load(window, ramp_path)
    target = tmp_path / "converted.sdf"
    window._exporter.write_export(
        sdfio.read(ramp_path),
        str(target),
        sdfio.SdfDialect.BCR_1_0,
        sdfio.DataType.INT16,
        sdfio.FileFormat.ASCII,
    )
    loaded = sdfio.read(target)

    assert loaded.header.dialect is sdfio.SdfDialect.BCR_1_0
    assert not loaded.header.binary
    assert loaded.data_type is sdfio.DataType.INT16
    assert np.array_equal(np.isnan(loaded.data), np.isnan(make_ramp()))


def test_save_as_reports_failures(
    ramp_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages: list[tuple[object, ...]] = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: messages.append(args))
    window = MainWindow()
    _load(window, ramp_path)
    window._exporter.write_export(
        sdfio.read(ramp_path),
        str(tmp_path / "missing-dir" / "out.sdf"),
        sdfio.SdfDialect.ISO_2_0,
        sdfio.DataType.BINARY64,
        sdfio.FileFormat.BINARY,
    )

    assert len(messages) == 1
    assert QApplication.overrideCursor() is None


def test_options_dialog_for_x3p_offers_its_versions_encodings_and_data_types(
    ramp_path: Path,
) -> None:
    dialog = SaveOptionsDialog(sdfio.read(ramp_path), is_x3p=True)

    assert dialog.windowTitle() == "Save as x3p"
    assert _enabled(dialog._version) == ["ISO25178-72:2017/DAM1", "ISO5436 - 2000"]
    assert _enabled(dialog._encoding) == ["Binary", "XML"]
    assert _enabled(dialog._data_type) == ["int16", "int32", "float32", "float64"]
    assert dialog.dialect is x3pio.Revision.ISO25178_72_2017_DAM1  # the newest
    assert dialog.file_format is x3pio.DataStorage.BINARY
    assert dialog.data_type is x3pio.DataType.FLOAT64
    assert "comment" in dialog._note.text()  # what the conversion does to the trailer


def test_options_dialog_of_an_x3p_file_starts_with_its_own_options() -> None:
    x3p = make_x3p().with_z_type(x3pio.DataType.INT16)
    x3p.storage = x3pio.DataStorage.XML
    dialog = SaveOptionsDialog(x3p, is_x3p=True)

    assert dialog.file_format is x3pio.DataStorage.XML
    assert dialog.data_type is x3pio.DataType.INT16
    assert dialog._note.text() == ""


def test_options_dialog_says_what_an_sdf_file_of_an_x3p_file_loses() -> None:
    layers = np.stack([make_ramp(), make_ramp()])
    dialog = SaveOptionsDialog(
        x3pio.Surface.from_array(layers, x_scale=1e-6, y_scale=1e-6), is_x3p=False
    )

    assert dialog.dialect is sdfio.SdfDialect.ISO_2_0
    assert "vendor extensions" in dialog._note.text()
    assert "Only layer 1 of 2 is saved" in dialog._note.text()
    assert _ok(dialog).isEnabled()
    chosen = SaveOptionsDialog(
        x3pio.Surface.from_array(layers, x_scale=1e-6, y_scale=1e-6), is_x3p=False, layer=1
    )
    assert "Only layer 2 of 2 is saved" in chosen._note.text()


def test_options_dialog_warns_when_the_vendor_extensions_do_not_fit() -> None:
    dialog = SaveOptionsDialog(make_x3p(), is_x3p=True)  # two IDs

    assert dialog._note.text() == ""
    dialog._version.setCurrentText("ISO5436 - 2000")
    assert "vendor extensions" in dialog._note.text()
    assert _ok(dialog).isEnabled()


def test_options_dialog_remembers_an_x3p_choice() -> None:
    dialog = SaveOptionsDialog(
        make_sdf(make_ramp()), is_x3p=True, last=("ISO5436 - 2000", "XML", "float32")
    )

    assert dialog.dialect is x3pio.Revision.ISO5436_2000
    assert dialog.file_format is x3pio.DataStorage.XML
    assert dialog.data_type is x3pio.DataType.FLOAT32


def _fake_file_dialog(
    monkeypatch: pytest.MonkeyPatch, path: Path, selected: str
) -> list[tuple[str, list[tuple[str, str]], str]]:
    seen: list[tuple[str, list[tuple[str, str]], str]] = []

    def fake(
        self: Exporter, name: str, filters: list[tuple[str, str]], default_filter: str
    ) -> tuple[str, str]:
        seen.append((name, filters, default_filter))
        return str(path), selected

    monkeypatch.setattr(Exporter, "_ask_save_as", fake)
    return seen


def test_save_as_asks_for_the_file_first_with_a_filter_for_each_format(
    ramp_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    _load(window, ramp_path)
    seen = _fake_file_dialog(monkeypatch, tmp_path / "out", "x3p file (*.x3p)")
    shown: list[bool] = []

    def exec_options(self: SaveOptionsDialog) -> QDialog.DialogCode:
        shown.append(self.windowTitle() == "Save as x3p")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(SaveOptionsDialog, "exec", exec_options)
    window._save_as()

    name, filters, initial = seen[0]
    assert name == "ramp.sdf"  # SDF is the default type
    assert filters == [("Surface data file (*.sdf)", ".sdf"), ("x3p file (*.x3p)", ".x3p")]
    assert initial == "Surface data file (*.sdf)"
    assert shown == [True]  # the filter chose x3p, so its options are asked
    assert (tmp_path / "out.x3p").exists()  # the missing extension is added
    window.close()


def test_save_as_saves_the_layer_that_is_shown_when_the_dialog_asks_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    _load(window, _layered_path(tmp_path, make_ramp(), 10.0 * make_ramp(), 100.0 * make_ramp()))
    _choose_layer(window, 3)
    answers = [False, True]  # the first save keeps all layers, the second only the one shown

    def exec_options(self: SaveOptionsDialog) -> QDialog.DialogCode:
        self._single_layer.setChecked(answers.pop(0))
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(SaveOptionsDialog, "exec", exec_options)
    for name in ("all", "one"):
        _fake_file_dialog(monkeypatch, tmp_path / name, "x3p file (*.x3p)")
        window._save_as()

    assert len(x3pio.read(tmp_path / "all.x3p").layers) == 3
    chosen = x3pio.read(tmp_path / "one.x3p")
    assert len(chosen.layers) == 1
    np.testing.assert_allclose(chosen.layer.z, 100.0 * make_ramp(), atol=1e-7, equal_nan=True)
    window.close()


def test_save_as_of_an_x3p_file_still_starts_with_the_sdf_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "surface.x3p"
    make_x3p().save(path)
    window = MainWindow()
    _load(window, path)
    seen = _fake_file_dialog(monkeypatch, tmp_path / "out", "")
    monkeypatch.setattr(SaveOptionsDialog, "exec", lambda self: QDialog.DialogCode.Rejected)
    window._save_as()

    name, filters, initial = seen[0]
    assert name == "surface.sdf"
    assert initial == "Surface data file (*.sdf)"
    assert [text for text, _suffix in filters] == [
        "Surface data file (*.sdf)",
        "x3p file (*.x3p)",
    ]
    window.close()


def test_changing_the_file_type_in_the_save_dialog_changes_the_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    window._settings.last_directory = str(tmp_path)
    names: list[str] = []

    def drive() -> None:
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, QFileDialog)
        combo = dialog.findChild(QComboBox, "fileTypeCombo")
        assert isinstance(combo, QComboBox)
        names.append(Path(dialog.selectedFiles()[0]).name)
        combo.setCurrentIndex(1)  # the file type of x3p, chosen as the user does
        combo.activated.emit(1)
        names.append(Path(dialog.selectedFiles()[0]).name)
        combo.setCurrentIndex(0)
        combo.activated.emit(0)
        names.append(Path(dialog.selectedFiles()[0]).name)
        dialog.reject()

    QTimer.singleShot(200, drive)
    path, chosen = window._exporter._ask_save_as(
        "surface.sdf",
        [("SDF (*.sdf)", ".sdf"), ("x3p (*.x3p)", ".x3p")],
        "SDF (*.sdf)",
    )

    assert names == ["surface.sdf", "surface.x3p", "surface.sdf"]
    assert (path, chosen) == ("", "")  # cancelled
    window.close()


def test_save_as_takes_the_format_from_a_typed_extension(
    ramp_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    _load(window, ramp_path)
    _fake_file_dialog(monkeypatch, tmp_path / "typed.X3P", "Surface data file (*.sdf)")
    titles: list[str] = []

    def exec_options(self: SaveOptionsDialog) -> QDialog.DialogCode:
        titles.append(self.windowTitle())
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(SaveOptionsDialog, "exec", exec_options)
    window._save_as()

    assert titles == ["Save as x3p"]  # the typed extension wins over the selected filter
    assert not (tmp_path / "typed.X3P").exists()  # cancelling the options writes nothing
    window.close()


def test_save_as_offers_only_x3p_for_a_point_cloud_and_refuses_an_sdf_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cloud_path = tmp_path / "cloud.x3p"
    x3pio.write_points(cloud_path, make_cloud())
    messages: list[tuple[object, ...]] = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: messages.append(args))
    window = MainWindow()
    _load(window, cloud_path)
    seen = _fake_file_dialog(monkeypatch, tmp_path / "cloud.sdf", "x3p file (*.x3p)")
    window._save_as()

    assert seen[0][1] == [("x3p file (*.x3p)", ".x3p")]  # no SDF filter for a point cloud
    assert seen[0][0] == "cloud.x3p"
    assert len(messages) == 1  # a typed .sdf name cannot be honoured
    assert not (tmp_path / "cloud.sdf").exists()
    window.close()


def test_save_as_writes_x3p_from_sdf_and_sdf_from_x3p(ramp_path: Path, tmp_path: Path) -> None:
    window = MainWindow()
    _load(window, ramp_path)
    target = tmp_path / "converted.x3p"
    window._exporter.write_export(
        sdfio.read(ramp_path),
        str(target),
        x3pio.Revision.ISO5436_2000,
        x3pio.DataType.INT16,
        x3pio.DataStorage.XML,
    )
    loaded = x3pio.read(target)

    assert loaded.revision is x3pio.Revision.ISO5436_2000
    assert loaded.header.z.data_type is x3pio.DataType.INT16
    assert loaded.storage is x3pio.DataStorage.XML
    np.testing.assert_allclose(loaded.layer.z, make_ramp(), atol=1e-9, equal_nan=True)

    back = tmp_path / "back.sdf"
    window._exporter.write_export(
        loaded,
        str(back),
        sdfio.SdfDialect.ISO_2_0,
        sdfio.DataType.BINARY64,
        sdfio.FileFormat.BINARY,
    )
    np.testing.assert_allclose(sdfio.read(back).data, make_ramp(), atol=1e-9, equal_nan=True)
    window.close()


def test_save_as_reports_a_point_cloud_that_cannot_be_an_sdf_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages: list[tuple[object, ...]] = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: messages.append(args))
    window = MainWindow()
    window._exporter.write_export(
        x3pio.PointCloud.from_points(make_cloud()),
        str(tmp_path / "cloud.sdf"),
        sdfio.SdfDialect.ISO_2_0,
        sdfio.DataType.BINARY64,
        sdfio.FileFormat.BINARY,
    )

    assert len(messages) == 1
    assert not (tmp_path / "cloud.sdf").exists()
    window.close()


def test_an_irregular_surface_is_drawn_as_a_surface_in_3d_only(tmp_path: Path) -> None:
    rows, columns = make_ramp().shape
    x, y = np.meshgrid(np.arange(columns) * 1e-6, np.arange(rows) * 2e-6)
    x = x + 0.3e-6 * np.arange(rows)[:, np.newaxis]
    path = tmp_path / "irregular.x3p"
    x3pio.Surface.from_points(np.stack([x, y, make_ramp()], axis=-1)).save(path)
    window = MainWindow()
    _load(window, path)

    assert window._surface_view._mesh is not None
    assert not window._surface_view._mesh.is_cloud
    assert window._render_menu.isEnabled()  # surface, wireframe and points
    assert not window._view_2d_action.isEnabled()
    assert not window._view_profile_action.isEnabled()
    assert "points" not in window._stats_label.text()
    assert window._stats_label.text().startswith("size = ")

    cloud_path = tmp_path / "cloud.x3p"
    x3pio.write_points(cloud_path, make_cloud())
    _load(window, cloud_path)

    assert window._surface_view._mesh is not None
    assert window._surface_view._mesh.is_cloud
    assert not window._render_menu.isEnabled()
    assert window._stats_label.text().endswith("200 points")
    window.close()


def test_the_save_dialog_offers_a_single_layer_for_an_x3p_file_of_layers_only() -> None:
    layers = np.stack([make_ramp(), make_ramp() + 1.0, make_ramp() + 2.0])
    layered = x3pio.Surface.from_array(layers, x_scale=1e-6, y_scale=2e-6)

    dialog = SaveOptionsDialog(layered, is_x3p=True, layer=1)
    assert dialog._single_layer.isVisibleTo(dialog)
    assert dialog._single_layer.text() == "Save only layer 2 of 3"
    assert not dialog.single_layer  # everything is saved unless it is asked for
    dialog._single_layer.setChecked(True)
    assert dialog.single_layer

    assert not SaveOptionsDialog(layered, is_x3p=False)._single_layer.isVisibleTo(dialog)  # SDF
    assert not SaveOptionsDialog(make_x3p(), is_x3p=True)._single_layer.isVisibleTo(dialog)
    assert not SaveOptionsDialog(make_sdf(make_ramp()), is_x3p=True)._single_layer.isVisibleTo(
        dialog
    )


def test_an_x3p_file_is_saved_with_all_layers_or_with_the_chosen_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layers = np.stack([make_ramp(), 10.0 * make_ramp()])
    layered = x3pio.Surface.from_array(layers, x_scale=1e-6, y_scale=2e-6)
    window = MainWindow()
    export = window._exporter.write_export
    args = (x3pio.Revision.ISO25178_72_2017_DAM1, x3pio.DataType.FLOAT64, x3pio.DataStorage.BINARY)

    export(layered, str(tmp_path / "all.x3p"), *args, 1)
    export(layered, str(tmp_path / "one.x3p"), *args, 1, True)

    assert len(x3pio.read(tmp_path / "all.x3p").layers) == 2
    chosen = x3pio.read(tmp_path / "one.x3p")
    assert len(chosen.layers) == 1
    np.testing.assert_allclose(chosen.layer.z, 10.0 * make_ramp(), atol=1e-8, equal_nan=True)
    window.close()


def _layered_path(tmp_path: Path, *layers: np.ndarray) -> Path:
    path = tmp_path / "layers.x3p"
    x3pio.Surface.from_array(np.stack(layers), x_scale=1e-6, y_scale=2e-6).save(path)
    return path


def _choose_layer(window: MainWindow, number: int) -> None:
    window._layer_spin.setValue(number)
    _wait_for(window._loader.layer_loaded, window._loader.layer_failed)


def test_the_layer_selector_is_shown_only_for_a_file_of_several_layers(tmp_path: Path) -> None:
    window = MainWindow()
    assert not any(control.isVisible() for control in window._layer_controls)
    single = tmp_path / "single.x3p"
    make_x3p().save(single)
    _load(window, single)

    assert not any(control.isVisible() for control in window._layer_controls)
    assert not window._next_layer_action.isEnabled()

    _load(window, _layered_path(tmp_path, make_ramp(), make_ramp(), make_ramp()))

    assert all(control.isVisible() for control in window._layer_controls)
    assert window._layer_spin.value() == 1
    assert window._layer_spin.maximum() == 3
    assert window._layer_spin.suffix() == " of 3"
    assert window._next_layer_action.isEnabled()
    window.close()


def test_a_layer_is_shown_in_every_view_with_its_own_color_range(tmp_path: Path) -> None:
    window = MainWindow()
    _load(window, _layered_path(tmp_path, make_ramp(), 10.0 * make_ramp()))
    first = window._model
    assert isinstance(first, SurfaceModel)
    assert window._colorbar._range == first.value_range

    _choose_layer(window, 2)

    second = window._model
    assert isinstance(second, SurfaceModel)
    assert window._layer == 1
    assert second.value_range.hi == pytest.approx(10.0 * first.value_range.hi)
    assert window._colorbar._range == second.value_range  # a range for each layer
    assert window._heatmap_view._model is second
    assert window._profile_view._model is second
    assert window._surface_view._model is second
    assert "layer 2 of 2" in window._stats_label.text()
    _choose_layer(window, 1)
    assert window._model is not None
    assert window._model.value_range == first.value_range
    assert "layer 1 of 2" in window._stats_label.text()
    window.close()


def test_changing_the_layer_keeps_the_zoom_the_camera_and_the_profile(tmp_path: Path) -> None:
    window = MainWindow()
    _load(window, _layered_path(tmp_path, make_ramp(), make_ramp() + 1e-9))
    window._heatmap_view.zoom_in()
    window._surface_view._camera.azimuth = 12.0
    window._profile_view._spin.setValue(3)

    _choose_layer(window, 2)

    assert not window._heatmap_view._fitted
    assert window._surface_view._camera.azimuth == 12.0
    assert window._profile_view._spin.value() == 3
    window.close()


def test_a_new_file_starts_at_the_first_layer(tmp_path: Path) -> None:
    window = MainWindow()
    _load(window, _layered_path(tmp_path, make_ramp(), make_ramp(), make_ramp()))
    _choose_layer(window, 3)
    assert window._layer == 2

    _load(window, _layered_path(tmp_path, make_ramp(), make_ramp()))

    assert window._layer == 0
    assert window._layer_spin.value() == 1
    assert window._layer_spin.maximum() == 2
    window.close()


def test_the_previous_and_the_next_layer_actions_step_through_the_layers(tmp_path: Path) -> None:
    window = MainWindow()
    _load(window, _layered_path(tmp_path, make_ramp(), make_ramp(), make_ramp()))

    window._next_layer_action.trigger()
    _wait_for(window._loader.layer_loaded)
    assert window._layer == 1
    window._previous_layer_action.trigger()
    _wait_for(window._loader.layer_loaded)
    assert window._layer == 0
    window._previous_layer_action.trigger()  # the first layer has no previous one
    assert window._layer_spin.value() == 1
    window.close()


def test_the_last_choice_wins_while_a_layer_is_being_built(tmp_path: Path) -> None:
    window = MainWindow()
    _load(window, _layered_path(tmp_path, make_ramp(), make_ramp(), make_ramp()))

    window._layer_spin.setValue(2)  # starts building the second layer
    window._layer_spin.setValue(3)  # asked for while it is built
    window._layer_spin.setValue(1)  # and back again before it is done
    while window._loader.busy:
        _wait_for(window._loader.layer_loaded, timeout_ms=2000)

    assert window._layer == 0
    assert window._layer_spin.value() == 1
    window.close()


def test_a_layer_without_measured_points_is_refused(tmp_path: Path) -> None:
    window = MainWindow()
    _load(window, _layered_path(tmp_path, make_ramp(), np.full(make_ramp().shape, np.nan)))

    _choose_layer(window, 2)

    assert window._layer == 0
    assert window._layer_spin.value() == 1  # back at the layer that is shown
    assert "Layer 2" in window.statusBar().currentMessage()
    assert "no measured points" in window.statusBar().currentMessage()
    window.close()


def test_the_chosen_layer_is_saved_as_an_sdf_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    path = _layered_path(tmp_path, make_ramp(), 10.0 * make_ramp())
    _load(window, path)
    _choose_layer(window, 2)
    target = tmp_path / "second.sdf"
    asked: list[int] = []

    def fake_save_as(file: object, source: object, layer: int = 0) -> None:
        asked.append(layer)
        window._exporter.write_export(
            file,  # type: ignore[arg-type]
            str(target),
            sdfio.SdfDialect.ISO_2_0,
            sdfio.DataType.BINARY64,
            sdfio.FileFormat.BINARY,
            layer,
        )

    monkeypatch.setattr(window._exporter, "save_as", fake_save_as)
    window._save_as_action.trigger()

    assert asked == [1]
    np.testing.assert_allclose(
        sdfio.read(target).data, 10.0 * make_ramp(), atol=1e-8, equal_nan=True
    )
    window.close()
