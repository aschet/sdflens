# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenGL 3D view of the surface: colormapped, lit, with orbit/pan/zoom interaction."""

from __future__ import annotations

import math
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QMatrix4x4,
    QMouseEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLFunctions_3_3_Core,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget
from shiboken6 import VoidPtr

from .camera import Camera, Tool
from .mesh import SurfaceMesh
from .pick import pick, project
from .surface import Model3D
from .units import format_length
from .zscale import clamp_z_factor

__all__ = ["RenderMode", "SurfaceView"]

_GL_POINTS = 0x0000
_GL_LINES = 0x0001
_GL_TRIANGLES = 0x0004
_GL_UNSIGNED_INT = 0x1405
_GL_FLOAT = 0x1406
_GL_DEPTH_TEST = 0x0B71
_GL_PROGRAM_POINT_SIZE = 0x8642
_GL_COLOR_BUFFER_BIT = 0x4000
_GL_DEPTH_BUFFER_BIT = 0x0100

_ZOOM_STEP = 1.2
_POINT_SIZE = 3.0  # device-independent pixels
_GIZMO_LENGTH = 30.0
_GIZMO_HIT_RADIUS = 12.0
_CLICK_DISTANCE = 4.0  # pixels a press may move and still be a click
_PICK_RADIUS = 6.0  # pixels around a click within which a point is picked
_MARKER_RADIUS = 4.5
_PICK_COLOR = QColor(230, 120, 20)
_GIZMO_AXES = (
    ("x", "X", QColor(230, 90, 90)),
    ("y", "Y", QColor(110, 210, 110)),
    ("z", "Z", QColor(110, 150, 240)),
)
# Camera azimuth and elevation seeing each axis head on from its positive side; the camera
# limits the elevation of the z view to just below 90 degrees.
_AXIS_VIEWS = {"x": (0.0, 0.0), "y": (90.0, 0.0), "z": (-90.0, 90.0)}


class RenderMode(StrEnum):
    """How the surface is drawn."""

    SURFACE = "surface"
    WIREFRAME = "wireframe"
    POINTS = "points"


def _primitives(mesh: SurfaceMesh, mode: RenderMode) -> tuple[int, NDArray[np.uint32]]:
    """Return the GL primitive type and the vertex indices that draw ``mesh`` in ``mode``."""
    if mode is RenderMode.POINTS or mesh.is_cloud:
        return _GL_POINTS, np.flatnonzero(mesh.valid).astype(np.uint32)
    if mode is RenderMode.SURFACE and mesh.triangles.size:
        return _GL_TRIANGLES, mesh.triangles
    return _GL_LINES, mesh.lines


_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
uniform mat4 view;
uniform mat4 proj;
uniform vec3 params;
uniform vec2 colorMap;
out float vFraction;
out vec3 vViewPosition;
void main() {
    vFraction = position.z * colorMap.x + colorMap.y;
    vec4 viewPosition = view * vec4(position.xy, position.z * params.x, 1.0);
    vViewPosition = viewPosition.xyz;
    gl_Position = proj * viewPosition;
    gl_PointSize = params.z;
}
"""

# Face normals from screen-space derivatives: no per-vertex normals, so no NaN can leak in.
_FRAGMENT_SHADER = """
#version 330 core
in float vFraction;
in vec3 vViewPosition;
uniform sampler2D lut;
uniform vec3 params;
out vec4 fragColor;
void main() {
    float u = clamp(vFraction, 0.0, 1.0) * (255.0 / 256.0) + 0.5 / 256.0;
    vec3 base = texture(lut, vec2(u, 0.5)).rgb;
    if (params.y > 0.5) {
        vec3 normal = normalize(cross(dFdx(vViewPosition), dFdy(vViewPosition)));
        vec3 light = normalize(vec3(-0.4, 0.5, 0.8));
        base *= 0.35 + 0.65 * abs(dot(normal, light));
    }
    fragColor = vec4(base, 1.0);
}
"""


class SurfaceView(QOpenGLWidget):
    """3D surface view; left drag runs the active tool, middle drag pans, right drag zooms.

    A click on the surface picks the nearest measured point and reports its position.
    """

    #: The position of the picked point as text, empty when a click picks nothing.
    picked = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty view."""
        super().__init__(parent)
        self.setMouseTracking(True)
        self._camera = Camera()
        self._model: Model3D | None = None
        self._mesh: SurfaceMesh | None = None
        self._z_factor = 1.0
        self._tool = Tool.ROTATE
        self._lut: NDArray[np.uint8] | None = None
        self._lut_dirty = False
        self._geometry_dirty = False
        self._render_mode = RenderMode.SURFACE
        self._draw_mode = _GL_TRIANGLES
        self._draw_count = 0
        self._last_pos = QPointF()
        self._drag_tool: Tool | None = None
        self._press_pos: QPointF | None = None
        self._picked: int | None = None
        self._gl: QOpenGLFunctions_3_3_Core | None = None
        self._program: QOpenGLShaderProgram | None = None
        self._uniforms: dict[str, int] = {}
        self._vao: QOpenGLVertexArrayObject | None = None
        self._vertex_buffer: QOpenGLBuffer | None = None
        self._index_buffer: QOpenGLBuffer | None = None
        self._texture: QOpenGLTexture | None = None

    def set_surface(self, model: Model3D, mesh: SurfaceMesh, *, keep_view: bool = False) -> None:
        """Show ``mesh`` of ``model`` and reset the camera, unless ``keep_view`` keeps it.

        Another layer of the same file is shown with ``keep_view``, so that the view stays.
        """
        self._model = model
        self._mesh = mesh
        self._picked = None
        self._geometry_dirty = True
        if keep_view:
            self._camera.set_radius(self._scene_radius())
        else:
            self._camera.fit(self._scene_radius())
        self.update()

    def set_lut(self, lut: NDArray[np.uint8]) -> None:
        """Set the ``(256, 3)`` color lookup table."""
        self._lut = lut
        self._lut_dirty = True
        self.update()

    def set_z_factor(self, factor: float) -> None:
        """Set the height exaggeration relative to the true metric scale."""
        self._z_factor = clamp_z_factor(factor)
        if self._model is not None:
            self._camera.set_radius(self._scene_radius())
        self.update()

    def set_tool(self, tool: Tool) -> None:
        """Set the action of the left mouse button."""
        self._tool = tool

    def set_render_mode(self, mode: RenderMode) -> None:
        """Draw the surface as shaded faces, grid lines or measured points."""
        if mode is not self._render_mode:
            self._render_mode = mode
            self._geometry_dirty = True
            self.update()

    def home(self) -> None:
        """Reset the camera to the default view."""
        if self._model is not None:
            self._camera.fit(self._scene_radius())
        self.update()

    def zoom_in(self) -> None:
        """Zoom in one step."""
        self._camera.zoom(_ZOOM_STEP)
        self.update()

    def zoom_out(self) -> None:
        """Zoom out one step."""
        self._camera.zoom(1.0 / _ZOOM_STEP)
        self.update()

    def grab_image(self) -> QImage:
        """Return the rendered view."""
        return self.grabFramebuffer()

    def initializeGL(self) -> None:
        """Compile the shaders and create the GL objects."""
        gl = QOpenGLFunctions_3_3_Core()
        if not gl.initializeOpenGLFunctions():
            raise RuntimeError("OpenGL 3.3 core profile is required")
        self._gl = gl

        program = QOpenGLShaderProgram(self)
        if not (
            program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, _VERTEX_SHADER)
            and program.addShaderFromSourceCode(
                QOpenGLShader.ShaderTypeBit.Fragment, _FRAGMENT_SHADER
            )
            and program.link()
        ):
            raise RuntimeError(f"Shader error: {program.log()}")
        self._program = program
        self._uniforms = {
            name: program.uniformLocation(name) for name in ("view", "proj", "params", "colorMap")
        }

        self._vao = QOpenGLVertexArrayObject(self)
        self._vao.create()
        self._vertex_buffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self._vertex_buffer.create()
        self._index_buffer = QOpenGLBuffer(QOpenGLBuffer.Type.IndexBuffer)
        self._index_buffer.create()
        self._geometry_dirty = True
        self._lut_dirty = self._lut is not None

        context = self.context()
        context.aboutToBeDestroyed.connect(self.release_gl)

    def paintGL(self) -> None:
        """Draw the surface and the overlay."""
        gl = self._gl
        program = self._program
        if gl is None or program is None:
            return
        background = self.palette().window().color()
        gl.glClearColor(background.redF(), background.greenF(), background.blueF(), 1.0)
        gl.glClear(_GL_COLOR_BUFFER_BIT | _GL_DEPTH_BUFFER_BIT)
        self._upload_pending()

        model = self._model
        if model is not None and self._draw_count > 0 and self._texture is not None:
            self._draw_geometry(gl, program, model)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if model is not None:
            self._draw_gizmo(painter)
            self._draw_picked(painter)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Align the view to a clicked axis, or start a drag with the tool of the button."""
        button = event.button()
        if button == Qt.MouseButton.LeftButton:
            axis = self._axis_at(event.position())
            if axis is not None:
                self._camera.set_view(*_AXIS_VIEWS[axis])
                self.update()
                return
            self._drag_tool = self._tool
            self._press_pos = event.position()
        elif button == Qt.MouseButton.MiddleButton:
            self._drag_tool = Tool.PAN
        elif button == Qt.MouseButton.RightButton:
            self._drag_tool = Tool.ZOOM
        else:
            return
        self._last_pos = event.position()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Apply the active drag tool, or show a hand over a clickable axis."""
        if self._drag_tool is None:
            if self._axis_at(event.position()) is None:
                self.unsetCursor()
            else:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            return
        pos = event.position()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self._last_pos = pos
        if self._drag_tool is Tool.ROTATE:
            self._camera.rotate(dx, dy)
        elif self._drag_tool is Tool.PAN:
            self._camera.pan(dx, dy, self.height())
        else:
            self._camera.zoom(math.exp(-dy * 0.01))
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """End the drag; a left press that did not move is a click that picks a point."""
        press = self._press_pos
        self._press_pos = None
        self._drag_tool = None
        if press is not None and event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            if math.hypot(pos.x() - press.x(), pos.y() - press.y()) <= _CLICK_DISTANCE:
                self._pick(pos)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom with the wheel."""
        steps = event.angleDelta().y() / 120.0
        self._camera.zoom(_ZOOM_STEP**steps)
        self.update()

    def _pick(self, pos: QPointF) -> None:
        """Pick the measured point nearest to ``pos`` and report its position."""
        mesh = self._mesh
        if mesh is None:
            return
        view, proj = self._camera.matrices(self.width() / max(self.height(), 1))
        self._picked = pick(
            mesh.vertices,
            mesh.valid,
            mesh.triangles,
            view,
            proj,
            self._z_factor,
            (pos.x(), pos.y()),
            (float(self.width()), float(self.height())),
            _PICK_RADIUS,
        )
        self.update()
        if self._picked is None:
            self.picked.emit("")
            return
        x, y, z = mesh.position(self._picked)
        self.picked.emit(
            self.tr("x = {x}, y = {y}, z = {z}").format(
                x=format_length(x), y=format_length(y), z=format_length(z)
            )
        )

    def _scene_radius(self) -> float:
        """Bounding radius of the normalized scene including the current z exaggeration."""
        model = self._model
        if model is None:
            return 1.0
        extent_x = model.extent_x / model.scale
        extent_y = model.extent_y / model.scale
        height = model.height_fraction * self._z_factor
        return 0.5 * math.sqrt(extent_x**2 + extent_y**2 + height**2)

    def _upload_pending(self) -> None:
        """Upload geometry and color table changes; requires the GL context to be current."""
        if self._geometry_dirty:
            self._upload_geometry()
            self._geometry_dirty = False
        if self._lut_dirty and self._lut is not None:
            self._upload_lut(self._lut)
            self._lut_dirty = False

    def _upload_geometry(self) -> None:
        vao = self._vao
        vertex_buffer = self._vertex_buffer
        index_buffer = self._index_buffer
        program = self._program
        if vao is None or vertex_buffer is None or index_buffer is None or program is None:
            return
        mesh = self._mesh
        indices: NDArray[np.uint32] = np.empty(0, dtype=np.uint32)
        if mesh is not None:
            self._draw_mode, indices = _primitives(mesh, self._render_mode)
        self._draw_count = int(indices.size)

        vao.bind()
        vertex_buffer.bind()
        index_buffer.bind()
        if mesh is not None and self._draw_count:
            vertex_data = mesh.vertices.tobytes()
            vertex_buffer.allocate(vertex_data, len(vertex_data))
            index_data = indices.tobytes()
            index_buffer.allocate(index_data, len(index_data))
            program.bind()
            program.enableAttributeArray(0)
            program.setAttributeBuffer(0, _GL_FLOAT, 0, 3, 12)
            program.release()
        vao.release()

    def _upload_lut(self, lut: NDArray[np.uint8]) -> None:
        image = QImage(
            np.ascontiguousarray(lut).tobytes(),
            len(lut),
            1,
            len(lut) * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        if self._texture is not None:
            self._texture.destroy()
        texture = QOpenGLTexture(image, QOpenGLTexture.MipMapGeneration.DontGenerateMipMaps)
        texture.setMinMagFilters(QOpenGLTexture.Filter.Linear, QOpenGLTexture.Filter.Linear)
        texture.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
        self._texture = texture

    def _draw_geometry(
        self, gl: QOpenGLFunctions_3_3_Core, program: QOpenGLShaderProgram, model: Model3D
    ) -> None:
        vao = self._vao
        texture = self._texture
        if vao is None or texture is None:
            return
        view, proj = self._camera.matrices(self.width() / max(self.height(), 1))
        gl.glEnable(_GL_DEPTH_TEST)
        gl.glEnable(_GL_PROGRAM_POINT_SIZE)
        program.bind()
        program.setUniformValue(self._uniforms["view"], _to_qmatrix(view))
        program.setUniformValue(self._uniforms["proj"], _to_qmatrix(proj))
        lit = 1.0 if self._draw_mode == _GL_TRIANGLES else 0.0
        point_size = _POINT_SIZE * self.devicePixelRatioF()
        program.setUniformValue(self._uniforms["params"], float(self._z_factor), lit, point_size)
        program.setUniformValue(
            self._uniforms["colorMap"], float(model.color_scale), float(model.color_offset)
        )
        texture.bind(0)
        vao.bind()
        # The stubs say int, but only a VoidPtr offset is accepted at runtime.
        gl.glDrawElements(
            self._draw_mode,
            self._draw_count,
            _GL_UNSIGNED_INT,
            VoidPtr(0),  # type: ignore[arg-type]
        )
        vao.release()
        texture.release()
        program.release()
        gl.glDisable(_GL_DEPTH_TEST)
        gl.glDisable(_GL_PROGRAM_POINT_SIZE)

    def _gizmo_origin(self) -> QPointF:
        return QPointF(48.0, self.height() - 48.0)

    def _gizmo_tips(self) -> dict[str, QPointF]:
        """Return the screen position of the tip of each gizmo axis."""
        view, _ = self._camera.matrices(1.0)
        origin = self._gizmo_origin()
        return {
            axis: QPointF(
                origin.x() + view[0, column] * _GIZMO_LENGTH,
                origin.y() - view[1, column] * _GIZMO_LENGTH,
            )
            for column, (axis, _label, _color) in enumerate(_GIZMO_AXES)
        }

    def _axis_at(self, pos: QPointF) -> str | None:
        """Return the gizmo axis whose tip is at ``pos``, if a surface is shown."""
        if self._model is None:
            return None
        nearest, distance = None, _GIZMO_HIT_RADIUS
        for axis, tip in self._gizmo_tips().items():
            tip_distance = math.hypot(pos.x() - tip.x(), pos.y() - tip.y())
            if tip_distance <= distance:
                nearest, distance = axis, tip_distance
        return nearest

    def _draw_gizmo(self, painter: QPainter) -> None:
        """Draw x/y/z axes rotating with the view in the lower left corner."""
        origin = self._gizmo_origin()
        tips = self._gizmo_tips()
        painter.setFont(QFont(painter.font().family(), 9, QFont.Weight.Bold))
        for axis, label, color in _GIZMO_AXES:
            tip = tips[axis]
            painter.setPen(QPen(color, 2.0))
            painter.drawLine(origin, tip)
            dx = (tip.x() - origin.x()) / _GIZMO_LENGTH
            dy = (tip.y() - origin.y()) / _GIZMO_LENGTH
            painter.drawText(QPointF(tip.x() + dx * 8.0 - 4.0, tip.y() + dy * 8.0 + 4.0), label)

    def _draw_picked(self, painter: QPainter) -> None:
        """Mark the picked point."""
        mesh = self._mesh
        if mesh is None or self._picked is None:
            return
        view, proj = self._camera.matrices(self.width() / max(self.height(), 1))
        screen, depth = project(
            mesh.vertices[self._picked : self._picked + 1],
            view,
            proj,
            self._z_factor,
            float(self.width()),
            float(self.height()),
        )
        if depth[0] <= 0.0:
            return
        painter.setPen(QPen(self.palette().window().color(), 1.5))
        painter.setBrush(_PICK_COLOR)
        painter.drawEllipse(
            QPointF(float(screen[0, 0]), float(screen[0, 1])),
            _MARKER_RADIUS,
            _MARKER_RADIUS,
        )

    def release_gl(self) -> None:
        """Free the GL objects while the context is still alive."""
        if self._gl is None:
            return
        self.makeCurrent()
        if self._texture is not None:
            self._texture.destroy()
            self._texture = None
        for buffer in (self._vertex_buffer, self._index_buffer):
            if buffer is not None:
                buffer.destroy()
        if self._vao is not None:
            self._vao.destroy()
        self._program = None
        self._gl = None
        self.doneCurrent()


def _to_qmatrix(matrix: NDArray[np.float64]) -> QMatrix4x4:
    return QMatrix4x4(matrix.astype(np.float32).ravel().tolist())
