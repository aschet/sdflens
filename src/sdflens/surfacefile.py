# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reading SDF and x3p files and turning them into what the views display."""

from __future__ import annotations

import numpy as np
import sdfio
import x3pio

from .mesh import SurfaceMesh
from .pointcloud import PointCloudModel
from .surface import SurfaceModel

__all__ = [
    "FILE_ERRORS",
    "FILE_FILTER",
    "SurfaceFile",
    "build_mesh",
    "build_model",
    "is_grid",
    "layer_count",
    "read_surface_file",
]

#: A file the viewer can load: an SDF file or an x3p file.
SurfaceFile = sdfio.SdfFile | x3pio.X3pFile

#: Errors that reading a file may raise for a file that is damaged or cannot be accessed.
FILE_ERRORS = (sdfio.SdfError, x3pio.X3pError, OSError)

#: Extensions of the files the viewer opens, for the file dialog.
FILE_FILTER = "*.sdf *.x3p"

_ZIP_MAGIC = b"PK"


def read_surface_file(path: str, *, verify: bool = True) -> SurfaceFile:
    """Read ``path`` as an x3p file if it is a zip container, otherwise as an SDF file.

    The kind is taken from the content, not from the extension. ``verify`` checks the MD5
    checksums of an x3p file, see :func:`x3pio.read`; an SDF file has none.

    :raises x3pio.X3pChecksumError: If ``verify`` and a checksum of an x3p file does not match.
    """
    with open(path, "rb") as stream:
        is_x3p = stream.read(len(_ZIP_MAGIC)) == _ZIP_MAGIC
    return x3pio.read(path, verify=verify) if is_x3p else sdfio.read(path)


def is_grid(file: SurfaceFile) -> bool:
    """Whether ``file`` is a matrix of heights on a regular grid, which an SDF file can hold."""
    if isinstance(file, sdfio.SdfFile):
        return True
    return file.layers[0].is_regular_grid


def layer_count(file: SurfaceFile) -> int:
    """Return the number of layers of ``file``; an SDF file and a point cloud have one."""
    return 1 if isinstance(file, sdfio.SdfFile) else len(file.layers)


def build_model(file: SurfaceFile, layer: int = 0) -> SurfaceModel | PointCloudModel:
    """Return the model displaying ``layer`` of ``file``, counted from 0.

    An SDF file and an x3p surface or profile on a regular grid are a :class:`SurfaceModel`. All
    other x3p data is a :class:`PointCloudModel`: a point cloud, an irregular profile, or an
    irregular surface, whose x or y coordinates are stored for every point and which keeps its
    matrix so that it is drawn as a surface. The coordinates are those of the view system,
    without the offset and rotation of the file.

    :raises NoMeasuredPointsError: If the layer holds no measured points.
    :raises IndexError: If ``file`` has no such layer.
    """
    if isinstance(file, sdfio.SdfFile):
        return SurfaceModel.from_sdf(file)
    chosen = file.layers[layer]
    if chosen.is_regular_grid:
        header = chosen.header
        return SurfaceModel.from_grid(
            np.atleast_2d(chosen.z), abs(header.x.increment), abs(header.y.increment)
        )
    points = chosen.points(coordinate_system=x3pio.CoordinateSystem.VIEW)
    if isinstance(chosen, x3pio.SurfaceLayer):
        return PointCloudModel.from_grid(points)
    return PointCloudModel.from_points(points.reshape(-1, 3))


def build_mesh(model: SurfaceModel | PointCloudModel) -> SurfaceMesh:
    """Return the render mesh of ``model``."""
    if isinstance(model, PointCloudModel):
        return SurfaceMesh.from_point_cloud(model)
    return SurfaceMesh.from_model(model)
