# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reading SDF and x3p files and turning them into what the views display."""

from __future__ import annotations

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
    "read_surface_file",
]

#: A file the viewer can load: an SDF file or an x3p file.
SurfaceFile = sdfio.SdfFile | x3pio.X3pFile

#: Errors that reading a file may raise for a file that is damaged or cannot be accessed.
FILE_ERRORS = (sdfio.SdfError, x3pio.X3pError, OSError)

#: Extensions of the files the viewer opens, for the file dialog.
FILE_FILTER = "*.sdf *.x3p"

_ZIP_MAGIC = b"PK"


def read_surface_file(path: str) -> SurfaceFile:
    """Read ``path`` as an x3p file if it is a zip container, otherwise as an SDF file.

    The kind is taken from the content, not from the extension.
    """
    with open(path, "rb") as stream:
        is_x3p = stream.read(len(_ZIP_MAGIC)) == _ZIP_MAGIC
    return x3pio.read(path) if is_x3p else sdfio.read(path)


def is_grid(file: SurfaceFile) -> bool:
    """Whether ``file`` is a matrix of heights on a regular grid, which an SDF file can hold."""
    if isinstance(file, sdfio.SdfFile):
        return True
    return file.is_matrix and file.header.x.is_incremental and file.header.y.is_incremental


def build_model(file: SurfaceFile) -> SurfaceModel | PointCloudModel:
    """Return the model displaying ``file``.

    An SDF file and an x3p surface or profile on a regular grid are a :class:`SurfaceModel`; for
    several layers, the first one. All other x3p data is a :class:`PointCloudModel`: a point
    cloud, or a matrix whose x or y coordinates are stored for every point. The coordinates are
    those of the view system, without the offset and rotation of the file.

    :raises NoMeasuredPointsError: If the file holds no measured points.
    """
    if isinstance(file, sdfio.SdfFile):
        return SurfaceModel.from_sdf(file)
    if is_grid(file):
        return SurfaceModel.from_grid(
            file.data[0], abs(file.header.x.increment), abs(file.header.y.increment)
        )
    return PointCloudModel.from_points(file.to_points(global_coordinates=False))


def build_mesh(model: SurfaceModel | PointCloudModel) -> SurfaceMesh:
    """Return the render mesh of ``model``."""
    if isinstance(model, PointCloudModel):
        return SurfaceMesh.from_point_cloud(model)
    return SurfaceMesh.from_model(model)
