# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reading SDF and x3p files and turning them into what the views display."""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

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


class _Profile(NamedTuple):
    """A profile of a layer: the axis along which it runs, and its spacing if it is regular."""

    axis: str
    #: The spacing in meters; ``None`` if the points are not evenly spaced.
    step: float | None
    #: Whether the spacing was taken from the other lateral axis, as the positions are not usable.
    assumed: bool = False


#: How far the gaps between evenly spaced positions may differ from their mean, as a fraction.
_EVEN = 0.02

#: The largest position in meters that can be a position: a damaged file may hold any number.
_PLAUSIBLE = 1e3


def _usable(values: np.ndarray) -> bool:
    """Whether ``values`` can be positions: finite and not absurdly large."""
    return bool(np.all(np.isfinite(values)) and np.max(np.abs(values)) < _PLAUSIBLE)


def _profile_of(layer: x3pio.Layer) -> _Profile | None:
    """Return the profile that ``layer`` is, if it is one with an absolute axis.

    The axis along the profile decides. If it is incremental, its increment is the spacing and the
    other lateral axis, which a profile does not use, is ignored. Positions along an absolute
    axis that are evenly spaced give their spacing. Positions that cannot be used, as they are
    not finite or neither increasing nor decreasing, are replaced by the increment of the other
    axis if that is incremental; the spacing is then assumed. Positions that can be used but are
    not evenly spaced, and positions across the profile that vary, as along a diagonal, make an
    irregular profile, whose spacing is ``None``. Without usable positions or another regular
    axis to take the spacing from, the result is ``None``, as for a layer that is no profile.
    """
    if isinstance(layer, x3pio.ProfileLayer):
        along_x = True
    elif isinstance(layer, x3pio.SurfaceLayer) and layer.z.size > 1 and 1 in layer.z.shape:
        along_x = layer.z.shape[0] == 1
    else:
        return None
    name = "x" if along_x else "y"
    header = layer.header
    axis, other = (header.x, header.y) if along_x else (header.y, header.x)
    positions, across = (layer.x, layer.y) if along_x else (layer.y, layer.x)
    step: float | None
    if axis.is_incremental:
        step = abs(axis.increment)
    else:
        step, assumed = None, False
        values = np.ravel(positions) if positions is not None else np.empty(0)
        mean = (values[-1] - values[0]) / (values.size - 1) if values.size > 1 else 0.0
        if (
            values.size > 1
            and _usable(values)
            and mean != 0.0
            and np.all(np.diff(values) * mean > 0.0)
        ):
            if np.all(np.abs(np.diff(values) - mean) <= _EVEN * abs(mean)):
                step = abs(mean)
            else:
                return _Profile(name, None)
        elif other.is_incremental:
            step, assumed = abs(other.increment), True
        else:
            return None
        if across is not None and _usable(np.ravel(across)) and np.ptp(np.ravel(across)) > 0.0:
            return _Profile(name, None)
        return _Profile(name, step, assumed)
    if across is not None and _usable(np.ravel(across)) and np.ptp(np.ravel(across)) > 0.0:
        return _Profile(name, None)
    return _Profile(name, step)


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
    profile = _profile_of(chosen)
    if profile is not None and profile.step is not None:
        heights = np.asarray(chosen.z, dtype=np.float64)
        data = heights.reshape(1, -1) if profile.axis == "x" else heights.reshape(-1, 1)
        return SurfaceModel.from_grid(
            data, profile.step, profile.step, step_assumed=profile.assumed
        )
    points = chosen.points(coordinate_system=x3pio.CoordinateSystem.VIEW)
    if profile is not None:
        irregular = PointCloudModel.from_points(points.reshape(-1, 3))
        return replace(irregular, profile_axis=profile.axis)
    if isinstance(chosen, x3pio.SurfaceLayer):
        return PointCloudModel.from_grid(points)
    return PointCloudModel.from_points(points.reshape(-1, 3))


def build_mesh(model: SurfaceModel | PointCloudModel) -> SurfaceMesh:
    """Return the render mesh of ``model``."""
    if isinstance(model, PointCloudModel):
        return SurfaceMesh.from_point_cloud(model)
    return SurfaceMesh.from_model(model)
