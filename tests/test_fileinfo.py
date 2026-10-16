# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import sdfio
import x3pio

from helpers import make_cloud, make_ramp, make_sdf, make_x3p
from sdflens.fileinfo import InfoSection, describe_file

TIMES = chr(0x00D7)  # the multiplication sign


def sections(file: sdfio.SdfFile | x3pio.X3pFile) -> dict[str, dict[str, str]]:
    return {section.title: dict(section.rows) for section in describe_file(file)}


def test_an_sdf_file_has_the_four_sections_with_the_labels_of_sdflens_web() -> None:
    info = sections(make_sdf(make_ramp()))

    assert list(info) == ["File", "Data", "Instrument", "Probing"]
    assert list(info["File"]) == ["Format", "Revision", "Feature", "Creator", "Date"]
    assert list(info["Data"])[:4] == ["Axis", "Type", "Dimension", "Step"]
    assert list(info["Instrument"]) == [
        "Manufacturer",
        "Model",
        "Serial",
        "Version",
        "Calibration",
    ]
    assert list(info["Probing"]) == ["Type", "Identification"]


def test_an_sdf_file_says_what_it_stores() -> None:
    sdf = make_sdf(make_ramp(), x_scale=1e-6, y_scale=2e-6)
    info = sections(sdf)

    assert info["File"]["Format"] == "SDF binary"
    assert info["File"]["Revision"] == sdf.header.revision.value
    assert info["File"]["Feature"] == "Surface"
    assert info["File"]["Creator"] == "-"
    assert info["Data"]["Axis"] == "inc, inc, abs"
    assert info["Data"]["Type"] == "float64"
    rows, columns = sdf.data.shape
    assert info["Data"]["Dimension"] == f"{columns} {TIMES} {rows} points"
    assert info["Data"]["Step"] == f"1 µm {TIMES} 2 µm"
    # Only the manufacturer is something an SDF file knows of the instrument.
    assert info["Instrument"]["Manufacturer"] == sdf.header.manufacturer_id
    assert set(info["Probing"].values()) == {"-"}


def test_an_sdf_profile_is_a_profile_with_one_step_and_a_height_resolution() -> None:
    profile = sdfio.SdfFile.from_array(np.zeros((1, 7)), x_scale=2e-6, y_scale=2e-6, z_scale=1e-9)
    profile.header.z_resolution = 1e-10
    info = sections(profile)

    assert info["File"]["Feature"] == "Profile"
    assert info["Data"]["Dimension"] == "7 points"
    assert info["Data"]["Step"] == "2 µm"
    assert info["Data"]["Height resolution"] == "100 pm"
    assert "Height resolution" not in sections(make_sdf(make_ramp()))["Data"]


def test_an_x3p_surface_says_what_it_stores() -> None:
    x3p = make_x3p()
    info = sections(x3p)

    assert info["File"]["Format"] == "x3p binary"
    assert info["File"]["Revision"] == x3p.revision.value
    assert info["File"]["Feature"] == "Surface"
    assert info["File"]["Creator"] == "Jane Doe"
    assert info["Data"]["Axis"] == "inc, inc, abs"
    assert info["Data"]["Type"].count(",") == 2  # the type of each axis
    assert info["Data"]["Step"].count(TIMES) == 1
    assert info["Instrument"]["Manufacturer"] == "x3pio"
    assert info["Probing"]["Type"] == "Software"
    assert "Placement" not in info["Data"]


def test_an_x3p_point_cloud_is_absolute_in_all_axes_and_irregular() -> None:
    cloud = x3pio.PointCloud.from_points(make_cloud())
    info = sections(cloud)

    assert info["File"]["Feature"] == "Point cloud"
    assert info["Data"]["Axis"] == "abs, abs, abs"
    assert info["Data"]["Dimension"] == f"{len(make_cloud())} points"
    assert info["Data"]["Step"] == "irregular"


def test_an_x3p_profile_is_a_profile_and_gives_its_step() -> None:
    profile = x3pio.Profile.from_array(np.zeros(5), x_scale=3e-6)
    info = sections(profile)

    assert info["File"]["Feature"] == "Profile"
    assert info["Data"]["Dimension"] == "5 points"
    assert info["Data"]["Step"] == "3 µm"


def test_the_comment_of_an_x3p_file_is_a_section_of_its_own_at_the_end() -> None:
    described = describe_file(make_x3p())

    assert described[-1] == InfoSection("Comment", text="Test surface")
    assert [section.title for section in described[:-1]] == [
        "File",
        "Data",
        "Instrument",
        "Probing",
    ]
    without = x3pio.Surface.from_array(np.zeros((2, 2)), x_scale=1e-6, y_scale=1e-6)
    assert "Comment" not in sections(without)
    assert "Comment" not in sections(make_sdf(make_ramp()))


def test_placeholders_and_missing_values_show_as_a_dash() -> None:
    x3p = make_x3p().with_metadata(serial="N/A", model="  ", version="not set")
    instrument = sections(x3p)["Instrument"]

    assert instrument["Serial"] == "-"
    assert instrument["Model"] == "-"
    assert instrument["Version"] == "-"
    assert sections(x3p.with_metadata(creator=None))["File"]["Creator"] == "-"
    # An x3p file without metadata has a dash for everything about it.
    bare = make_x3p()
    bare.metadata = None
    info = sections(bare)
    assert info["File"]["Date"] == "-"
    assert set(info["Instrument"].values()) == {"-"}
    assert set(info["Probing"].values()) == {"-"}


def test_a_placement_of_an_x3p_file_is_said_not_to_be_applied() -> None:
    placed = make_x3p().with_placement(x3pio.Placement.translation(0.0, 0.0, 1e-3))

    assert sections(placed)["Data"]["Placement"] == "stored in the file, not applied"


def test_dates_are_written_by_the_given_function_or_as_local_time() -> None:
    when = datetime(2026, 1, 31, 12, 30, tzinfo=UTC)
    x3p = make_x3p().with_metadata(date=when, calibration_date=when)

    custom = describe_file(x3p, lambda value: f"<{value.year}>")
    assert dict(custom[0].rows)["Date"] == "<2026>"
    assert dict(custom[2].rows)["Calibration"] == "<2026>"
    default = dict(describe_file(x3p)[0].rows)["Date"]
    assert default.startswith("2026-01-31 ") or default.startswith("2026-02-01 ")
    assert len(default) == len("2026-01-31 12:30")
