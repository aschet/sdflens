# sdflens

[![CI](https://github.com/aschet/sdflens/actions/workflows/ci.yml/badge.svg)](https://github.com/aschet/sdflens/actions/workflows/ci.yml)

Qt based 3D, 2D and profile viewer for ISO 25178-71 Surface Data Files (`.sdf`), built on
[sdfio](https://github.com/aschet/sdfio) and PySide6.

![sdflens showing a synthetic micro-lens array](docs/screenshot.png)

## Features

- Opens ISO 25178-71 SDF files of all versions, in the binary and the ASCII format.
- Three views: 3D as surface, wireframe or points with adjustable height exaggeration; 2D with
  the coordinates and height under the cursor; and a profile plot of the height along any row
  or column.
- Selectable, reversible colormaps with a color bar; non-measured points are left empty.
- Shows the header fields and the trailer of the file, which can be copied or exported as text.
- Saves the surface as another version, format or data type of the standard.
- Screenshots as PNG file or on the clipboard.

Try it with the sample [samples/microlens-array.sdf](samples/microlens-array.sdf).

## Build and Run

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install .
sdflens
```

Linux and macOS:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install .
sdflens
```

`python -m sdflens` starts the viewer as well.

## Windows Installer

With the virtual environment activated and the [.NET SDK](https://dotnet.microsoft.com/download)
installed, this builds a standalone app and the [WiX](https://wixtoolset.org/) installer
`build\installer\sdflens-<version>.msi`, next to a [CycloneDX](https://cyclonedx.org/) SBOM of the
runtime dependencies, `build\installer\sdflens-<version>.cdx.json`:

```powershell
.\packaging\build_windows.ps1
```

The build installs exactly the package versions listed in `pylock.build.toml`. The uv command that updates them is at the top of `packaging\build_windows.ps1`.
