#!/bin/sh
# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Builds the standalone Linux app (build/dist/sdflens) and an AppDir (build/AppDir) around it,
# and, when appimagetool is found (in PATH or named by $APPIMAGETOOL), the AppImage
# build/appimage/sdflens-<version>-<arch>.AppImage, next to its SBOM, which has the same name with
# .cdx.json instead of .AppImage.
#
# Run it from the repository root with the packaging dependencies installed in the active Python:
#   uv pip install -r pylock.build.toml   (or the same with pip), then
#   pip install --no-deps --no-build-isolation -e .
# Build on the oldest distribution that should run the AppImage: it needs its glibc or newer.
set -eu

cd "$(dirname "$0")/.."
mkdir -p build
python packaging/make_notices.py build/THIRD-PARTY-NOTICES.txt
python packaging/make_sbom.py build/sdflens.cdx.json
python -m PyInstaller packaging/sdflens.spec --noconfirm --distpath build/dist --workpath build/work

# The license files sit next to the executable, like in the Windows build.
cp build/THIRD-PARTY-NOTICES.txt build/dist/sdflens/THIRD-PARTY-NOTICES.txt
cp LICENSE build/dist/sdflens/LICENSE

rm -rf build/AppDir
mkdir -p build/AppDir/usr/lib build/AppDir/usr/share/applications \
    build/AppDir/usr/share/icons/hicolor/scalable/apps
cp -r build/dist/sdflens build/AppDir/usr/lib/sdflens
cp packaging/appimage/AppRun build/AppDir/AppRun
cp packaging/appimage/sdflens.desktop build/AppDir/sdflens.desktop
cp packaging/appimage/sdflens.desktop build/AppDir/usr/share/applications/sdflens.desktop
cp src/sdflens/icons/app.svg build/AppDir/sdflens.svg
cp src/sdflens/icons/app.svg build/AppDir/usr/share/icons/hicolor/scalable/apps/sdflens.svg

tool="${APPIMAGETOOL:-$(command -v appimagetool || true)}"
if [ -z "$tool" ]; then
    echo "appimagetool was not found; only the AppDir build/AppDir was created." >&2
    exit 0
fi
version="$(python -c 'import sdflens; print(sdflens.__version__)')"
arch="$(uname -m)"
mkdir -p build/appimage
ARCH="$arch" "$tool" build/AppDir "build/appimage/sdflens-$version-$arch.AppImage"
cp build/sdflens.cdx.json "build/appimage/sdflens-$version-$arch.cdx.json"
