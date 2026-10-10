# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Builds the standalone Windows app (build\dist\sdflens) and, when the .NET SDK is installed,
# the installer (build\installer\sdflens-<version>.msi) with WiX, next to a CycloneDX SBOM of the
# runtime dependencies (build\installer\sdflens-<version>.cdx.json).
# Run from the project root inside the activated virtual environment:
#   .\packaging\build_windows.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot)

# WiX needs the license as RTF; the text is escaped so that any character survives.
function ConvertTo-Rtf([string]$Text) {
    $lines = $Text -split "`r?`n" | ForEach-Object {
        $line = $_.Replace('\', '\\').Replace('{', '\{').Replace('}', '\}')
        [regex]::Replace($line, '[^\x00-\x7F]', { param($m) '\u' + [int][char]$m.Value + '?' })
    }
    '{\rtf1\ansi\deff0{\fonttbl{\f0\fmodern Consolas;}}\f0\fs18 ' + ($lines -join "\par`r`n") + '}'
}

# Installs exactly the versions of pylock.build.toml, which uv generates for all platforms:
#   uv pip compile pyproject.toml --group packaging --universal --python-version 3.14 `
#       --generate-hashes -o pylock.build.toml
python -m pip install -r pylock.build.toml
if ($LASTEXITCODE -ne 0) { throw "Installing the locked packages failed" }
python -m pip install --no-deps --no-build-isolation -e .
if ($LASTEXITCODE -ne 0) { throw "Installing sdflens failed" }
python packaging\make_icon.py build\sdflens.ico
python packaging\make_installer_art.py build
python packaging\make_notices.py build\THIRD-PARTY-NOTICES.txt
python packaging\make_sbom.py build\sdflens.cdx.json
if ($LASTEXITCODE -ne 0) { throw "Generating the SBOM failed" }
python -m PyInstaller packaging\sdflens.spec --noconfirm --distpath build\dist --workpath build\work
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# The license files sit next to the executable, which also makes the installer ship them.
Copy-Item build\THIRD-PARTY-NOTICES.txt build\dist\sdflens\THIRD-PARTY-NOTICES.txt
Copy-Item LICENSE build\dist\sdflens\LICENSE
if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
    Write-Warning "The .NET SDK was not found; only the app folder build\dist\sdflens was created."
    return
}

$version = python -c "import sdflens; print(sdflens.__version__)"
dotnet tool restore
if ($LASTEXITCODE -ne 0) { throw "dotnet tool restore failed" }
if (-not (dotnet wix extension list -g | Select-String "WixToolset.UI.wixext")) {
    dotnet wix extension add -g WixToolset.UI.wixext/6.0.2
    if ($LASTEXITCODE -ne 0) { throw "Adding the WiX UI extension failed" }
}

$license = Get-Content LICENSE -Raw -Encoding UTF8
Set-Content build\LICENSE.rtf (ConvertTo-Rtf $license) -Encoding ascii
New-Item -ItemType Directory -Force build\installer | Out-Null
# WiX resolves relative paths against the .wxs file, so pass absolute ones.
$build = (Resolve-Path build).Path
dotnet wix build packaging\installer.wxs -arch x64 -ext WixToolset.UI.wixext `
    -d Version=$version -d AppDir=$build\dist\sdflens -d BuildDir=$build `
    -o "build\installer\sdflens-$version.msi"
if ($LASTEXITCODE -ne 0) { throw "WiX failed" }
Copy-Item build\sdflens.cdx.json "build\installer\sdflens-$version.cdx.json"
