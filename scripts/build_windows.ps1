param(
    [string]$PythonExe = "",
    [switch]$Clean
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$specPath = Join-Path $repoRoot "screenlens.spec"
$distPath = Join-Path $repoRoot "dist"
$buildPath = Join-Path $repoRoot "build"
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$vendorBinary = Join-Path $repoRoot "vendor\tesseract\tesseract.exe"

function Resolve-BootstrapPython {
    $launcherCandidates = @(
        @{ Command = "py"; Args = @("-3.13", "-c", "import sys; print(sys.executable)") },
        @{ Command = "py"; Args = @("-3.12", "-c", "import sys; print(sys.executable)") },
        @{ Command = "py"; Args = @("-3.11", "-c", "import sys; print(sys.executable)") },
        @{ Command = "python"; Args = @("-c", "import sys; print(sys.executable)") }
    )

    foreach ($candidate in $launcherCandidates) {
        try {
            $resolved = & $candidate.Command @($candidate.Args) 2>$null
            if ($LASTEXITCODE -eq 0 -and $resolved) {
                $resolvedPath = $resolved | Select-Object -First 1
                if (Test-Path $resolvedPath) {
                    return (Resolve-Path $resolvedPath).Path
                }
            }
        } catch {
        }
    }

    throw "No usable Python 3.11+ interpreter was found. Install Python first."
}

function Resolve-ProjectPython {
    param(
        [string]$RequestedPythonPath
    )

    if ($RequestedPythonPath) {
        $candidatePath = $RequestedPythonPath
        if (-not [System.IO.Path]::IsPathRooted($candidatePath)) {
            $candidatePath = Join-Path $repoRoot $candidatePath
        }
        if (-not (Test-Path $candidatePath)) {
            throw "Python interpreter not found: $candidatePath"
        }
        return (Resolve-Path $candidatePath).Path
    }

    if (Test-Path $venvPython) {
        return (Resolve-Path $venvPython).Path
    }

    $bootstrapPython = Resolve-BootstrapPython
    Write-Host "Creating project virtual environment with $bootstrapPython" -ForegroundColor Cyan
    & $bootstrapPython -m venv (Join-Path $repoRoot ".venv")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    if (-not (Test-Path $venvPython)) {
        throw "Failed to create .venv at $venvPython"
    }

    return (Resolve-Path $venvPython).Path
}

$pythonPath = Resolve-ProjectPython -RequestedPythonPath $PythonExe

if ($Clean) {
    foreach ($target in @($buildPath, $distPath)) {
        if (Test-Path $target) {
            Remove-Item -LiteralPath $target -Recurse -Force
        }
    }
}

Write-Host "Using Python: $pythonPath" -ForegroundColor Cyan

& $pythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $pythonPath -m pip install -e ".[build]"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $pythonPath -m PyInstaller --noconfirm --clean $specPath
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Build completed:" -ForegroundColor Green
Write-Host "  $distPath\ScreenLens\ScreenLens.exe"

if (Test-Path $vendorBinary) {
    Write-Host "Bundled OCR runtime detected from vendor\\tesseract." -ForegroundColor Green
} else {
    Write-Host "No bundled Tesseract was found." -ForegroundColor Yellow
    Write-Host "The app will still run in detection-only mode unless Tesseract is installed on the VM or copied beside the EXE."
}
