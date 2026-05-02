param(
    [string]$PythonExe = "",
    [ValidateSet("auto", "cpu", "gpu")]
    [string]$TorchRuntime = "auto",
    [switch]$IncludeBuildTools
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$argosVendorDir = Join-Path $repoRoot "vendor\argos"
$argosDownloadScript = Join-Path $repoRoot "scripts\download_argos_models.py"
$torchVersion = "2.10.0"
$torchvisionVersion = "0.25.0"
$torchCpuIndex = "https://download.pytorch.org/whl/cpu"
$torchCudaIndex = "https://download.pytorch.org/whl/cu128"

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

function Test-NvidiaCudaPresent {
    $nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        return $false
    }

    try {
        & $nvidiaSmi.Source "--query-gpu=name" "--format=csv,noheader" 2>$null | Out-Null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Resolve-TorchRuntime {
    param(
        [string]$RequestedRuntime
    )

    if ($RequestedRuntime -ne "auto") {
        return $RequestedRuntime
    }

    if (Test-NvidiaCudaPresent) {
        return "gpu"
    }

    return "cpu"
}

function Invoke-PipInstall {
    param(
        [string]$PythonPath,
        [string[]]$Arguments
    )

    & $PythonPath -m pip @Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Install-BaseDependencies {
    param(
        [string]$PythonPath,
        [bool]$WithBuildTools
    )

    Write-Host "Upgrading pip" -ForegroundColor Cyan
    Invoke-PipInstall -PythonPath $PythonPath -Arguments @("install", "--upgrade", "pip")

    $editableTarget = if ($WithBuildTools) { ".[build]" } else { "." }
    Write-Host "Installing ScreenLens project dependencies from $editableTarget" -ForegroundColor Cyan
    Invoke-PipInstall -PythonPath $PythonPath -Arguments @("install", "-e", $editableTarget)
}

function Install-EasyOCR {
    param(
        [string]$PythonPath
    )

    Write-Host "Installing EasyOCR dependencies" -ForegroundColor Cyan
    Invoke-PipInstall -PythonPath $PythonPath -Arguments @("install", "--upgrade", "easyocr>=1.7.2")
}

function Install-RapidOCR {
    param(
        [string]$PythonPath
    )

    Write-Host "Installing RapidOCR dependencies" -ForegroundColor Cyan
    Invoke-PipInstall -PythonPath $PythonPath -Arguments @(
        "install",
        "--upgrade",
        "rapidocr>=3.0.0",
        "onnxruntime>=1.20.0"
    )
}

function Install-TorchRuntime {
    param(
        [string]$PythonPath,
        [string]$Runtime
    )

    if ($Runtime -eq "gpu") {
        $indexUrl = $torchCudaIndex
        $label = "CUDA 12.8"
    } else {
        $indexUrl = $torchCpuIndex
        $label = "CPU-only"
    }

    Write-Host "Installing PyTorch runtime: $label" -ForegroundColor Cyan
    Invoke-PipInstall -PythonPath $PythonPath -Arguments @(
        "install",
        "--upgrade",
        "--force-reinstall",
        "torch==$torchVersion",
        "torchvision==$torchvisionVersion",
        "--index-url",
        $indexUrl
    )
}

function Sync-ArgosModels {
    param(
        [string]$PythonPath
    )

    Write-Host "Downloading bundled Argos Translate models (en<->th)" -ForegroundColor Cyan
    & $PythonPath $argosDownloadScript --output-dir $argosVendorDir --pair en:th --pair th:en
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Get-TorchDiagnostics {
    param(
        [string]$PythonPath
    )

    $probeScript = @'
import json

try:
    import torch
except Exception as exc:
    print(json.dumps({"error": str(exc)}))
    raise SystemExit(0)

cuda_available = False
device_count = 0
device_name = ""

try:
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    if cuda_available and device_count > 0:
        device_name = str(torch.cuda.get_device_name(0))
except Exception:
    pass

print(json.dumps({
    "torch_version": getattr(torch, "__version__", ""),
    "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
    "cuda_available": cuda_available,
    "device_count": device_count,
    "device_name": device_name,
}))
'@

    $json = $probeScript | & $PythonPath -
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    return $json | ConvertFrom-Json
}

$pythonPath = Resolve-ProjectPython -RequestedPythonPath $PythonExe
$resolvedTorchRuntime = Resolve-TorchRuntime -RequestedRuntime $TorchRuntime

Write-Host "Using Python: $pythonPath" -ForegroundColor Cyan
Write-Host "Requested torch runtime: $TorchRuntime" -ForegroundColor Cyan
Write-Host "Resolved torch runtime: $resolvedTorchRuntime" -ForegroundColor Cyan

Install-BaseDependencies -PythonPath $pythonPath -WithBuildTools:$IncludeBuildTools
Install-EasyOCR -PythonPath $pythonPath
Install-RapidOCR -PythonPath $pythonPath
Install-TorchRuntime -PythonPath $pythonPath -Runtime $resolvedTorchRuntime
Sync-ArgosModels -PythonPath $pythonPath

$diagnostics = Get-TorchDiagnostics -PythonPath $pythonPath

if ($diagnostics.error) {
    throw "Failed to import torch after setup: $($diagnostics.error)"
}

Write-Host ""
Write-Host "Torch diagnostics:" -ForegroundColor Green
Write-Host "  Version: $($diagnostics.torch_version)"
if ($diagnostics.torch_cuda_version) {
    Write-Host "  CUDA runtime: $($diagnostics.torch_cuda_version)"
} else {
    Write-Host "  CUDA runtime: CPU-only build"
}
Write-Host "  CUDA available: $($diagnostics.cuda_available)"
Write-Host "  Device count: $($diagnostics.device_count)"
if ($diagnostics.device_name) {
    Write-Host "  Device: $($diagnostics.device_name)"
}

if ($TorchRuntime -eq "gpu" -and -not $diagnostics.cuda_available) {
    throw "GPU runtime was requested, but torch.cuda.is_available() is False after installation."
}
