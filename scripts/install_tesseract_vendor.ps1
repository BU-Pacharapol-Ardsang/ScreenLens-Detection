param(
    [string]$OutputDir = "",
    [string]$InstallerUrl = "https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe",
    [string[]]$Languages = @("eng", "tha", "osd")
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $OutputDir) {
    $OutputDir = Join-Path $repoRoot "vendor\tesseract"
}

$vendorDir = $OutputDir
if (-not [System.IO.Path]::IsPathRooted($vendorDir)) {
    $vendorDir = Join-Path $repoRoot $vendorDir
}

$vendorDir = [System.IO.Path]::GetFullPath($vendorDir)
$tesseractExe = Join-Path $vendorDir "tesseract.exe"
$tessdataDir = Join-Path $vendorDir "tessdata"

function Invoke-DownloadFile {
    param(
        [string]$Uri,
        [string]$Destination
    )

    Write-Host "Downloading $Uri" -ForegroundColor Cyan
    try {
        Invoke-WebRequest -Uri $Uri -OutFile $Destination -UseBasicParsing -TimeoutSec 120
    } catch {
        if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
            & curl.exe -L --fail --retry 3 --connect-timeout 30 --output $Destination $Uri
            if ($LASTEXITCODE -eq 0) {
                return
            }
        }
        throw
    }
}

function Install-TesseractRuntime {
    if (Test-Path $tesseractExe) {
        return
    }

    New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null

    Copy-InstalledTesseractRuntime
    if (Test-Path $tesseractExe) {
        return
    }

    $downloadDir = Join-Path $repoRoot ".download"
    New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
    $installerPath = Join-Path $downloadDir ([System.IO.Path]::GetFileName(([Uri]$InstallerUrl).AbsolutePath))

    if (-not (Test-Path $installerPath)) {
        Invoke-DownloadFile -Uri $InstallerUrl -Destination $installerPath
    }

    Write-Host "Installing Tesseract runtime into vendor\tesseract" -ForegroundColor Cyan
    $installArgs = @(
        "/S",
        "/D=$vendorDir"
    )
    $process = Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -ne 0) {
        throw "Tesseract installer failed with exit code $($process.ExitCode)."
    }

    if (-not (Test-Path $tesseractExe)) {
        Copy-InstalledTesseractRuntime
    }

    if (-not (Test-Path $tesseractExe)) {
        throw "Tesseract installer completed, but tesseract.exe was not found at $tesseractExe"
    }
}

function Copy-InstalledTesseractRuntime {
    $candidateDirs = @()
    $pathCommand = Get-Command "tesseract.exe" -ErrorAction SilentlyContinue
    if ($pathCommand) {
        $candidateDirs += (Split-Path -Parent $pathCommand.Source)
    }

    $candidateDirs += @(
        (Join-Path $env:ProgramFiles "Tesseract-OCR"),
        (Join-Path ${env:ProgramFiles(x86)} "Tesseract-OCR")
    )

    foreach ($candidateDir in $candidateDirs) {
        if (-not $candidateDir) {
            continue
        }

        $candidateExe = Join-Path $candidateDir "tesseract.exe"
        if (-not (Test-Path $candidateExe)) {
            continue
        }

        Write-Host "Copying installed Tesseract runtime from $candidateDir" -ForegroundColor Cyan
        New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null
        Get-ChildItem -LiteralPath $candidateDir -Force | Copy-Item -Destination $vendorDir -Recurse -Force
        return
    }
}

function Sync-Tessdata {
    New-Item -ItemType Directory -Force -Path $tessdataDir | Out-Null

    foreach ($language in $Languages) {
        $target = Join-Path $tessdataDir "$language.traineddata"
        if (Test-Path $target) {
            continue
        }

        $uri = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/$language.traineddata"
        Invoke-DownloadFile -Uri $uri -Destination $target
    }
}

function Remove-UnneededTesseractFiles {
    if (-not (Test-Path $vendorDir)) {
        return
    }

    foreach ($directory in @("doc")) {
        $target = Join-Path $vendorDir $directory
        if (Test-Path $target) {
            Remove-Item -LiteralPath $target -Recurse -Force
        }
    }

    $allowedNames = @(
        "README.md",
        "tesseract.exe"
    )

    Get-ChildItem -LiteralPath $vendorDir -File -Force | ForEach-Object {
        $isRuntimeDll = $_.Extension -ieq ".dll"
        $isAllowedName = $allowedNames -contains $_.Name
        $isLicenseFile = $_.Name -match '^(LICENSE|COPYING|NOTICE)(\..*)?$'
        if (-not ($isRuntimeDll -or $isAllowedName -or $isLicenseFile)) {
            Remove-Item -LiteralPath $_.FullName -Force
        }
    }
}

Install-TesseractRuntime
Sync-Tessdata
Remove-UnneededTesseractFiles

Write-Host "Tesseract runtime ready:" -ForegroundColor Green
Write-Host "  $tesseractExe"
Write-Host "  $tessdataDir"
