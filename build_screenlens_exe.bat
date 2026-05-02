@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "BUILD_SCRIPT=%SCRIPT_DIR%scripts\build_windows.ps1"
set "OUTPUT_EXE=%SCRIPT_DIR%dist\ScreenLens\ScreenLens.exe"
set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

if /i "%~1"=="/?" goto :usage
if /i "%~1"=="-h" goto :usage
if /i "%~1"=="--help" goto :usage

if not exist "%POWERSHELL_EXE%" (
    echo PowerShell was not found at "%POWERSHELL_EXE%".
    exit /b 1
)

if not exist "%BUILD_SCRIPT%" (
    echo Build script was not found:
    echo   %BUILD_SCRIPT%
    exit /b 1
)

pushd "%SCRIPT_DIR%" >nul
if errorlevel 1 (
    echo Failed to enter repository directory:
    echo   %SCRIPT_DIR%
    exit /b 1
)

echo Building ScreenLens executable...
echo Build script: %BUILD_SCRIPT%
echo.

"%POWERSHELL_EXE%" -NoProfile -ExecutionPolicy Bypass -File "%BUILD_SCRIPT%" -Clean %*
set "EXIT_CODE=%ERRORLEVEL%"

popd >nul

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Build failed with exit code %EXIT_CODE%.
    exit /b %EXIT_CODE%
)

echo.
echo Build finished. Output:
echo   %OUTPUT_EXE%
exit /b 0

:usage
echo Build ScreenLens as a Windows executable.
echo.
echo Usage:
echo   %~nx0 [build_windows.ps1 options]
echo.
echo Examples:
echo   %~nx0
echo   %~nx0 -TorchRuntime cpu
echo   %~nx0 -TorchRuntime gpu
echo   %~nx0 -PythonExe C:\Python313\python.exe -TorchRuntime auto
echo.
echo Notes:
echo   - Clean build is enabled automatically.
echo   - Extra options are forwarded to scripts\build_windows.ps1.
exit /b 0
