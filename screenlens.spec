# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.building.build_main import Analysis, COLLECT, EXE, PYZ
from PyInstaller.utils.hooks import collect_data_files, collect_submodules


project_root = Path(SPECPATH).resolve()
datas = []
hiddenimports = []


def collect_optional_datas(package_name):
    try:
        datas.extend(collect_data_files(package_name))
    except Exception as exc:
        print(f"Skipping optional package data for {package_name}: {exc}")


def collect_optional_submodules(package_name):
    try:
        hiddenimports.extend(collect_submodules(package_name))
    except Exception as exc:
        print(f"Skipping optional package imports for {package_name}: {exc}")

vendor_tesseract = project_root / "vendor" / "tesseract"
if vendor_tesseract.is_dir():
    datas.append((str(vendor_tesseract), "vendor/tesseract"))

vendor_argos = project_root / "vendor" / "argos"
if vendor_argos.is_dir():
    datas.append((str(vendor_argos), "vendor/argos"))

collect_optional_datas("rapidocr")
collect_optional_submodules("rapidocr.ch_ppocr_det")
collect_optional_submodules("rapidocr.inference_engine.onnxruntime")
collect_optional_submodules("rapidocr.utils")


a = Analysis(
    [str(project_root / "src" / "screenlens_detection" / "app_entry.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ScreenLens",
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ScreenLens",
)
