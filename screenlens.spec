# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.building.build_main import Analysis, COLLECT, EXE, PYZ


project_root = Path(SPECPATH).resolve()
datas = []

vendor_tesseract = project_root / "vendor" / "tesseract"
if vendor_tesseract.is_dir():
    datas.append((str(vendor_tesseract), "vendor/tesseract"))

vendor_argos = project_root / "vendor" / "argos"
if vendor_argos.is_dir():
    datas.append((str(vendor_argos), "vendor/argos"))


a = Analysis(
    [str(project_root / "src" / "screenlens_detection" / "app_entry.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=[],
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
