# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.building.build_main import Analysis, COLLECT, EXE, PYZ
from PyInstaller.building.datastruct import Tree


project_root = Path(SPECPATH).resolve()
datas = []

vendor_tesseract = project_root / "vendor" / "tesseract" / "tesseract.exe"
if vendor_tesseract.is_file():
    datas.extend(Tree(str(vendor_tesseract.parent), prefix="tesseract"))


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
