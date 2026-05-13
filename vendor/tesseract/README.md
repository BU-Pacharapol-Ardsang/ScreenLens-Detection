The Windows build script populates this folder automatically so Tesseract OCR works on a clean VM without a separate system installer.

Expected layout:

```text
vendor/tesseract/
  tesseract.exe
  *.dll
  tessdata/
    eng.traineddata
    tha.traineddata
    osd.traineddata
```

When `tesseract.exe` exists in this folder, `scripts/build_windows.ps1` bundles the whole directory into the PyInstaller output and the app will prefer that bundled copy at runtime.

To refresh the bundled runtime manually, run:

```powershell
.\scripts\install_tesseract_vendor.ps1
```
