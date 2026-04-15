Place the Windows Tesseract runtime here if you want OCR to work on a clean VM without a separate installer.

Expected layout:

```text
vendor/tesseract/
  tesseract.exe
  tessdata/
    eng.traineddata
    tha.traineddata
```

When `tesseract.exe` exists in this folder, `scripts/build_windows.ps1` bundles the whole directory into the PyInstaller output and the app will prefer that bundled copy at runtime.
