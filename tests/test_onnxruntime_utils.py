import sys
from types import SimpleNamespace

from screenlens_detection.onnxruntime_utils import (
    onnxruntime_available_providers,
    onnxruntime_cuda_available,
    onnxruntime_provider_summary,
    short_runtime_error,
)


def test_onnxruntime_cuda_available_accepts_auto_when_cuda_provider_exists(monkeypatch) -> None:
    fake_onnxruntime = SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)
    onnxruntime_available_providers.cache_clear()

    try:
        assert onnxruntime_cuda_available("auto") is True
        assert onnxruntime_cuda_available("gpu") is True
        assert onnxruntime_cuda_available("cpu") is False
        assert onnxruntime_provider_summary() == "CUDAExecutionProvider, CPUExecutionProvider"
    finally:
        onnxruntime_available_providers.cache_clear()


def test_onnxruntime_cuda_available_is_false_without_provider(monkeypatch) -> None:
    fake_onnxruntime = SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)
    onnxruntime_available_providers.cache_clear()

    try:
        assert onnxruntime_cuda_available("auto") is False
        assert onnxruntime_cuda_available("gpu") is False
    finally:
        onnxruntime_available_providers.cache_clear()


def test_short_runtime_error_compacts_whitespace_and_truncates() -> None:
    assert short_runtime_error("CUDA\nsession\tfailed", limit=20) == "CUDA session failed"
    assert short_runtime_error("x" * 40, limit=12) == "xxxxxxxxx..."
