from __future__ import annotations

from functools import lru_cache

CUDA_EXECUTION_PROVIDER = "CUDAExecutionProvider"


@lru_cache(maxsize=1)
def onnxruntime_available_providers() -> tuple[str, ...]:
    try:
        import onnxruntime
    except Exception:
        return ()

    try:
        providers = onnxruntime.get_available_providers()
    except Exception:
        return ()
    return tuple(str(provider) for provider in providers)


def onnxruntime_cuda_available(device_preference: str | None) -> bool:
    normalized = (device_preference or "auto").strip().casefold()
    if normalized == "cpu":
        return False
    if normalized not in {"auto", "gpu", "cuda", "nvidia"}:
        return False
    return CUDA_EXECUTION_PROVIDER in onnxruntime_available_providers()


def onnxruntime_provider_summary() -> str:
    providers = onnxruntime_available_providers()
    if not providers:
        return "unavailable"
    return ", ".join(providers)


def short_runtime_error(message: str, *, limit: int = 120) -> str:
    normalized = " ".join(message.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3]}..."
