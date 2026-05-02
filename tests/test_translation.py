from time import sleep

from screenlens_detection.translation import (
    GoogleTranslateBackend,
    QueuedTranslationBackend,
    TranslationBackend,
    create_default_translation_backend,
)


class RecordingTranslator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def translate(self, text: str, *, timeout_s: float) -> str:
        self.calls.append((text, timeout_s))
        return f"translated:{text}"


class FlakyTranslator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def translate(self, text: str, *, timeout_s: float) -> str:
        self.calls.append(text)
        raise TimeoutError(text)


def test_translation_backend_limits_network_requests_per_batch() -> None:
    backend = GoogleTranslateBackend(max_requests_per_batch=2, batch_timeout_s=10.0, request_timeout_s=0.25)
    translator = RecordingTranslator()
    backend._get_translator = lambda _source, _target: translator  # type: ignore[method-assign]

    result = backend.translate_batch(
        ["one", "two", "three"],
        source_language_code="eng",
        target_language_code="tha",
    )

    assert result == ["translated:one", "translated:two", ""]
    assert [text for text, _timeout in translator.calls] == ["one", "two"]


def test_translation_backend_cools_down_failed_requests() -> None:
    backend = GoogleTranslateBackend(max_requests_per_batch=2, batch_timeout_s=10.0, request_timeout_s=0.25)
    translator = FlakyTranslator()
    backend._get_translator = lambda _source, _target: translator  # type: ignore[method-assign]

    first_result = backend.translate_batch(
        ["one", "two", "three"],
        source_language_code="eng",
        target_language_code="tha",
    )
    second_result = backend.translate_batch(
        ["one", "two", "three"],
        source_language_code="eng",
        target_language_code="tha",
    )

    assert first_result == ["", "", ""]
    assert second_result == ["", "", ""]
    assert translator.calls == ["one", "two", "three"]


class RecordingBatchTranslationBackend(TranslationBackend):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def is_available(self) -> bool:
        return True

    def translate_batch(
        self,
        texts: list[str],
        *,
        source_language_code: str,
        target_language_code: str,
    ) -> list[str]:
        self.calls.append(list(texts))
        return [f"translated:{text}" for text in texts]


def test_queued_translation_backend_returns_cached_results_after_background_batch() -> None:
    backend = RecordingBatchTranslationBackend()
    queued = QueuedTranslationBackend(backend, max_batch_size=8)

    try:
        first = queued.translate_batch(
            ["one", "two"],
            source_language_code="eng",
            target_language_code="tha",
        )

        resolved = []
        for _attempt in range(20):
            resolved = queued.translate_batch(
                ["one", "two"],
                source_language_code="eng",
                target_language_code="tha",
            )
            if all(resolved):
                break
            sleep(0.05)

        assert first == ["", ""]
        assert resolved == ["translated:one", "translated:two"]
        assert backend.calls == [["one", "two"]]
    finally:
        queued.close()


def test_queued_translation_backend_processes_priority_item_synchronously() -> None:
    backend = RecordingBatchTranslationBackend()
    queued = QueuedTranslationBackend(backend, max_batch_size=8, synchronous_batch_size=1)

    try:
        first = queued.translate_batch(
            ["one", "two"],
            source_language_code="eng",
            target_language_code="tha",
        )

        assert first[0] == "translated:one"
        assert first[1] == ""
        assert backend.calls[0] == ["one"]
    finally:
        queued.close()


def test_create_default_translation_backend_auto_prefers_argos_without_initializing_google(monkeypatch) -> None:
    init_counts = {"argos": 0, "google": 0}

    class DummyArgosBackend(TranslationBackend):
        name = "argos"

        def __init__(self) -> None:
            init_counts["argos"] += 1

        def is_available(self) -> bool:
            return True

        def describe(self) -> str:
            return "Argos Translate (Offline)"

    class DummyGoogleBackend(TranslationBackend):
        name = "google"

        def __init__(self) -> None:
            init_counts["google"] += 1

        def is_available(self) -> bool:
            return True

    monkeypatch.setattr("screenlens_detection.translation.ArgosTranslateBackend", DummyArgosBackend)
    monkeypatch.setattr("screenlens_detection.translation.GoogleTranslateBackend", DummyGoogleBackend)
    monkeypatch.setattr(
        "screenlens_detection.translation.QueuedTranslationBackend",
        lambda backend, **_kwargs: backend,
    )

    backend = create_default_translation_backend(mode="auto")

    assert backend.describe() == "Argos Translate (Offline)"
    assert init_counts == {"argos": 1, "google": 0}
