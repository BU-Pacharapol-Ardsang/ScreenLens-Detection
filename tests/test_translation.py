from screenlens_detection.translation import GoogleTranslateBackend


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
