from __future__ import annotations

from dataclasses import dataclass

from .languages import resolve_translation_language

try:
    from deep_translator import GoogleTranslator
except ImportError:  # pragma: no cover - optional runtime dependency
    GoogleTranslator = None


@dataclass(slots=True, frozen=True)
class TranslationResult:
    text: str = ""


class TranslationBackend:
    name = "disabled"

    def is_available(self) -> bool:
        return False

    def describe(self) -> str:
        return "Translation unavailable"

    def translate(
        self,
        text: str,
        *,
        source_language_code: str,
        target_language_code: str,
    ) -> TranslationResult:
        return TranslationResult(text="")

    def translate_batch(
        self,
        texts: list[str],
        *,
        source_language_code: str,
        target_language_code: str,
    ) -> list[str]:
        return [
            self.translate(
                text,
                source_language_code=source_language_code,
                target_language_code=target_language_code,
            ).text
            for text in texts
        ]


class NoOpTranslationBackend(TranslationBackend):
    name = "disabled"

    def describe(self) -> str:
        return "Translation unavailable"


class GoogleTranslateBackend(TranslationBackend):
    name = "google-translate"

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, str], str] = {}
        self._translators: dict[tuple[str, str], GoogleTranslator] = {}

    def is_available(self) -> bool:
        return GoogleTranslator is not None

    def describe(self) -> str:
        if self.is_available():
            return "Google Translate"
        return "Install deep-translator to enable translation"

    def translate(
        self,
        text: str,
        *,
        source_language_code: str,
        target_language_code: str,
    ) -> TranslationResult:
        translated = self.translate_batch(
            [text],
            source_language_code=source_language_code,
            target_language_code=target_language_code,
        )[0]
        return TranslationResult(text=translated)

    def translate_batch(
        self,
        texts: list[str],
        *,
        source_language_code: str,
        target_language_code: str,
    ) -> list[str]:
        if not texts:
            return []

        normalized_texts = [" ".join(text.split()) for text in texts]
        source_language = resolve_translation_language(source_language_code)
        target_language = resolve_translation_language(target_language_code)
        if target_language == "auto":
            return [""] * len(texts)
        if source_language != "auto" and source_language == target_language:
            return normalized_texts
        if not self.is_available():
            return [""] * len(texts)

        results = [""] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for index, normalized in enumerate(normalized_texts):
            if not normalized:
                continue

            cache_key = (normalized, source_language, target_language)
            cached = self._cache.get(cache_key)
            if cached is not None:
                results[index] = cached
                continue

            missing_indices.append(index)
            missing_texts.append(normalized)

        if not missing_texts:
            return results

        translator = self._get_translator(source_language, target_language)
        try:
            translated_batch = translator.translate_batch(missing_texts)
        except Exception:
            translated_batch = self._translate_batch_fallback(
                translator,
                missing_texts,
            )

        for index, translated in zip(missing_indices, translated_batch, strict=False):
            normalized_source = normalized_texts[index]
            normalized_translated = " ".join((translated or "").split())
            self._cache[(normalized_source, source_language, target_language)] = normalized_translated
            results[index] = normalized_translated

        return results

    def _get_translator(self, source_language: str, target_language: str) -> GoogleTranslator:
        cache_key = (source_language, target_language)
        translator = self._translators.get(cache_key)
        if translator is None:
            translator = GoogleTranslator(source=source_language, target=target_language)
            self._translators[cache_key] = translator
        return translator

    @staticmethod
    def _translate_batch_fallback(translator: GoogleTranslator, texts: list[str]) -> list[str]:
        results: list[str] = []
        for text in texts:
            try:
                results.append(translator.translate(text))
            except Exception:
                results.append("")
        return results


def create_default_translation_backend() -> TranslationBackend:
    backend = GoogleTranslateBackend()
    if backend.is_available():
        return backend
    return NoOpTranslationBackend()
