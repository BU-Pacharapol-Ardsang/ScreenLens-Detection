from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class LanguageOption:
    code: str
    label: str
    ocr_language: str
    translation_language: str


SOURCE_LANGUAGE_OPTIONS: tuple[LanguageOption, ...] = (
    LanguageOption(code="auto", label="Auto detect", ocr_language="tha+eng", translation_language="auto"),
    LanguageOption(code="eng", label="English", ocr_language="eng", translation_language="en"),
    LanguageOption(code="tha", label="Thai", ocr_language="tha", translation_language="th"),
    LanguageOption(code="tha+eng", label="Thai + English", ocr_language="tha+eng", translation_language="auto"),
)

TARGET_LANGUAGE_OPTIONS: tuple[LanguageOption, ...] = (
    LanguageOption(code="tha", label="Thai", ocr_language="tha", translation_language="th"),
    LanguageOption(code="eng", label="English", ocr_language="eng", translation_language="en"),
)

LANGUAGE_LABELS: dict[str, str] = {
    "auto": "Auto detect",
    "eng": "English",
    "tha": "Thai",
    "tha+eng": "Thai + English",
    "mixed": "Mixed (Thai + English)",
    "unknown": "Unknown",
}

_THAI_PATTERN = re.compile(r"[\u0E00-\u0E7F]")
_LATIN_PATTERN = re.compile(r"[A-Za-z]")


def source_language_options() -> tuple[LanguageOption, ...]:
    return SOURCE_LANGUAGE_OPTIONS


def target_language_options() -> tuple[LanguageOption, ...]:
    return TARGET_LANGUAGE_OPTIONS


def get_source_language_option(code: str) -> LanguageOption:
    for option in SOURCE_LANGUAGE_OPTIONS:
        if option.code == code:
            return option
    return SOURCE_LANGUAGE_OPTIONS[0]


def get_target_language_option(code: str) -> LanguageOption:
    for option in TARGET_LANGUAGE_OPTIONS:
        if option.code == code:
            return option
    return TARGET_LANGUAGE_OPTIONS[0]


def resolve_ocr_language(source_code: str) -> str:
    return get_source_language_option(source_code).ocr_language


def resolve_translation_language(code: str) -> str:
    if code in {"mixed", "unknown"}:
        return "auto"

    for option in (*SOURCE_LANGUAGE_OPTIONS, *TARGET_LANGUAGE_OPTIONS):
        if option.code == code:
            return option.translation_language
    return "auto"


def language_label(code: str) -> str:
    return LANGUAGE_LABELS.get(code, code.upper())


def detect_language_code(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return "unknown"

    thai_count = len(_THAI_PATTERN.findall(normalized))
    latin_count = len(_LATIN_PATTERN.findall(normalized))

    if thai_count and latin_count:
        return "mixed"
    if thai_count:
        return "tha"
    if latin_count:
        return "eng"
    return "unknown"
