# Language Module — Multilingual Configuration, Detection, STT, TTS
from src.language.languages import (
    SUPPORTED_LANGUAGES,
    LanguageMetadata,
    DEFAULT_LANGUAGE,
    FALLBACK_LANGUAGE,
    get_language,
    is_supported_language,
    normalize_language_code,
    list_supported_languages,
)
from src.language.detector import (
    detect_language,
    detect_language_with_confidence,
)

__all__ = [
    "SUPPORTED_LANGUAGES",
    "LanguageMetadata",
    "DEFAULT_LANGUAGE",
    "FALLBACK_LANGUAGE",
    "get_language",
    "is_supported_language",
    "normalize_language_code",
    "list_supported_languages",
    "detect_language",
    "detect_language_with_confidence",
]
