"""
BhoomiMitra AI — Centralized Language Configuration & Metadata Registry

Defines the 13 supported Indian languages:
1. Telugu (te) - primary/default
2. Hindi (hi)
3. English (en)
4. Tamil (ta)
5. Kannada (kn)
6. Malayalam (ml)
7. Marathi (mr)
8. Bengali (bn)
9. Gujarati (gu)
10. Odia (or)
11. Punjabi (pa)
12. Assamese (as)
13. Urdu (ur)
"""

from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass(frozen=True)
class LanguageMetadata:
    code: str
    display_name: str
    native_name: str
    prompt_name: str
    script: str
    stt_code: str
    supported: bool = True


SUPPORTED_LANGUAGES: Dict[str, LanguageMetadata] = {
    "te": LanguageMetadata(
        code="te",
        display_name="Telugu",
        native_name="తెలుగు",
        prompt_name="Telugu",
        script="telu",
        stt_code="te-IN",
        supported=True,
    ),
    "hi": LanguageMetadata(
        code="hi",
        display_name="Hindi",
        native_name="हिन्दी",
        prompt_name="Hindi",
        script="deva",
        stt_code="hi-IN",
        supported=True,
    ),
    "en": LanguageMetadata(
        code="en",
        display_name="English",
        native_name="English",
        prompt_name="English",
        script="latn",
        stt_code="en-IN",
        supported=True,
    ),
    "ta": LanguageMetadata(
        code="ta",
        display_name="Tamil",
        native_name="தமிழ்",
        prompt_name="Tamil",
        script="taml",
        stt_code="ta-IN",
        supported=True,
    ),
    "kn": LanguageMetadata(
        code="kn",
        display_name="Kannada",
        native_name="ಕನ್ನಡ",
        prompt_name="Kannada",
        script="knda",
        stt_code="kn-IN",
        supported=True,
    ),
    "ml": LanguageMetadata(
        code="ml",
        display_name="Malayalam",
        native_name="മലയാളം",
        prompt_name="Malayalam",
        script="mlym",
        stt_code="ml-IN",
        supported=True,
    ),
    "mr": LanguageMetadata(
        code="mr",
        display_name="Marathi",
        native_name="मराठी",
        prompt_name="Marathi",
        script="deva",
        stt_code="mr-IN",
        supported=True,
    ),
    "bn": LanguageMetadata(
        code="bn",
        display_name="Bengali",
        native_name="বাংলা",
        prompt_name="Bengali",
        script="beng",
        stt_code="bn-IN",
        supported=True,
    ),
    "gu": LanguageMetadata(
        code="gu",
        display_name="Gujarati",
        native_name="ગુજરાતી",
        prompt_name="Gujarati",
        script="gujr",
        stt_code="gu-IN",
        supported=True,
    ),
    "or": LanguageMetadata(
        code="or",
        display_name="Odia",
        native_name="ଓଡ଼ିଆ",
        prompt_name="Odia",
        script="orya",
        stt_code="or-IN",
        supported=True,
    ),
    "pa": LanguageMetadata(
        code="pa",
        display_name="Punjabi",
        native_name="ਪੰਜਾਬੀ",
        prompt_name="Punjabi",
        script="guru",
        stt_code="pa-IN",
        supported=True,
    ),
    "as": LanguageMetadata(
        code="as",
        display_name="Assamese",
        native_name="অসমীয়া",
        prompt_name="Assamese",
        script="beng",
        stt_code="as-IN",
        supported=True,
    ),
    "ur": LanguageMetadata(
        code="ur",
        display_name="Urdu",
        native_name="اردو",
        prompt_name="Urdu",
        script="arab",
        stt_code="ur-IN",
        supported=True,
    ),
}

DEFAULT_LANGUAGE = "te"
FALLBACK_LANGUAGE = "en"


def get_language(code: Optional[str]) -> Optional[LanguageMetadata]:
    """Retrieve language metadata by 2-letter ISO code or STT code."""
    if not code:
        return None
    normalized = code.strip().lower()
    if "-" in normalized:
        normalized = normalized.split("-")[0]
    return SUPPORTED_LANGUAGES.get(normalized)


def is_supported_language(code: Optional[str]) -> bool:
    """Check if the provided language code is among the 13 supported languages."""
    return get_language(code) is not None


def normalize_language_code(code: Optional[str], default: str = FALLBACK_LANGUAGE) -> str:
    """Normalize language code to 2-letter format, returning default if invalid or unsupported."""
    lang = get_language(code)
    return lang.code if lang else default


def list_supported_languages() -> List[LanguageMetadata]:
    """Return all 13 supported languages metadata objects."""
    return list(SUPPORTED_LANGUAGES.values())
