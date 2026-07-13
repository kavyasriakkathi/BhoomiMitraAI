from src.language.service import LanguageService

def get_language_service() -> LanguageService:
    """
    Dependency provider for the LanguageService.
    Instantiates and returns the service to handle audio transcription and translation.
    """
    return LanguageService()
