import os
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from src.config import Settings
from src.language.service import LanguageService, resolve_google_credentials
from src.core.exceptions import BhoomiMitraException


@pytest.fixture
def mock_service_account_dict():
    return {
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "123456",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC3\n-----END PRIVATE KEY-----\n",
        "client_email": "test@test-project.iam.gserviceaccount.com",
        "client_id": "987654321",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/test",
    }


def test_resolve_google_credentials_from_json_string(mock_service_account_dict):
    settings = Settings(
        google_application_credentials_json=json.dumps(mock_service_account_dict),
        google_application_credentials="",
    )
    with patch("src.language.service.service_account.Credentials.from_service_account_info") as mock_from_info:
        mock_from_info.return_value = MagicMock()
        creds = resolve_google_credentials(settings)
        assert creds is not None
        mock_from_info.assert_called_once()


def test_resolve_google_credentials_from_raw_json_in_path_field(mock_service_account_dict):
    settings = Settings(
        google_application_credentials=json.dumps(mock_service_account_dict),
        google_application_credentials_json="",
    )
    with patch("src.language.service.service_account.Credentials.from_service_account_info") as mock_from_info:
        mock_from_info.return_value = MagicMock()
        creds = resolve_google_credentials(settings)
        assert creds is not None
        mock_from_info.assert_called_once()


def test_resolve_google_credentials_from_existing_file(tmp_path, mock_service_account_dict):
    creds_file = tmp_path / "service_account.json"
    creds_file.write_text(json.dumps(mock_service_account_dict), encoding="utf-8")

    settings = Settings(
        google_application_credentials=str(creds_file),
        google_application_credentials_json="",
    )
    with patch("src.language.service.service_account.Credentials.from_service_account_file") as mock_from_file:
        mock_from_file.return_value = MagicMock()
        creds = resolve_google_credentials(settings)
        assert creds is not None
        mock_from_file.assert_called_once_with(str(creds_file))


def test_resolve_google_credentials_nonexistent_file_does_not_poison_env():
    non_existent = "/non/existent/path/creds.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = non_existent

    settings = Settings(
        google_application_credentials=non_existent,
        google_application_credentials_json="",
    )

    creds = resolve_google_credentials(settings)
    assert creds is None
    # Environment variable pointing to non-existent file should have been removed
    assert os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") != non_existent


def test_resolve_google_credentials_render_secret_discovery(tmp_path, mock_service_account_dict, monkeypatch):
    render_secrets_dir = tmp_path / "render_secrets"
    render_secrets_dir.mkdir()
    secret_file = render_secrets_dir / "gen-lang-client-0304347321-3ef730155599.json"
    secret_file.write_text(json.dumps(mock_service_account_dict), encoding="utf-8")

    monkeypatch.setattr("src.language.service.Path", lambda p: render_secrets_dir if p == "/etc/secrets" else Path(p))

    settings = Settings(
        google_application_credentials="",
        google_application_credentials_json="",
    )

    with patch("src.language.service.service_account.Credentials.from_service_account_file") as mock_from_file:
        mock_from_file.return_value = MagicMock()
        creds = resolve_google_credentials(settings)
        assert creds is not None


def test_resolve_google_credentials_render_service_account_mounted(tmp_path, mock_service_account_dict):
    secret_file = tmp_path / "service-account.json"
    secret_file.write_text(json.dumps(mock_service_account_dict), encoding="utf-8")

    settings = Settings(
        google_application_credentials=str(secret_file),
        google_application_credentials_json="",
    )

    with patch("src.language.service.service_account.Credentials.from_service_account_file") as mock_from_file:
        mock_creds = MagicMock()
        mock_from_file.return_value = mock_creds
        creds = resolve_google_credentials(settings)
        assert creds is mock_creds
        mock_from_file.assert_called_once_with(str(secret_file))


def test_language_service_google_client_loads_credentials():
    mock_creds = MagicMock()
    with patch("src.language.service.resolve_google_credentials", return_value=mock_creds), \
         patch("src.language.service.speech.SpeechAsyncClient") as mock_speech_client:
        service = LanguageService()
        client = service.google_client
        mock_speech_client.assert_called_once_with(credentials=mock_creds)
        assert client == mock_speech_client.return_value



@pytest.mark.asyncio
async def test_language_service_empty_audio_error():
    service = LanguageService()
    with pytest.raises(BhoomiMitraException) as exc_info:
        await service.transcribe_audio(b"", "audio/ogg")
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_language_service_google_stt_success():
    service = LanguageService()
    
    mock_client = AsyncMock()
    mock_result = MagicMock()
    mock_alternative = MagicMock()
    mock_alternative.transcript = "వరి పంటలో తెగులు వచ్చింది"
    mock_alternative.confidence = 0.94
    mock_result.alternatives = [mock_alternative]
    mock_result.language_code = "te-IN"

    mock_response = MagicMock()
    mock_response.results = [mock_result]
    mock_client.recognize.return_value = mock_response

    service._google_client = mock_client

    result = await service.transcribe_audio(b"fake-audio-bytes", "audio/ogg")
    assert result.transcription_text == "వరి పంటలో తెగులు వచ్చింది"
    assert result.detected_language == "te-IN"
    assert result.confidence == 0.94
    assert result.provider_used == "google"
