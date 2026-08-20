"""
BhoomiMitra AI — Weather Dependency Injection

Provides dependencies for FastAPI routes.
"""
from src.config import get_settings
from src.weather.openweather_client import OpenWeatherClient
from src.weather.service import WeatherService


async def get_weather_service() -> WeatherService:
    settings = get_settings()
    client = OpenWeatherClient(
        api_key=settings.openweather_api_key,
        api_url=settings.openweather_api_url,
        cache_ttl_seconds=settings.weather_cache_ttl_seconds,
    )
    return WeatherService(client=client)
