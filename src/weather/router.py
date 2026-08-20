"""
BhoomiMitra AI — Weather Forecast Router

REST endpoint for querying the weather forecast.
"""
from typing import Optional

from fastapi import APIRouter, Depends, Query, status

from src.weather.schemas import WeatherForecastResponse
from src.weather.service import WeatherService
from src.weather.dependencies import get_weather_service

router = APIRouter()


@router.get(
    "/forecast",
    response_model=WeatherForecastResponse,
    status_code=status.HTTP_200_OK,
    summary="Get weather forecast for a location",
    description=(
        "Returns the current weather and 5-day forecast. "
        "Attempts to query the live OpenWeatherMap API, caching results in Redis. "
        "Falls back to simulated mock data if OpenWeatherMap is unconfigured."
    ),
)
async def get_weather_forecast(
    latitude: Optional[float] = Query(None, description="Latitude, e.g. 17.3850"),
    longitude: Optional[float] = Query(None, description="Longitude, e.g. 78.4867"),
    district: Optional[str] = Query(None, description="Filter by district name, e.g. 'Warangal'"),
    state: Optional[str] = Query(None, description="Filter by state name, e.g. 'Telangana'"),
    service: WeatherService = Depends(get_weather_service),
) -> WeatherForecastResponse:
    return await service.get_weather_for_query(
        latitude=latitude,
        longitude=longitude,
        district=district,
        state=state,
    )
