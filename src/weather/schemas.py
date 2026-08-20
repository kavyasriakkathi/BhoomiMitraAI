"""
BhoomiMitra AI — Weather Schemas

Pydantic models for validation and response formatting.
"""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class WeatherCondition(BaseModel):
    """Current weather condition details."""
    temp: float = Field(..., description="Temperature in Celsius")
    feels_like: float = Field(..., description="Feels-like temperature in Celsius")
    humidity: int = Field(..., ge=0, le=100, description="Humidity percentage")
    wind_speed: float = Field(..., ge=0.0, description="Wind speed in km/h")
    description: str = Field(..., description="Weather condition description, e.g. 'Light Rain'")
    condition_code: int = Field(..., description="OpenWeatherMap condition code id")


class WeatherForecastItem(BaseModel):
    """Forecast item representing a future weather point."""
    dt_txt: str = Field(..., description="Forecast timestamp string")
    temp: float = Field(..., description="Temperature in Celsius")
    humidity: int = Field(..., ge=0, le=100)
    description: str = Field(..., description="Condition description")
    condition_code: int = Field(..., description="Condition code id")


class WeatherForecastResponse(BaseModel):
    """Full current weather and forecast payload."""
    location_name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    current: WeatherCondition
    forecast: List[WeatherForecastItem]
    data_available: bool
    data_freshness_minutes: Optional[float] = None
    source_note: str
    is_live: bool

    model_config = ConfigDict(from_attributes=True)
