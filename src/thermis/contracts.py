from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ThermalEvent(BaseModel):
    """Canonical point event shared by adapters, features, and the API."""

    model_config = ConfigDict(extra="allow")

    event_id: str
    timestamp_utc: datetime
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    brightness_temperature: float | None = None
    frp: float = Field(ge=0)
    frp_uncertainty: float | None = Field(default=None, ge=0)
    source_dataset: str
    source_record_id: str

    @field_validator("timestamp_utc")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware")
        return value
