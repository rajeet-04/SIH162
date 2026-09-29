from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """Validated paths and runtime settings loaded from YAML."""

    model_config = ConfigDict(extra="allow")

    sources: dict[str, Path]
    artifacts_root: Path = Path("data")
    seed: int = 26162
    timezone: str = "UTC"
    ranking_fraction: float = Field(default=0.10, gt=0, lt=1)


def load_settings(path: Path) -> Settings:
    """Load source configuration without touching any external data."""
    payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Settings.model_validate(payload)
