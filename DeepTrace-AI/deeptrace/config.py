"""Central, configurable settings for DeepTrace AI.

All paths are derived from the project root with ``pathlib`` - nothing is
hard-coded. Override any value by creating ``Settings(...)`` yourself, or set
the ``DEEPTRACE_DATA_DIR`` / ``DEEPTRACE_LOG_LEVEL`` environment variables.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

#: extension -> media type
ALLOWED_EXTENSIONS: Dict[str, str] = {
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".mp4": "video",
    ".mov": "video",
    ".wav": "audio",
    ".mp3": "audio",
}

#: extension -> MIME type we expect for that extension
EXPECTED_MIME: Dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}

#: extension -> file-signature family detected from the first bytes
EXPECTED_SIGNATURE: Dict[str, str] = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".mp4": "isobmff",
    ".mov": "isobmff",
    ".wav": "wav",
    ".mp3": "mp3",
}


class Settings(BaseModel):
    """Runtime configuration (validation limits + processing parameters)."""

    model_config = ConfigDict(validate_assignment=True)

    # --- directories -----------------------------------------------------
    data_dir: Path = Field(default_factory=lambda: Path(os.environ.get("DEEPTRACE_DATA_DIR", PROJECT_ROOT / "data")))
    log_dir: Path = PROJECT_ROOT / "logs"
    log_level: str = Field(default_factory=lambda: os.environ.get("DEEPTRACE_LOG_LEVEL", "INFO"))
    db_path: Path = Field(default_factory=lambda: Path(os.environ.get("DEEPTRACE_DB_PATH", PROJECT_ROOT / "data" / "deeptrace.db")))
    models_dir: Path = PROJECT_ROOT / "models"

    # --- validation limits -----------------------------------------------
    max_image_size_mb: float = Field(25.0, gt=0)
    max_video_size_mb: float = Field(500.0, gt=0)
    max_audio_size_mb: float = Field(100.0, gt=0)
    max_video_duration_s: float = Field(300.0, gt=0)
    max_audio_duration_s: float = Field(600.0, gt=0)
    max_image_pixels: int = Field(50_000_000, gt=0)
    min_image_side: int = Field(32, gt=0)
    max_video_side: int = Field(7680, gt=0)
    max_filename_length: int = Field(255, gt=0)

    # --- processing parameters (used by later Phase 1 modules) ------------
    processing_max_seconds: float = Field(30.0, gt=0)
    max_sampled_fps: float = Field(10.0, gt=0)
    image_max_side: int = Field(1280, gt=0)
    audio_sample_rate: int = Field(16000, gt=0)
    audio_segment_seconds: float = Field(3.0, gt=0)
    audio_segment_overlap_seconds: float = Field(0.0, ge=0)

    # --- derived paths -----------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def temp_dir(self) -> Path:
        return self.data_dir / "temp"

    def max_size_bytes(self, media_type: str) -> int:
        """Return the size limit in bytes for ``image`` / ``video`` / ``audio``."""
        mb = {
            "image": self.max_image_size_mb,
            "video": self.max_video_size_mb,
            "audio": self.max_audio_size_mb,
        }[media_type]
        return int(mb * 1024 * 1024)

    def ensure_directories(self) -> None:
        """Create all working directories if they do not exist."""
        for directory in (self.raw_dir, self.processed_dir, self.temp_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the shared default settings instance."""
    return Settings()
