from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT_DIR.parent


class AppSettings(BaseSettings):
    app_name: str = Field(
        default="ASR NEC Demo API",
        validation_alias=AliasChoices("APP_NAME", "NEC_APP_NAME"),
    )
    api_host: str = Field(default="127.0.0.1", validation_alias="API_HOST")
    api_port: int = Field(default=8016, validation_alias="API_PORT")
    api_reload: bool = Field(default=True, validation_alias="API_RELOAD")
    root_path: str = Field(default="", validation_alias="ROOT_PATH")
    frontend_dist_dir: Path = Field(
        default=ROOT_DIR / "ui" / "dist",
        validation_alias=AliasChoices("FRONTEND_DIST_DIR", "NEC_FRONTEND_DIST_DIR"),
    )
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default=["http://127.0.0.1:5009", "http://localhost:5009"],
        validation_alias=AliasChoices("CORS_ORIGINS", "NEC_CORS_ORIGINS"),
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> list[str]:
        return _split_csv(value, ["http://127.0.0.1:5009", "http://localhost:5009"])

    @field_validator("root_path")
    @classmethod
    def _normalize_root_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or normalized == "/":
            return ""
        return f"/{normalized.strip('/')}"


class DatabaseSettings(BaseSettings):
    database_url: str = Field(
        default=f"sqlite:///{(ROOT_DIR / 'storage' / 'app.db').as_posix()}",
        validation_alias=AliasChoices("DATABASE_URL", "NEC_DATABASE_URL"),
    )


class StorageSettings(BaseSettings):
    storage_dir: Path = Field(
        default=ROOT_DIR / "storage",
        validation_alias=AliasChoices("STORAGE_DIR", "NEC_STORAGE_DIR"),
    )
    upload_dir: Path = Field(
        default=ROOT_DIR / "storage" / "uploads",
        validation_alias=AliasChoices("UPLOAD_DIR", "NEC_UPLOAD_DIR"),
    )


class NecModelSettings(BaseSettings):
    nec_skip_model_load: bool = Field(
        default=False, validation_alias="NEC_SKIP_MODEL_LOAD"
    )
    nec_device: str = Field(default="", validation_alias="NEC_DEVICE")
    whisper_model_path: Path = Field(
        default=WORKSPACE_ROOT / "weights" / "base.pt",
        validation_alias="WHISPER_MODEL_PATH",
    )
    ss_checkpoint_path: Path = Field(
        default=WORKSPACE_ROOT / "runs" / "ss_full_seed_20260724" / "best.pt",
        validation_alias="SS_CHECKPOINT_PATH",
    )
    gl_checkpoint_path: Path = Field(
        default=WORKSPACE_ROOT / "runs" / "gl_augmented_aligned_e5" / "best.pt",
        validation_alias="GL_CHECKPOINT_PATH",
    )
    nec_data_dir: Path = Field(
        default=WORKSPACE_ROOT / "data" / "speech_searcher",
        validation_alias="NEC_DATA_DIR",
    )
    nec_feature_dir: Path = Field(
        default=WORKSPACE_ROOT / "data" / "speech_searcher" / "ss_features_full",
        validation_alias="NEC_FEATURE_DIR",
    )
    nec_runs_dir: Path = Field(
        default=WORKSPACE_ROOT / "runs",
        validation_alias="NEC_RUNS_DIR",
    )
    examples_file: Path = Field(
        default=ROOT_DIR / "data" / "examples.json",
        validation_alias="EXAMPLES_FILE",
    )
    examples_audio_dir: Path = Field(
        default=WORKSPACE_ROOT / "data" / "speech_searcher" / "audio_full",
        validation_alias="EXAMPLES_AUDIO_DIR",
    )
    default_top_k: int = Field(default=5, validation_alias="DEFAULT_TOP_K")
    default_threshold: float = Field(default=0.3, validation_alias="DEFAULT_THRESHOLD")


class LoggingSettings(BaseSettings):
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL", "NEC_LOG_LEVEL"),
    )

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if normalized not in valid_levels:
            raise ValueError(f"Invalid LOG_LEVEL: {value}")
        return normalized


class Settings(
    AppSettings,
    DatabaseSettings,
    StorageSettings,
    NecModelSettings,
    LoggingSettings,
):
    model_config = SettingsConfigDict(
        env_file=(ROOT_DIR.parent / ".env", ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_parse_none_str="",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _split_csv(value: Any, default: list[str]) -> list[str]:
    if not value:
        return default
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]
