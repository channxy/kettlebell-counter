"""Application configuration."""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "Kettlebell Counter"
    debug: bool = False
    api_prefix: str = "/api"
    
    # Database (SQLite for local dev, PostgreSQL for production)
    database_url: str = "sqlite+aiosqlite:///./kettlebell.db"
    database_url_sync: str = "sqlite:///./kettlebell.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # JWT Authentication
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days
    
    # Storage
    storage_backend: str = "local"  # "local" or "s3"
    upload_dir: str = "./uploads"
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    
    # Video Processing
    max_video_duration_minutes: int = 60
    max_video_size_mb: int = 5000  # 5GB
    processing_fps: float = 6.0  # Reduced from 12 - 6fps is enough for kettlebell movements
    processing_frame_width: int = 640  # Resize frames for faster processing (0 = no resize)
    
    # Pose Estimation
    pose_confidence_threshold: float = 0.6
    ambiguous_confidence_threshold: float = 0.4
    pose_model_complexity: int = 0  # 0=lite, 1=full, 2=heavy (0 is 3x faster)
    
    # Rep Detection Thresholds
    min_lockout_angle_degrees: float = 150.0  # Elbow angle for valid lockout (lowered from 165)
    min_fixation_frames: int = 2  # Minimum frames at lockout (lowered from 3)
    max_torso_lean_degrees: float = 25.0  # Allow more lean for snatch (raised from 15)
    min_overhead_height_ratio: float = 0.70  # KB height relative to body (lowered from 0.85)
    
    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

