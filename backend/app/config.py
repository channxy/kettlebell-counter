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
    processing_fps: float = 15.0  # 15 FPS optimal for kettlebell - captures hand insertion phase
    processing_frame_width: int = 640  # Resize frames for faster processing (0 = no resize)
    
    # Video Quality Thresholds
    min_video_fps: float = 24.0  # Minimum acceptable source FPS
    min_blur_score: float = 100.0  # Laplacian variance threshold for blur detection
    min_athlete_height_ratio: float = 0.50  # Athlete must be at least 50% of frame height
    
    # Pose Estimation
    pose_confidence_threshold: float = 0.5  # Slightly relaxed for real-world videos
    ambiguous_confidence_threshold: float = 0.35
    pose_model_complexity: int = 0  # 0=lite, 1=full, 2=heavy (0 is 3x faster)
    
    # Rep Detection Thresholds - COMPETITION ACCURATE
    min_lockout_angle_degrees: float = 165.0  # Competition standard: near full extension
    min_fixation_frames: int = 4  # ~250ms at 15fps - competition requires visible pause
    max_torso_lean_degrees: float = 20.0  # Competition limit for jerk/LC
    min_overhead_height_ratio: float = 0.75  # Wrist must be clearly above head level
    
    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

