"""Workout schemas."""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator

from app.models.workout import LiftType


class WorkoutCreate(BaseModel):
    """Schema for creating a workout (video upload)."""
    lift_type: str = Field(..., description="Type of lift: jerk, long_cycle, snatch, or auto (auto-detect)")
    workout_date: Optional[datetime] = None
    
    @field_validator("lift_type")
    @classmethod
    def validate_lift_type(cls, v: str) -> str:
        valid_types = LiftType.all() + ["auto"]
        if v not in valid_types:
            raise ValueError(f"lift_type must be one of: {valid_types}")
        return v


class WorkoutResponse(BaseModel):
    """Schema for workout list response."""
    id: str  # String ID for SQLite compatibility
    lift_type: str
    detected_lift_type: Optional[str] = None  # Auto-detected lift type
    workout_date: datetime
    processing_status: str
    processing_progress: float
    processing_started_at: Optional[datetime]
    
    # CRITICAL: Separated rep counts
    total_attempts: int
    valid_reps: int
    no_reps: int
    ambiguous_reps: int
    
    video_duration_seconds: Optional[float]
    video_filename: Optional[str] = None
    thumbnail_path: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class WorkoutListResponse(BaseModel):
    """Schema for paginated workout list."""
    items: List[WorkoutResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class TimelinePoint(BaseModel):
    """
    Timeline overlay point for video visualization.
    
    Color coding:
    - green: valid rep
    - red: no-rep
    - yellow: ambiguous
    """
    timestamp_start: float
    timestamp_end: float
    classification: str
    color: str  # "green", "red", "yellow"
    rep_number: int
    failure_reasons: Optional[List[str]] = None


class WorkoutDetailResponse(BaseModel):
    """Schema for detailed workout response with all rep attempts."""
    id: str  # String ID for SQLite compatibility
    user_id: str
    lift_type: str
    workout_date: datetime
    duration_seconds: Optional[float]
    
    # CRITICAL: Separated rep counts
    total_attempts: int
    valid_reps: int
    no_reps: int
    ambiguous_reps: int
    
    # Processing info
    processing_status: str
    processing_progress: float
    processing_started_at: Optional[datetime]
    processing_error: Optional[str]
    video_duration_seconds: Optional[float]
    video_filename: Optional[str] = None
    thumbnail_path: Optional[str] = None
    
    # User notes/log
    notes: Optional[str] = None
    perceived_effort: Optional[int] = None  # RPE 1-10
    mood: Optional[str] = None  # How user felt: great, good, okay, tired, heavy
    
    # Analytics
    analytics_summary: Optional[Dict[str, Any]]
    
    # Rep attempts (for detailed view)
    rep_attempts: List["RepAttemptResponse"]
    
    # Timeline data for video overlay
    timeline: List[TimelinePoint]
    
    # Health export status
    exported_to_health: bool
    health_export_date: Optional[datetime]
    
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class WorkoutUpdateRequest(BaseModel):
    """Schema for updating workout notes/log."""
    notes: Optional[str] = None
    perceived_effort: Optional[int] = None  # RPE 1-10 scale
    
    class Config:
        from_attributes = True


# Forward reference update - import at runtime to avoid circular import
from app.schemas.rep_attempt import RepAttemptResponse
WorkoutDetailResponse.model_rebuild()

