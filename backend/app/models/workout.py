"""Workout model."""

import uuid
import json
from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, Integer, Float, ForeignKey, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class LiftType:
    """Supported kettlebell lift types."""
    JERK = "jerk"
    LONG_CYCLE = "long_cycle"
    SNATCH = "snatch"
    
    @classmethod
    def all(cls) -> List[str]:
        return [cls.JERK, cls.LONG_CYCLE, cls.SNATCH]


class ProcessingStatus:
    """Video processing status."""
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"


class Workout(Base, TimestampMixin):
    """
    Workout model storing video analysis results.
    
    CRITICAL: Enforces the separation between total_attempts, valid_reps, and no_reps.
    Invariant: total_attempts = valid_reps + no_reps + ambiguous_reps
    """
    
    __tablename__ = "workouts"
    
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Video metadata
    video_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    video_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    video_duration_seconds: Mapped[float] = mapped_column(Float, nullable=True)
    video_fps: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Processing status
    processing_status: Mapped[str] = mapped_column(
        String(50),
        default=ProcessingStatus.PENDING,
        nullable=False,
        index=True
    )
    processing_progress: Mapped[float] = mapped_column(Float, default=0.0)
    processing_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    processing_started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    processing_completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # Workout metadata
    workout_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )
    lift_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True
    )
    detected_lift_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # Auto-detected
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # REP COUNTS - CRITICAL SEPARATION
    # These MUST maintain the invariant: total_attempts = valid_reps + no_reps + ambiguous_reps
    total_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    valid_reps: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    no_reps: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    ambiguous_reps: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Analytics summary (stored as JSON string)
    _analytics_summary: Mapped[Optional[str]] = mapped_column("analytics_summary", Text, nullable=True)
    
    # Video thumbnail
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    
    # User workout log/notes
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    perceived_effort: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # RPE 1-10
    
    @property
    def analytics_summary(self) -> Optional[dict]:
        if self._analytics_summary:
            return json.loads(self._analytics_summary)
        return None
    
    @analytics_summary.setter
    def analytics_summary(self, value: Optional[dict]):
        if value is not None:
            self._analytics_summary = json.dumps(value)
        else:
            self._analytics_summary = None
    
    # Apple Health sync
    exported_to_health: Mapped[bool] = mapped_column(
        default=False,
        nullable=False
    )
    health_export_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="workouts")
    rep_attempts: Mapped[List["RepAttempt"]] = relationship(
        "RepAttempt",
        back_populates="workout",
        cascade="all, delete-orphan",
        order_by="RepAttempt.timestamp_start"
    )
    
    def validate_rep_counts(self) -> bool:
        """Validate that rep counts maintain invariant."""
        return self.total_attempts == (self.valid_reps + self.no_reps + self.ambiguous_reps)
    
    def recalculate_from_attempts(self) -> None:
        """Recalculate rep counts from individual rep attempts."""
        from app.models.rep_attempt import RepClassification
        
        self.total_attempts = len(self.rep_attempts)
        self.valid_reps = sum(
            1 for r in self.rep_attempts 
            if r.classification == RepClassification.VALID
        )
        self.no_reps = sum(
            1 for r in self.rep_attempts 
            if r.classification == RepClassification.NO_REP
        )
        self.ambiguous_reps = sum(
            1 for r in self.rep_attempts 
            if r.classification == RepClassification.AMBIGUOUS
        )
    
    def __repr__(self) -> str:
        return (
            f"<Workout(id={self.id}, lift={self.lift_type}, "
            f"total={self.total_attempts}, valid={self.valid_reps}, no_reps={self.no_reps})>"
        )

