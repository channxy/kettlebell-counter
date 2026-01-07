"""Rep attempt model with classification and failure reasons."""

import uuid
import json
from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, Integer, Float, ForeignKey, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class RepClassification:
    """Rep attempt classification."""
    VALID = "valid"
    NO_REP = "no_rep"
    AMBIGUOUS = "ambiguous"
    
    @classmethod
    def all(cls) -> List[str]:
        return [cls.VALID, cls.NO_REP, cls.AMBIGUOUS]


class FailureReason:
    """
    Explicit failure reasons for no-reps.
    Each reason maps to a specific biomechanical check.
    """
    # Lockout issues
    INCOMPLETE_LOCKOUT = "incomplete_lockout"
    ELBOWS_NOT_EXTENDED = "elbows_not_extended"
    KETTLEBELL_NOT_OVERHEAD = "kettlebell_not_overhead"
    
    # Symmetry issues
    ASYMMETRIC_ARMS = "asymmetric_arms"
    ASYMMETRIC_LOCKOUT = "asymmetric_lockout"
    
    # Timing issues
    INSUFFICIENT_FIXATION = "insufficient_fixation"
    PREMATURE_DROP = "premature_drop"
    
    # Position issues
    FAILED_RACK_POSITION = "failed_rack_position"
    EXCESSIVE_TORSO_LEAN = "excessive_torso_lean"
    INCOMPLETE_EXTENSION = "incomplete_extension"
    
    # Technical issues
    IMPROPER_CATCH = "improper_catch"
    DOUBLE_BOUNCE = "double_bounce"
    SEGMENTED_PULL = "segmented_pull"  # Velocity dropped before reaching overhead
    JERKY_MOVEMENT = "jerky_movement"  # Non-smooth trajectory
    
    # Detection issues (for ambiguous)
    LOW_POSE_CONFIDENCE = "low_pose_confidence"
    OCCLUDED_KEYPOINTS = "occluded_keypoints"
    UNCLEAR_MOVEMENT = "unclear_movement"
    
    @classmethod
    def all(cls) -> List[str]:
        return [
            cls.INCOMPLETE_LOCKOUT,
            cls.ELBOWS_NOT_EXTENDED,
            cls.KETTLEBELL_NOT_OVERHEAD,
            cls.ASYMMETRIC_ARMS,
            cls.ASYMMETRIC_LOCKOUT,
            cls.INSUFFICIENT_FIXATION,
            cls.PREMATURE_DROP,
            cls.FAILED_RACK_POSITION,
            cls.EXCESSIVE_TORSO_LEAN,
            cls.INCOMPLETE_EXTENSION,
            cls.IMPROPER_CATCH,
            cls.DOUBLE_BOUNCE,
            cls.SEGMENTED_PULL,
            cls.JERKY_MOVEMENT,
            cls.LOW_POSE_CONFIDENCE,
            cls.OCCLUDED_KEYPOINTS,
            cls.UNCLEAR_MOVEMENT,
        ]
    
    @classmethod
    def get_description(cls, reason: str) -> str:
        """Get human-readable description of failure reason."""
        descriptions = {
            cls.INCOMPLETE_LOCKOUT: "Arms did not achieve full lockout position",
            cls.ELBOWS_NOT_EXTENDED: "Elbows were not fully extended at the top",
            cls.KETTLEBELL_NOT_OVERHEAD: "Kettlebell did not reach overhead position",
            cls.ASYMMETRIC_ARMS: "Arms were not symmetric during the lift",
            cls.ASYMMETRIC_LOCKOUT: "Lockout position was not symmetric",
            cls.INSUFFICIENT_FIXATION: "Did not hold lockout position long enough",
            cls.PREMATURE_DROP: "Kettlebell was dropped before fixation",
            cls.FAILED_RACK_POSITION: "Did not return to proper rack position",
            cls.EXCESSIVE_TORSO_LEAN: "Torso leaned excessively during lockout",
            cls.INCOMPLETE_EXTENSION: "Full extension was not achieved",
            cls.IMPROPER_CATCH: "Catch phase was not executed properly",
            cls.DOUBLE_BOUNCE: "Multiple bounces detected in rack position",
            cls.SEGMENTED_PULL: "Velocity stopped before reaching overhead (segmented pull)",
            cls.JERKY_MOVEMENT: "Movement trajectory was not smooth",
            cls.LOW_POSE_CONFIDENCE: "Pose detection confidence was too low",
            cls.OCCLUDED_KEYPOINTS: "Key body points were not visible",
            cls.UNCLEAR_MOVEMENT: "Movement pattern could not be clearly identified",
        }
        return descriptions.get(reason, reason)


class RepAttempt(Base, TimestampMixin):
    """
    Individual rep attempt with classification and analysis data.
    
    Every detected movement cycle creates a RepAttempt.
    Classification determines if it counts as a valid rep.
    """
    
    __tablename__ = "rep_attempts"
    
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )
    workout_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workouts.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Temporal boundaries (in video time)
    timestamp_start: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp_end: Mapped[float] = mapped_column(Float, nullable=False)
    frame_start: Mapped[int] = mapped_column(Integer, nullable=False)
    frame_end: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Classification
    classification: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True
    )
    
    # Failure reasons (stored as JSON string for SQLite compatibility)
    _failure_reasons: Mapped[Optional[str]] = mapped_column("failure_reasons", Text, nullable=True)
    
    @property
    def failure_reasons(self) -> Optional[List[str]]:
        if self._failure_reasons:
            return json.loads(self._failure_reasons)
        return None
    
    @failure_reasons.setter
    def failure_reasons(self, value: Optional[List[str]]):
        if value is not None:
            self._failure_reasons = json.dumps(value)
        else:
            self._failure_reasons = None
    
    # Confidence scores
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    pose_confidence_avg: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Biomechanical metrics (stored as JSON string)
    _metrics: Mapped[Optional[str]] = mapped_column("metrics", Text, nullable=True)
    
    @property
    def metrics(self) -> Optional[dict]:
        if self._metrics:
            return json.loads(self._metrics)
        return None
    
    @metrics.setter
    def metrics(self, value: Optional[dict]):
        if value is not None:
            self._metrics = json.dumps(value)
        else:
            self._metrics = None
    # Expected metrics structure:
    # {
    #     "lockout_angle_left": float,
    #     "lockout_angle_right": float,
    #     "fixation_duration_ms": float,
    #     "torso_lean_angle": float,
    #     "overhead_height_ratio": float,
    #     "symmetry_score": float,
    #     "tempo_ms": float,
    # }
    
    # Relationship
    workout: Mapped["Workout"] = relationship("Workout", back_populates="rep_attempts")
    
    @property
    def duration_seconds(self) -> float:
        """Duration of the rep attempt in seconds."""
        return self.timestamp_end - self.timestamp_start
    
    @property
    def is_valid(self) -> bool:
        """Check if this is a valid rep."""
        return self.classification == RepClassification.VALID
    
    @property
    def is_no_rep(self) -> bool:
        """Check if this is a no-rep."""
        return self.classification == RepClassification.NO_REP
    
    @property
    def is_ambiguous(self) -> bool:
        """Check if this is ambiguous."""
        return self.classification == RepClassification.AMBIGUOUS
    
    def get_failure_descriptions(self) -> List[str]:
        """Get human-readable failure descriptions."""
        if not self.failure_reasons:
            return []
        return [
            FailureReason.get_description(reason) 
            for reason in self.failure_reasons
        ]
    
    def __repr__(self) -> str:
        return (
            f"<RepAttempt(id={self.id}, classification={self.classification}, "
            f"time={self.timestamp_start:.2f}s-{self.timestamp_end:.2f}s)>"
        )

