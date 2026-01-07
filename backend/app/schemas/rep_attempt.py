"""Rep attempt schemas."""

import uuid
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from app.models.rep_attempt import FailureReason


class RepAttemptResponse(BaseModel):
    """Schema for individual rep attempt response."""
    id: str  # String ID for SQLite compatibility
    rep_number: Optional[int] = None
    workout_id: Optional[str] = None
    timestamp_start: float
    timestamp_end: float
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None
    classification: str
    failure_reasons: Optional[List[str]] = None
    confidence_score: float
    pose_confidence_avg: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True
    
    def get_failure_descriptions(self) -> List[str]:
        """Get human-readable failure descriptions."""
        if not self.failure_reasons:
            return []
        return [FailureReason.get_description(r) for r in self.failure_reasons]


class RepAttemptListResponse(BaseModel):
    """Schema for list of rep attempts."""
    items: List[RepAttemptResponse]
    total: int
    
    # Summary counts
    valid_count: int
    no_rep_count: int
    ambiguous_count: int


class NoRepSummary(BaseModel):
    """Summary of no-reps with reasons."""
    total_no_reps: int
    reasons_breakdown: Dict[str, int]
    no_reps: List[RepAttemptResponse]

