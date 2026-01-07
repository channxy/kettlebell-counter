"""Apple HealthKit integration API endpoints."""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_active_user
from app.database import get_db
from app.models.user import User
from app.models.workout import Workout

router = APIRouter()


class HealthKitConsentRequest(BaseModel):
    """Request to enable/disable HealthKit sync."""
    enabled: bool


class HealthKitExportRequest(BaseModel):
    """Request to export a workout to HealthKit."""
    workout_id: str


class HealthKitExportPayload(BaseModel):
    """
    Payload format for HealthKit export.
    
    This is the data structure that would be sent to
    the iOS app for HealthKit integration.
    
    CRITICAL: Separates total_attempts from valid_reps.
    - valid_reps goes to HKQuantityType.workoutDurationMinutes equivalent
    - total_attempts and no_reps stored as metadata
    """
    workout_id: str
    workout_date: datetime
    lift_type: str
    duration_seconds: float
    
    # The official rep count (valid reps only)
    valid_reps: int
    
    # Metadata (for reference, not as main metric)
    total_attempts: int
    no_reps: int
    ambiguous_reps: int
    
    # Additional metadata
    calories_estimate: Optional[float] = None
    
    def to_healthkit_workout(self) -> Dict[str, Any]:
        """
        Convert to HealthKit-compatible format.
        
        In a real implementation, this would map to:
        - HKWorkout with activityType = .functionalStrengthTraining
        - HKQuantityType.workoutDurationMinutes
        - Custom metadata for kettlebell-specific data
        """
        return {
            "activityType": "functionalStrengthTraining",
            "startDate": self.workout_date.isoformat(),
            "duration": self.duration_seconds,
            "totalEnergyBurned": self.calories_estimate,
            "metadata": {
                "lift_type": self.lift_type,
                "valid_reps": self.valid_reps,
                "total_attempts": self.total_attempts,
                "no_reps": self.no_reps,
                "ambiguous_reps": self.ambiguous_reps,
                "source": "kettlebell_counter"
            }
        }


@router.post("/consent")
async def update_healthkit_consent(
    request: HealthKitConsentRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Update HealthKit consent status.
    
    IMPORTANT: This requires explicit user consent before syncing any data.
    """
    current_user.healthkit_enabled = request.enabled
    
    if request.enabled:
        current_user.healthkit_consent_date = datetime.utcnow()
    
    await db.commit()
    
    return {
        "healthkit_enabled": current_user.healthkit_enabled,
        "consent_date": current_user.healthkit_consent_date.isoformat() if current_user.healthkit_consent_date else None,
        "message": "HealthKit sync enabled" if request.enabled else "HealthKit sync disabled"
    }


@router.get("/consent")
async def get_healthkit_consent(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Get current HealthKit consent status."""
    return {
        "healthkit_enabled": current_user.healthkit_enabled,
        "consent_date": current_user.healthkit_consent_date.isoformat() if current_user.healthkit_consent_date else None
    }


@router.post("/export")
async def export_to_healthkit(
    request: HealthKitExportRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Export a workout to Apple Health.
    
    REQUIREMENTS:
    - User must have enabled HealthKit sync (explicit consent)
    - Workout must belong to user
    - Workout must be fully processed
    
    The response includes the HealthKit-formatted payload that
    would be sent to the iOS app for actual HealthKit write.
    """
    # Check consent
    if not current_user.healthkit_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="HealthKit sync not enabled. Please enable in settings first."
        )
    
    # Get workout
    result = await db.execute(
        select(Workout).where(
            Workout.id == request.workout_id,
            Workout.user_id == current_user.id
        )
    )
    workout = result.scalar_one_or_none()
    
    if not workout:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workout not found"
        )
    
    if workout.processing_status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workout processing not complete"
        )
    
    if workout.exported_to_health:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workout already exported to Apple Health"
        )
    
    # Estimate calories (rough estimate based on rep count and duration)
    # Competition kettlebell averages ~15-20 calories per minute
    duration_minutes = (workout.duration_seconds or 0) / 60
    calories_estimate = duration_minutes * 17.5  # Mid-range estimate
    
    # Create export payload
    payload = HealthKitExportPayload(
        workout_id=str(workout.id),
        workout_date=workout.workout_date,
        lift_type=workout.lift_type,
        duration_seconds=workout.duration_seconds or 0,
        valid_reps=workout.valid_reps,
        total_attempts=workout.total_attempts,
        no_reps=workout.no_reps,
        ambiguous_reps=workout.ambiguous_reps,
        calories_estimate=calories_estimate
    )
    
    # Mark as exported
    workout.exported_to_health = True
    workout.health_export_date = datetime.utcnow()
    await db.commit()
    
    return {
        "success": True,
        "message": "Workout exported to Apple Health",
        "export_date": workout.health_export_date.isoformat(),
        "healthkit_payload": payload.to_healthkit_workout(),
        "note": "In production, this payload would be sent to the iOS app for HealthKit write"
    }


@router.get("/export-history")
async def get_export_history(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get history of workouts exported to HealthKit."""
    result = await db.execute(
        select(Workout).where(
            Workout.user_id == current_user.id,
            Workout.exported_to_health == True
        ).order_by(Workout.health_export_date.desc())
    )
    workouts = result.scalars().all()
    
    exports = []
    for w in workouts:
        exports.append({
            "workout_id": str(w.id),
            "workout_date": w.workout_date.isoformat(),
            "export_date": w.health_export_date.isoformat() if w.health_export_date else None,
            "lift_type": w.lift_type,
            "valid_reps": w.valid_reps,
            "total_attempts": w.total_attempts
        })
    
    return {
        "total_exports": len(exports),
        "exports": exports
    }

