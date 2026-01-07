"""Workout API endpoints."""

import uuid
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_active_user
from app.database import get_db
from app.models.user import User
from app.models.workout import Workout
from app.models.rep_attempt import RepAttempt, RepClassification
from app.schemas.workout import (
    WorkoutResponse,
    WorkoutListResponse,
    WorkoutDetailResponse,
    TimelinePoint,
)
from app.schemas.rep_attempt import RepAttemptResponse

router = APIRouter()


def _create_timeline(rep_attempts: List[RepAttempt]) -> List[TimelinePoint]:
    """Create timeline overlay data from rep attempts."""
    timeline = []
    
    for i, rep in enumerate(rep_attempts, 1):
        if rep.classification == RepClassification.VALID:
            color = "green"
        elif rep.classification == RepClassification.NO_REP:
            color = "red"
        else:
            color = "yellow"
        
        timeline.append(TimelinePoint(
            timestamp_start=rep.timestamp_start,
            timestamp_end=rep.timestamp_end,
            classification=rep.classification,
            color=color,
            rep_number=i,
            failure_reasons=rep.failure_reasons
        ))
    
    return timeline


@router.get("", response_model=WorkoutListResponse)
async def list_workouts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    lift_type: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """List user's workouts with pagination."""
    # Base query
    query = select(Workout).where(Workout.user_id == current_user.id)
    count_query = select(func.count(Workout.id)).where(Workout.user_id == current_user.id)
    
    # Filter by lift type
    if lift_type:
        query = query.where(Workout.lift_type == lift_type)
        count_query = count_query.where(Workout.lift_type == lift_type)
    
    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Paginate
    offset = (page - 1) * page_size
    query = query.order_by(desc(Workout.workout_date)).offset(offset).limit(page_size)
    
    result = await db.execute(query)
    workouts = result.scalars().all()
    
    return WorkoutListResponse(
        items=[WorkoutResponse.model_validate(w) for w in workouts],
        total=total,
        page=page,
        page_size=page_size,
        has_more=(offset + len(workouts)) < total
    )


@router.get("/{workout_id}", response_model=WorkoutDetailResponse)
async def get_workout(
    workout_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed workout information including all rep attempts."""
    result = await db.execute(
        select(Workout)
        .options(selectinload(Workout.rep_attempts))
        .where(
            Workout.id == workout_id,
            Workout.user_id == current_user.id
        )
    )
    workout = result.scalar_one_or_none()
    
    if not workout:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workout not found"
        )
    
    # Create rep attempt responses with rep numbers
    rep_responses = []
    for i, rep in enumerate(workout.rep_attempts, 1):
        rep_data = RepAttemptResponse(
            id=rep.id,
            rep_number=i,
            timestamp_start=rep.timestamp_start,
            timestamp_end=rep.timestamp_end,
            classification=rep.classification,
            failure_reasons=rep.failure_reasons,
            confidence_score=rep.confidence_score,
            metrics=rep.metrics
        )
        rep_responses.append(rep_data)
    
    # Create timeline
    timeline = _create_timeline(workout.rep_attempts)
    
    return WorkoutDetailResponse(
        id=workout.id,
        user_id=workout.user_id,
        lift_type=workout.lift_type,
        workout_date=workout.workout_date,
        duration_seconds=workout.duration_seconds,
        total_attempts=workout.total_attempts,
        valid_reps=workout.valid_reps,
        no_reps=workout.no_reps,
        ambiguous_reps=workout.ambiguous_reps,
        processing_status=workout.processing_status,
        processing_progress=workout.processing_progress,
        processing_started_at=workout.processing_started_at,
        processing_error=workout.processing_error,
        video_duration_seconds=workout.video_duration_seconds,
        video_filename=workout.video_filename,
        thumbnail_path=workout.thumbnail_path,
        notes=workout.notes,
        perceived_effort=workout.perceived_effort,
        mood=workout.mood,
        analytics_summary=workout.analytics_summary,
        rep_attempts=rep_responses,
        timeline=timeline,
        exported_to_health=workout.exported_to_health,
        health_export_date=workout.health_export_date,
        created_at=workout.created_at,
        updated_at=workout.updated_at
    )


@router.get("/{workout_id}/timeline")
async def get_workout_timeline(
    workout_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> List[TimelinePoint]:
    """Get timeline overlay data for video visualization."""
    result = await db.execute(
        select(Workout)
        .options(selectinload(Workout.rep_attempts))
        .where(
            Workout.id == workout_id,
            Workout.user_id == current_user.id
        )
    )
    workout = result.scalar_one_or_none()
    
    if not workout:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workout not found"
        )
    
    return _create_timeline(workout.rep_attempts)


@router.get("/{workout_id}/no-reps")
async def get_workout_no_reps(
    workout_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed breakdown of all no-reps in a workout."""
    result = await db.execute(
        select(Workout)
        .options(selectinload(Workout.rep_attempts))
        .where(
            Workout.id == workout_id,
            Workout.user_id == current_user.id
        )
    )
    workout = result.scalar_one_or_none()
    
    if not workout:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workout not found"
        )
    
    # Filter to no-reps
    no_reps = [
        r for r in workout.rep_attempts 
        if r.classification == RepClassification.NO_REP
    ]
    
    # Count reasons
    reasons_breakdown = {}
    for rep in no_reps:
        for reason in (rep.failure_reasons or []):
            reasons_breakdown[reason] = reasons_breakdown.get(reason, 0) + 1
    
    # Create responses
    no_rep_responses = []
    for i, rep in enumerate(no_reps, 1):
        # Find actual rep number in full list
        full_index = workout.rep_attempts.index(rep) + 1
        
        no_rep_responses.append({
            "id": str(rep.id),
            "rep_number": full_index,
            "timestamp_start": rep.timestamp_start,
            "timestamp_end": rep.timestamp_end,
            "failure_reasons": rep.failure_reasons,
            "failure_descriptions": rep.get_failure_descriptions(),
            "confidence_score": rep.confidence_score,
            "metrics": rep.metrics
        })
    
    return {
        "total_no_reps": len(no_reps),
        "reasons_breakdown": reasons_breakdown,
        "no_reps": no_rep_responses
    }


@router.patch("/{workout_id}")
async def update_workout(
    workout_id: str,
    notes: str = Query(None),
    perceived_effort: int = Query(None),
    mood: str = Query(None),
    workout_date: str = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update workout notes, perceived effort (RPE), mood, and date."""
    from datetime import datetime as dt
    
    result = await db.execute(
        select(Workout).where(
            Workout.id == workout_id,
            Workout.user_id == current_user.id
        )
    )
    workout = result.scalar_one_or_none()
    
    if not workout:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workout not found"
        )
    
    if notes is not None:
        workout.notes = notes
    if perceived_effort is not None:
        if perceived_effort < 1 or perceived_effort > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Perceived effort must be between 1 and 10"
            )
        workout.perceived_effort = perceived_effort
    if mood is not None:
        valid_moods = ["great", "good", "okay", "tired", "heavy"]
        if mood and mood not in valid_moods:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Mood must be one of: {valid_moods}"
            )
        workout.mood = mood
    if workout_date is not None:
        try:
            parsed_date = dt.fromisoformat(workout_date.replace('Z', '+00:00'))
            workout.workout_date = parsed_date
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format"
            )
    
    await db.commit()
    await db.refresh(workout)
    
    return {
        "id": workout.id,
        "notes": workout.notes,
        "perceived_effort": workout.perceived_effort,
        "mood": workout.mood,
        "workout_date": workout.workout_date.isoformat() if workout.workout_date else None,
        "message": "Workout updated successfully"
    }


@router.delete("/{workout_id}")
async def delete_workout(
    workout_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a workout and its associated data."""
    import os
    
    result = await db.execute(
        select(Workout).where(
            Workout.id == workout_id,
            Workout.user_id == current_user.id
        )
    )
    workout = result.scalar_one_or_none()
    
    if not workout:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workout not found"
        )
    
    # Delete video file if exists
    if workout.video_path and os.path.exists(workout.video_path):
        try:
            os.remove(workout.video_path)
        except OSError:
            pass  # Ignore file deletion errors
    
    # Delete thumbnail if exists
    if workout.thumbnail_path and os.path.exists(workout.thumbnail_path):
        try:
            os.remove(workout.thumbnail_path)
        except OSError:
            pass
    
    # Delete workout (cascade will delete rep_attempts)
    await db.delete(workout)
    await db.commit()
    
    return {"message": "Workout deleted successfully", "id": workout_id}

