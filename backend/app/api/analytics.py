"""Analytics API endpoints."""

import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import get_current_active_user
from app.database import get_db
from app.models.user import User
from app.models.workout import Workout
from app.models.rep_attempt import RepAttempt, RepClassification

router = APIRouter()


@router.get("/trends")
async def get_analytics_trends(
    days: int = Query(30, ge=7, le=365),
    lift_type: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get analytics trends over time.
    
    Includes:
    - Valid rep rate trends
    - Lockout consistency
    - No-rep frequency by reason
    - Fatigue indicators
    """
    since_date = datetime.now() - timedelta(days=days)
    
    # Build query
    query = (
        select(Workout)
        .options(selectinload(Workout.rep_attempts))
        .where(
            Workout.user_id == current_user.id,
            Workout.workout_date >= since_date,
            Workout.processing_status == "completed"
        )
        .order_by(Workout.workout_date)
    )
    
    if lift_type:
        query = query.where(Workout.lift_type == lift_type)
    
    result = await db.execute(query)
    workouts = result.scalars().all()
    
    if not workouts:
        return {
            "period_days": days,
            "total_workouts": 0,
            "message": "No completed workouts in this period"
        }
    
    # Aggregate metrics
    total_attempts = sum(w.total_attempts for w in workouts)
    total_valid = sum(w.valid_reps for w in workouts)
    total_no_reps = sum(w.no_reps for w in workouts)
    
    # Valid rep rate over time
    valid_rate_trend = []
    for w in workouts:
        if w.total_attempts > 0:
            valid_rate_trend.append({
                "date": w.workout_date.isoformat(),
                "workout_id": str(w.id),
                "valid_rate": w.valid_reps / w.total_attempts,
                "total_attempts": w.total_attempts
            })
    
    # No-rep reasons aggregate
    no_rep_reasons: Dict[str, int] = {}
    for w in workouts:
        for rep in w.rep_attempts:
            if rep.classification == RepClassification.NO_REP:
                for reason in (rep.failure_reasons or []):
                    no_rep_reasons[reason] = no_rep_reasons.get(reason, 0) + 1
    
    # Calculate averages from analytics summaries
    tempo_values = []
    lockout_degradation_values = []
    
    for w in workouts:
        if w.analytics_summary:
            tempo = w.analytics_summary.get("tempo", {})
            if "avg_ms" in tempo:
                tempo_values.append(tempo["avg_ms"])
            
            degradation = w.analytics_summary.get("lockout_degradation", {})
            if "degradation_degrees" in degradation:
                lockout_degradation_values.append(degradation["degradation_degrees"])
    
    return {
        "period_days": days,
        "total_workouts": len(workouts),
        "summary": {
            "total_attempts": total_attempts,
            "total_valid": total_valid,
            "total_no_reps": total_no_reps,
            "overall_valid_rate": total_valid / total_attempts if total_attempts > 0 else 0
        },
        "valid_rate_trend": valid_rate_trend,
        "no_rep_reasons": no_rep_reasons,
        "averages": {
            "avg_tempo_ms": sum(tempo_values) / len(tempo_values) if tempo_values else None,
            "avg_lockout_degradation": sum(lockout_degradation_values) / len(lockout_degradation_values) if lockout_degradation_values else None
        }
    }


@router.get("/workout/{workout_id}")
async def get_workout_analytics(
    workout_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get detailed analytics for a specific workout.
    
    Includes all form metrics, fatigue analysis, and explainable insights.
    """
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
        raise HTTPException(status_code=404, detail="Workout not found")
    
    if workout.processing_status != "completed":
        return {
            "workout_id": str(workout_id),
            "status": workout.processing_status,
            "message": "Analytics available after processing completes"
        }
    
    # Extract per-rep metrics
    lockout_angles = []
    tempos = []
    symmetry_scores = []
    
    for rep in workout.rep_attempts:
        if rep.metrics:
            # Lockout angles
            lockout = rep.metrics.get("lockout_angle", {})
            if "min_angle" in lockout:
                lockout_angles.append({
                    "timestamp": rep.timestamp_start,
                    "angle": lockout["min_angle"],
                    "classification": rep.classification
                })
            
            # Tempo
            if "tempo_ms" in rep.metrics:
                tempos.append({
                    "timestamp": rep.timestamp_start,
                    "tempo_ms": rep.metrics["tempo_ms"]
                })
            
            # Symmetry
            symmetry = rep.metrics.get("symmetry", {})
            if "min_symmetry_score" in symmetry:
                symmetry_scores.append({
                    "timestamp": rep.timestamp_start,
                    "score": symmetry["min_symmetry_score"]
                })
    
    # Fatigue analysis (compare first vs last quarter)
    reps = workout.rep_attempts
    if len(reps) >= 8:
        quarter = len(reps) // 4
        first_quarter = reps[:quarter]
        last_quarter = reps[-quarter:]
        
        first_valid_rate = sum(1 for r in first_quarter if r.is_valid) / len(first_quarter)
        last_valid_rate = sum(1 for r in last_quarter if r.is_valid) / len(last_quarter)
        
        fatigue_analysis = {
            "first_quarter_valid_rate": first_valid_rate,
            "last_quarter_valid_rate": last_valid_rate,
            "degradation": first_valid_rate - last_valid_rate,
            "fatigue_detected": (first_valid_rate - last_valid_rate) > 0.1
        }
    else:
        fatigue_analysis = {"message": "Need at least 8 reps for fatigue analysis"}
    
    return {
        "workout_id": str(workout_id),
        "summary": workout.analytics_summary,
        "lockout_angles": lockout_angles,
        "tempos": tempos,
        "symmetry_scores": symmetry_scores,
        "fatigue_analysis": fatigue_analysis,
        "rep_count": {
            "total_attempts": workout.total_attempts,
            "valid_reps": workout.valid_reps,
            "no_reps": workout.no_reps,
            "ambiguous_reps": workout.ambiguous_reps
        }
    }


@router.get("/compare")
async def compare_workouts(
    workout_ids: List[uuid.UUID] = Query(..., min_length=2, max_length=5),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Compare analytics between multiple workouts."""
    result = await db.execute(
        select(Workout)
        .where(
            Workout.id.in_(workout_ids),
            Workout.user_id == current_user.id
        )
        .order_by(Workout.workout_date)
    )
    workouts = result.scalars().all()
    
    if len(workouts) != len(workout_ids):
        raise HTTPException(status_code=404, detail="One or more workouts not found")
    
    comparisons = []
    for w in workouts:
        comparisons.append({
            "workout_id": str(w.id),
            "date": w.workout_date.isoformat(),
            "lift_type": w.lift_type,
            "total_attempts": w.total_attempts,
            "valid_reps": w.valid_reps,
            "no_reps": w.no_reps,
            "valid_rate": w.valid_reps / w.total_attempts if w.total_attempts > 0 else 0,
            "analytics_summary": w.analytics_summary
        })
    
    return {
        "workouts": comparisons,
        "improvement": {
            "valid_rate_change": comparisons[-1]["valid_rate"] - comparisons[0]["valid_rate"]
            if comparisons else 0
        }
    }

