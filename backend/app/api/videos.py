"""Video upload and processing API endpoints."""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_active_user
from app.config import get_settings
from app.database import get_db
from app.models.user import User
from app.models.workout import Workout, ProcessingStatus, LiftType
from app.schemas.workout import WorkoutCreate, WorkoutResponse

router = APIRouter()
settings = get_settings()


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MAX_FILE_SIZE = settings.max_video_size_mb * 1024 * 1024  # Convert to bytes


def validate_video_file(filename: str, file_size: int) -> None:
    """Validate video file extension and size."""
    ext = Path(filename).suffix.lower()
    
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"
        )
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum: {settings.max_video_size_mb}MB"
        )


@router.post("/upload", response_model=WorkoutResponse, status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    lift_type: str = Form(...),
    workout_date: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a workout video for processing.
    
    The video will be queued for offline processing. Use the status endpoint
    to check processing progress.
    """
    # Validate lift type (including "auto" for auto-detection)
    valid_lift_types = LiftType.all() + ["auto"]
    if lift_type not in valid_lift_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid lift_type. Must be one of: {valid_lift_types}"
        )
    
    # Get file size by reading first chunk
    contents = await file.read()
    file_size = len(contents)
    
    validate_video_file(file.filename, file_size)
    
    # Parse workout date
    parsed_date = datetime.now()
    if workout_date:
        try:
            parsed_date = datetime.fromisoformat(workout_date)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid workout_date format. Use ISO format."
            )
    
    # Create upload directory
    upload_dir = Path(settings.upload_dir) / str(current_user.id)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    file_ext = Path(file.filename).suffix.lower()
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = upload_dir / unique_filename
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Create workout record
    workout = Workout(
        user_id=current_user.id,
        video_filename=file.filename,
        video_path=str(file_path),
        workout_date=parsed_date,
        lift_type=lift_type,
        processing_status=ProcessingStatus.PENDING,
        processing_progress=0.0,
        total_attempts=0,
        valid_reps=0,
        no_reps=0,
        ambiguous_reps=0
    )
    
    db.add(workout)
    await db.commit()
    await db.refresh(workout)
    
    # Queue processing task
    from app.worker import process_video_task
    process_video_task.delay(str(workout.id))
    
    return workout


@router.get("/{video_id}/status", response_model=WorkoutResponse)
async def get_video_status(
    video_id: uuid.UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get processing status for a video."""
    from sqlalchemy import select
    
    result = await db.execute(
        select(Workout).where(
            Workout.id == video_id,
            Workout.user_id == current_user.id
        )
    )
    workout = result.scalar_one_or_none()
    
    if not workout:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    return workout


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(
    video_id: uuid.UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a video and its workout data."""
    from sqlalchemy import select
    
    result = await db.execute(
        select(Workout).where(
            Workout.id == video_id,
            Workout.user_id == current_user.id
        )
    )
    workout = result.scalar_one_or_none()
    
    if not workout:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found"
        )
    
    # Delete video file
    if os.path.exists(workout.video_path):
        os.remove(workout.video_path)
    
    # Delete workout record (cascade deletes rep_attempts)
    await db.delete(workout)
    await db.commit()

