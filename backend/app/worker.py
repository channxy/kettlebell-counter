"""Celery worker for async video processing."""

import os
import uuid
import logging
from datetime import datetime
from typing import Optional

from celery import Celery
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import SyncSessionLocal
from app.models.workout import Workout, ProcessingStatus
from app.models.rep_attempt import RepAttempt, RepClassification
from app.cv.video_processor import VideoProcessor

settings = get_settings()
logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "kettlebell_counter",
    broker=settings.redis_url,
    backend=settings.redis_url
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    worker_prefetch_multiplier=1,  # Process one task at a time
)


def update_progress(db: Session, workout_id: str, progress: float, status: str = None):
    """Update processing progress in database."""
    workout = db.query(Workout).filter(Workout.id == workout_id).first()
    if workout:
        workout.processing_progress = progress
        if status:
            workout.processing_status = status
        db.commit()


@celery_app.task(bind=True, name="process_video")
def process_video_task(self, workout_id: str):
    """
    Process a workout video asynchronously.
    
    Steps:
    1. Load video file
    2. Extract frames
    3. Run pose estimation
    4. Detect rep cycles
    5. Validate each rep
    6. Store results
    
    CRITICAL: Maintains separation between total_attempts, valid_reps, no_reps.
    """
    logger.info(f"Starting video processing for workout {workout_id}")
    
    db = SyncSessionLocal()
    
    try:
        # Get workout
        workout = db.query(Workout).filter(Workout.id == workout_id).first()
        if not workout:
            logger.error(f"Workout {workout_id} not found")
            return {"error": "Workout not found"}
        
        # Update status
        workout.processing_status = ProcessingStatus.PROCESSING
        workout.processing_started_at = datetime.utcnow()
        db.commit()
        
        # Check video file exists
        if not os.path.exists(workout.video_path):
            raise FileNotFoundError(f"Video file not found: {workout.video_path}")
        
        # Create progress callback
        def progress_callback(progress: float):
            update_progress(db, workout_id, progress * 0.8)  # Reserve 20% for post-processing
        
        # Initialize processor
        processor = VideoProcessor(
            lift_type=workout.lift_type,
            progress_callback=progress_callback
        )
        
        # Process video
        result = processor.process_video(workout.video_path)
        
        # Update progress
        update_progress(db, workout_id, 0.9, ProcessingStatus.ANALYZING)
        
        # Validate count invariant
        if not result.validate_counts():
            logger.error(f"Count invariant violation for workout {workout_id}")
            raise ValueError("Count invariant violation")
        
        # Store results
        workout.video_duration_seconds = result.video_duration_seconds
        workout.video_fps = result.video_fps
        workout.duration_seconds = result.video_duration_seconds
        
        # Store auto-detected lift type
        if result.detected_lift_type:
            workout.detected_lift_type = result.detected_lift_type
            # If original was "auto", update to detected type
            if workout.lift_type == "auto":
                workout.lift_type = result.detected_lift_type
                logger.info(f"Updated lift type from 'auto' to '{result.detected_lift_type}'")
        
        # CRITICAL: Store separated counts
        workout.total_attempts = result.total_attempts
        workout.valid_reps = result.valid_reps
        workout.no_reps = result.no_reps
        workout.ambiguous_reps = result.ambiguous_reps
        
        # Store analytics summary
        workout.analytics_summary = result.analytics
        
        # Create rep attempt records
        for rep_data in result.rep_results:
            rep_attempt = RepAttempt(
                workout_id=workout.id,
                timestamp_start=rep_data["timestamp_start"],
                timestamp_end=rep_data["timestamp_end"],
                frame_start=rep_data["frame_start"],
                frame_end=rep_data["frame_end"],
                classification=rep_data["classification"],
                failure_reasons=rep_data.get("failure_reasons"),
                confidence_score=rep_data["confidence_score"],
                pose_confidence_avg=rep_data["pose_confidence_avg"],
                metrics=rep_data.get("metrics")
            )
            db.add(rep_attempt)
        
        # Flush to persist rep attempts, then recalculate from database
        db.flush()
        
        # Verify counts match rep attempts
        # Refresh relationship to see newly added rep attempts
        db.refresh(workout)
        workout.recalculate_from_attempts()
        if not workout.validate_rep_counts():
            logger.error(f"Rep count mismatch for workout {workout_id}")
            raise ValueError("Rep count mismatch")
        
        # Mark complete
        workout.processing_status = ProcessingStatus.COMPLETED
        workout.processing_progress = 1.0
        workout.processing_completed_at = datetime.utcnow()
        db.commit()
        
        logger.info(
            f"Processing complete for workout {workout_id}: "
            f"{workout.total_attempts} attempts, {workout.valid_reps} valid, "
            f"{workout.no_reps} no-reps, {workout.ambiguous_reps} ambiguous"
        )
        
        return {
            "workout_id": workout_id,
            "total_attempts": workout.total_attempts,
            "valid_reps": workout.valid_reps,
            "no_reps": workout.no_reps,
            "ambiguous_reps": workout.ambiguous_reps,
            "processing_time_seconds": result.processing_time_seconds
        }
        
    except Exception as e:
        logger.exception(f"Error processing workout {workout_id}: {e}")
        
        # Update error status
        workout = db.query(Workout).filter(Workout.id == workout_id).first()
        if workout:
            workout.processing_status = ProcessingStatus.FAILED
            workout.processing_error = str(e)
            db.commit()
        
        raise
        
    finally:
        db.close()


@celery_app.task(name="cleanup_old_videos")
def cleanup_old_videos_task(days_old: int = 30):
    """
    Cleanup old processed videos to save storage.
    
    Only removes video files, keeps workout data.
    """
    from datetime import timedelta
    
    db = SyncSessionLocal()
    cutoff = datetime.utcnow() - timedelta(days=days_old)
    
    try:
        workouts = db.query(Workout).filter(
            Workout.processing_completed_at < cutoff,
            Workout.processing_status == ProcessingStatus.COMPLETED
        ).all()
        
        deleted_count = 0
        for workout in workouts:
            if os.path.exists(workout.video_path):
                os.remove(workout.video_path)
                workout.video_path = f"[deleted] {workout.video_path}"
                deleted_count += 1
        
        db.commit()
        logger.info(f"Cleaned up {deleted_count} old video files")
        
        return {"deleted_count": deleted_count}
        
    finally:
        db.close()

