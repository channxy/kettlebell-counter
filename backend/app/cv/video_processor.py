"""
Video processing pipeline for kettlebell rep analysis.

This module handles:
1. Video decoding and frame extraction
2. Pose estimation on extracted frames
3. Rep detection and validation
4. Analytics computation
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Generator, Tuple
from datetime import datetime

import cv2
import numpy as np

from app.cv.hybrid_estimator import HybridEstimator, HybridPose as PoseKeypoints
from app.cv.rep_detector import RepDetector, RepDetectorFactory, RepCycle
from app.cv.rep_validator import RepValidator, ValidationResult
from app.cv.lift_classifier import LiftClassifier, auto_detect_lift_type
from app.models.rep_attempt import RepClassification, FailureReason
from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """
    Complete result of video processing.
    
    Contains all detected reps, validations, and analytics.
    """
    # Counts - CRITICAL SEPARATION
    total_attempts: int = 0
    valid_reps: int = 0
    no_reps: int = 0
    ambiguous_reps: int = 0
    
    # Detailed results
    rep_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Video metadata
    video_duration_seconds: float = 0.0
    video_fps: float = 0.0
    frames_processed: int = 0
    
    # Auto-detection results
    detected_lift_type: Optional[str] = None
    lift_detection_confidence: float = 0.0
    lift_detection_reasoning: str = ""
    
    # Analytics summary
    analytics: Dict[str, Any] = field(default_factory=dict)
    
    # Processing metadata
    processing_time_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def validate_counts(self) -> bool:
        """Verify count invariant."""
        return self.total_attempts == (self.valid_reps + self.no_reps + self.ambiguous_reps)


class VideoProcessor:
    """
    Main video processing pipeline.
    
    Orchestrates:
    1. Frame extraction from video
    2. Pose estimation on each frame
    3. Rep cycle detection
    4. Rep validation and classification
    5. Analytics computation
    """
    
    def __init__(
        self,
        lift_type: str,
        processing_fps: Optional[float] = None,
        pose_model_complexity: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ):
        """
        Initialize video processor.
        
        Args:
            lift_type: Type of lift to detect
            processing_fps: Target FPS for processing (None = use config default)
            pose_model_complexity: MediaPipe model complexity (0, 1, or 2)
            progress_callback: Optional callback for progress updates
        """
        self.settings = get_settings()
        self.lift_type = lift_type
        self.processing_fps = processing_fps or self.settings.processing_fps
        self.pose_model_complexity = pose_model_complexity if pose_model_complexity is not None else self.settings.pose_model_complexity
        self.progress_callback = progress_callback
        self.frame_width = self.settings.processing_frame_width
        
    def process_video(self, video_path: str) -> ProcessingResult:
        """
        Process a video file and detect/validate all reps.
        
        Args:
            video_path: Path to video file
            
        Returns:
            ProcessingResult with all detected reps and analytics
        """
        start_time = datetime.now()
        result = ProcessingResult()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            result.errors.append(f"Failed to open video: {video_path}")
            return result
        
        try:
            # Get video properties
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / video_fps if video_fps > 0 else 0
            
            result.video_fps = video_fps
            result.video_duration_seconds = duration
            
            logger.info(f"Processing video: {duration:.1f}s, {video_fps:.1f}fps, {total_frames} frames")
            
            # Calculate frame sampling
            frame_skip = max(1, int(video_fps / self.processing_fps))
            
            # Initialize components - Hybrid MoveNet + MediaPipe Hands
            pose_estimator = HybridEstimator()
            validator = RepValidator()
            lift_classifier = LiftClassifier()
            
            # Storage for poses and cycles
            all_poses: List[PoseKeypoints] = []
            cycles: List[RepCycle] = []
            
            # Determine lift type (auto-detect if needed)
            effective_lift_type = self.lift_type
            auto_detect = self.lift_type == "auto" or self.lift_type is None
            rep_detector = None  # Will be initialized after auto-detection
            
            if not auto_detect:
                rep_detector = RepDetectorFactory.create(self.lift_type, self.processing_fps)
            
            # Process frames
            frame_number = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames
                if frame_number % frame_skip == 0:
                    timestamp = frame_number / video_fps
                    
                    # Resize frame for faster processing
                    if self.frame_width > 0 and frame.shape[1] > self.frame_width:
                        scale = self.frame_width / frame.shape[1]
                        new_height = int(frame.shape[0] * scale)
                        frame = cv2.resize(frame, (self.frame_width, new_height), interpolation=cv2.INTER_LINEAR)
                    
                    # Estimate pose
                    pose = pose_estimator.process_frame(
                        frame,
                        frame_number=processed_frames,
                        timestamp=timestamp
                    )
                    all_poses.append(pose)
                    
                    # Auto-detect lift type if needed (using first ~5 seconds)
                    if auto_detect:
                        lift_classifier.add_pose(pose)
                        if lift_classifier.has_enough_data() and rep_detector is None:
                            classification = lift_classifier.classify()
                            effective_lift_type = classification.lift_type
                            if effective_lift_type == "unknown":
                                effective_lift_type = "snatch"  # Default to snatch (most common single KB)
                            
                            # Store detection results
                            result.detected_lift_type = effective_lift_type
                            result.lift_detection_confidence = classification.confidence
                            result.lift_detection_reasoning = classification.reasoning
                            
                            logger.info(f"Auto-detected lift type: {effective_lift_type} "
                                       f"(confidence: {classification.confidence:.2f})")
                            logger.info(f"Detection reasoning: {classification.reasoning}")
                            logger.info(f"Single arm: {classification.single_arm}, "
                                       f"Dominant hand: {classification.dominant_hand}")
                            
                            # Initialize rep detector with detected type
                            rep_detector = RepDetectorFactory.create(effective_lift_type, self.processing_fps)
                            
                            # Process all collected poses through the detector
                            for past_pose in all_poses:
                                cycle = rep_detector.process_pose(past_pose)
                                if cycle:
                                    cycles.append(cycle)
                    
                    # Detect rep cycles (only if detector initialized)
                    if rep_detector is not None:
                        cycle = rep_detector.process_pose(pose)
                        if cycle:
                            cycles.append(cycle)
                    
                    processed_frames += 1
                    
                    # Progress update - every 10 frames for smoother UI updates
                    if self.progress_callback and processed_frames % 10 == 0:
                        progress = frame_number / total_frames
                        self.progress_callback(progress)
                
                frame_number += 1
            
            # Ensure rep_detector was initialized (fallback if auto-detect didn't run)
            if rep_detector is None:
                logger.warning("Rep detector not initialized, using jerk as default")
                effective_lift_type = "jerk"
                rep_detector = RepDetectorFactory.create(effective_lift_type, self.processing_fps)
                for past_pose in all_poses:
                    cycle = rep_detector.process_pose(past_pose)
                    if cycle:
                        cycles.append(cycle)
            
            # Finalize detection
            cycles = rep_detector.finalize()
            
            result.frames_processed = processed_frames
            result.total_attempts = len(cycles)
            
            logger.info(f"Detected {len(cycles)} rep attempts")
            
            # Validate each cycle
            for i, cycle in enumerate(cycles):
                validation = validator.validate(cycle, all_poses)
                
                # Update counts
                if validation.classification == RepClassification.VALID:
                    result.valid_reps += 1
                elif validation.classification == RepClassification.NO_REP:
                    result.no_reps += 1
                else:
                    result.ambiguous_reps += 1
                
                # Store detailed result
                rep_result = {
                    "rep_number": i + 1,
                    "timestamp_start": cycle.start_timestamp,
                    "timestamp_end": cycle.end_timestamp,
                    "frame_start": cycle.start_frame,
                    "frame_end": cycle.end_frame,
                    "classification": validation.classification,
                    "failure_reasons": validation.failure_reasons,
                    "confidence_score": validation.confidence_score,
                    "pose_confidence_avg": validation.pose_confidence_avg,
                    "metrics": validation.metrics
                }
                result.rep_results.append(rep_result)
            
            # Compute analytics
            result.analytics = self._compute_analytics(result.rep_results, all_poses)
            
            # Cleanup
            pose_estimator.close()
            
        finally:
            cap.release()
        
        # Processing time
        result.processing_time_seconds = (datetime.now() - start_time).total_seconds()
        
        # Validate counts
        if not result.validate_counts():
            logger.error("Count invariant violation!")
            result.errors.append("Count invariant violation")
        
        logger.info(
            f"Processing complete: {result.total_attempts} attempts, "
            f"{result.valid_reps} valid, {result.no_reps} no-reps, "
            f"{result.ambiguous_reps} ambiguous"
        )
        
        return result
    
    def _compute_analytics(
        self,
        rep_results: List[Dict[str, Any]],
        poses: List[PoseKeypoints]
    ) -> Dict[str, Any]:
        """
        Compute form and technique analytics.
        
        Analyzes:
        - Lockout consistency
        - Tempo trends
        - Fatigue indicators
        - Symmetry over time
        """
        if not rep_results:
            return {}
        
        analytics = {}
        
        # Valid rep rate
        valid_count = sum(1 for r in rep_results if r["classification"] == RepClassification.VALID)
        analytics["valid_rep_rate"] = valid_count / len(rep_results)
        
        # Lockout angle trends
        lockout_angles = []
        for r in rep_results:
            metrics = r.get("metrics", {})
            lockout_data = metrics.get("lockout_angle", {})
            if "min_angle" in lockout_data:
                lockout_angles.append({
                    "timestamp": r["timestamp_start"],
                    "angle": lockout_data["min_angle"]
                })
        
        if lockout_angles:
            # Calculate degradation over time
            first_half = lockout_angles[:len(lockout_angles)//2]
            second_half = lockout_angles[len(lockout_angles)//2:]
            
            first_avg = np.mean([a["angle"] for a in first_half]) if first_half else 0
            second_avg = np.mean([a["angle"] for a in second_half]) if second_half else 0
            
            analytics["lockout_degradation"] = {
                "first_half_avg": first_avg,
                "second_half_avg": second_avg,
                "degradation_degrees": first_avg - second_avg
            }
        
        # Tempo analysis
        tempos = [r["timestamp_end"] - r["timestamp_start"] for r in rep_results]
        if tempos:
            analytics["tempo"] = {
                "avg_ms": np.mean(tempos) * 1000,
                "std_ms": np.std(tempos) * 1000,
                "min_ms": np.min(tempos) * 1000,
                "max_ms": np.max(tempos) * 1000
            }
            
            # Tempo consistency (coefficient of variation)
            if np.mean(tempos) > 0:
                analytics["tempo"]["consistency"] = 1 - (np.std(tempos) / np.mean(tempos))
        
        # No-rep frequency by reason
        no_rep_reasons = {}
        for r in rep_results:
            if r["classification"] == RepClassification.NO_REP:
                for reason in r.get("failure_reasons", []):
                    no_rep_reasons[reason] = no_rep_reasons.get(reason, 0) + 1
        
        analytics["no_rep_breakdown"] = no_rep_reasons
        
        # Fatigue indicators (later reps vs earlier)
        if len(rep_results) >= 10:
            first_10 = rep_results[:10]
            last_10 = rep_results[-10:]
            
            first_valid = sum(1 for r in first_10 if r["classification"] == RepClassification.VALID)
            last_valid = sum(1 for r in last_10 if r["classification"] == RepClassification.VALID)
            
            analytics["fatigue_indicator"] = {
                "first_10_valid_rate": first_valid / 10,
                "last_10_valid_rate": last_valid / 10,
                "degradation": (first_valid - last_valid) / 10
            }
        
        return analytics
    
    def extract_frames_generator(
        self,
        video_path: str
    ) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """
        Generator that yields frames from video.
        
        Yields:
            Tuple of (frame, frame_number, timestamp)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        
        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = max(1, int(video_fps / self.processing_fps))
            
            frame_number = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_number % frame_skip == 0:
                    timestamp = frame_number / video_fps
                    yield frame, processed_frames, timestamp
                    processed_frames += 1
                
                frame_number += 1
        finally:
            cap.release()

