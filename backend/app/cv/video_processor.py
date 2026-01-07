"""
Video processing pipeline for kettlebell rep analysis.

PRODUCTION-GRADE PIPELINE:
1. Pre-analysis: Video metadata, auto-crop detection, camera angle
2. Frame extraction with optional cropping
3. Pose estimation (MoveNet Thunder + MediaPipe Hands)
4. Keypoint smoothing (temporal EMA)
5. Lift type classification
6. Window-based rep detection
7. Rep validation and classification
8. Fatigue-aware analytics

Prioritizes ACCURACY over speed. Never inflates counts.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Generator, Tuple
from datetime import datetime

import cv2
import numpy as np

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

# New modular components
from app.cv.video_analyzer import VideoAnalyzer, VideoMetadata, CropRegion, CameraAngle
# KeypointSmoother no longer used - HybridPose has built-in temporal smoothing
from app.cv.window_rep_detector import WindowRepDetector, RepWindow, AngleAdaptiveThresholds
from app.cv.fatigue_handler import FatigueHandler
from app.cv.lift_classifier import LiftClassifier, LiftClassification

# Pose estimator
from app.cv.hybrid_estimator import HybridEstimator, HybridPose

# Validation
from app.cv.rep_validator import RepValidator, ValidationResult
from app.models.rep_attempt import RepClassification, FailureReason
from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """
    Complete result of video processing.
    
    CRITICAL: Maintains count invariant:
    Total Attempts = Valid Reps + No-Reps + Ambiguous
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
    
    # Pre-analysis results
    camera_angle: str = "unknown"
    auto_cropped: bool = False
    confidence_downgrade: bool = False
    
    # Auto-detection results
    detected_lift_type: Optional[str] = None
    lift_detection_confidence: float = 0.0
    lift_detection_reasoning: str = ""
    dominant_hand: Optional[str] = None
    
    # Fatigue analysis
    fatigue_score: float = 0.0
    baselines_created: int = 0
    
    # Analytics summary
    analytics: Dict[str, Any] = field(default_factory=dict)
    
    # Processing metadata
    processing_time_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def validate_counts(self) -> bool:
        """Verify count invariant."""
        return self.total_attempts == (self.valid_reps + self.no_reps + self.ambiguous_reps)


class VideoProcessor:
    """
    Main video processing pipeline.
    
    PIPELINE STAGES:
    1. Pre-analysis (auto-crop, camera angle detection)
    2. Frame extraction (with cropping if needed)
    3. Pose estimation (MoveNet Thunder)
    4. Keypoint smoothing (per-joint EMA)
    5. Lift type classification (correlation-based)
    6. Rep window detection (state machine)
    7. Rep validation (rule-based)
    8. Analytics computation
    """
    
    def __init__(
        self,
        lift_type: str,
        processing_fps: Optional[float] = None,
        progress_callback: Optional[callable] = None
    ):
        """
        Initialize video processor.
        
        Args:
            lift_type: Type of lift ("snatch", "jerk", "long_cycle", or "auto")
            processing_fps: Target FPS for processing (default: 10)
            progress_callback: Optional callback for progress updates
        """
        self.settings = get_settings()
        self.lift_type = lift_type
        self.processing_fps = processing_fps or 10.0  # 10 FPS is good balance
        self.progress_callback = progress_callback
        
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
        
        logger.info(f"Starting video processing: {video_path}")
        
        # =========================================================
        # STAGE 1: Pre-analysis
        # =========================================================
        logger.info("Stage 1: Pre-analysis...")
        
        try:
            analyzer = VideoAnalyzer()
            analysis = analyzer.analyze(video_path)
        except Exception as e:
            logger.error(f"Pre-analysis failed: {e}")
            result.errors.append(f"Pre-analysis failed: {e}")
            return result
        
        # Check if video is processable
        if not analysis.can_process:
            logger.error(f"Video quality too low to process: {analysis.errors}")
            result.errors.extend(analysis.errors)
            result.warnings.extend(analysis.warnings)
            return result
        
        metadata = analysis.metadata
        result.video_fps = metadata.fps
        result.video_duration_seconds = metadata.duration_seconds
        result.camera_angle = metadata.camera_angle.value
        result.auto_cropped = metadata.needs_crop
        result.confidence_downgrade = metadata.confidence_downgrade
        result.warnings.extend(analysis.warnings)
        result.errors.extend(analysis.errors)
        
        # Log quality assessment
        logger.info(f"Video quality: {metadata.quality.value} (score: {metadata.quality_score:.2f})")
        
        logger.info(f"Video: {metadata.duration_seconds:.1f}s, {metadata.fps:.1f}fps, "
                   f"{metadata.width}x{metadata.height}")
        logger.info(f"Camera angle: {metadata.camera_angle.value}, "
                   f"Crop needed: {metadata.needs_crop}")
        
        # =========================================================
        # STAGE 2: Initialize components
        # =========================================================
        logger.info("Stage 2: Initializing components...")
        
        # Pose estimator (MoveNet Thunder + MediaPipe Hands)
        # Note: HybridEstimator has built-in temporal smoothing
        pose_estimator = HybridEstimator()
        
        # Lift classifier
        lift_classifier = LiftClassifier()
        
        # Fatigue handler
        fatigue_handler = FatigueHandler(fps=self.processing_fps)
        
        # Validator
        validator = RepValidator()
        
        # Rep detector will be initialized after lift type detection
        rep_detector = None
        
        # Get crop region if needed
        crop_region = analysis.crop_regions[0] if analysis.crop_regions else None
        
        # =========================================================
        # STAGE 3: Frame processing loop
        # =========================================================
        logger.info("Stage 3: Processing frames...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            result.errors.append(f"Failed to open video: {video_path}")
            return result
        
        try:
            # Calculate frame sampling
            frame_skip = max(1, int(metadata.fps / self.processing_fps))
            actual_fps = metadata.fps / frame_skip
            
            logger.info(f"Processing at {actual_fps:.1f} FPS (skip every {frame_skip} frames)")
            
            # Storage
            all_poses: List[HybridPose] = []
            all_windows: List[RepWindow] = []
            
            # Processing state
            frame_number = 0
            processed_frames = 0
            effective_lift_type = self.lift_type if self.lift_type != "auto" else None
            dominant_hand = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames
                if frame_number % frame_skip == 0:
                    timestamp = frame_number / metadata.fps
                    
                    # Apply crop if needed
                    if crop_region:
                        frame = crop_region.apply(frame)
                    
                    # Resize for faster processing (maintain aspect ratio)
                    if frame.shape[1] > 640:
                        scale = 640 / frame.shape[1]
                        new_height = int(frame.shape[0] * scale)
                        frame = cv2.resize(frame, (640, new_height), interpolation=cv2.INTER_LINEAR)
                    
                    # =============================================
                    # Pose estimation
                    # =============================================
                    pose = pose_estimator.process_frame(
                        frame,
                        frame_number=processed_frames,
                        timestamp=timestamp
                    )
                    
                    # =============================================
                    # Store pose for processing
                    # =============================================
                    # Store the HybridPose directly - has built-in temporal smoothing
                    all_poses.append(pose)
                    
                    # =============================================
                    # Fatigue tracking (uses pose directly)
                    # =============================================
                    fatigue_handler.process_pose(pose, timestamp)
                    
                    # =============================================
                    # Lift type classification (first ~5 seconds)
                    # =============================================
                    if effective_lift_type is None:
                        # Add pose to classifier
                        lift_classifier.add_pose(pose)
                        
                        if lift_classifier.has_enough_data():
                            classification = lift_classifier.classify()
                            effective_lift_type = classification.lift_type
                            dominant_hand = classification.dominant_hand
                            
                            if effective_lift_type == "unknown":
                                effective_lift_type = "snatch"  # Default
                            
                            result.detected_lift_type = effective_lift_type
                            result.lift_detection_confidence = classification.confidence
                            result.lift_detection_reasoning = classification.reasoning
                            result.dominant_hand = dominant_hand
                            
                            logger.info(f"Auto-detected: {effective_lift_type} "
                                       f"(confidence: {classification.confidence:.2%})")
                            logger.info(f"Reasoning: {classification.reasoning}")
                            
                            # Initialize rep detector
                            rep_detector = WindowRepDetector(
                                lift_type=effective_lift_type,
                                fps=actual_fps,
                                camera_angle=metadata.camera_angle,
                                dominant_hand=dominant_hand
                            )
                            
                            # Process all collected poses
                            for past_pose in all_poses:
                                window = rep_detector.process_frame(past_pose)
                                if window:
                                    all_windows.append(window)
                    
                    # =============================================
                    # Rep detection
                    # =============================================
                    if rep_detector is not None:
                        window = rep_detector.process_frame(pose)
                        if window:
                            all_windows.append(window)
                            # Track rep duration for fatigue
                            fatigue_handler.add_rep_duration(window.duration_seconds)
                    
                    processed_frames += 1
                    
                    # Progress update (every 10 frames for smooth updates)
                    if self.progress_callback and processed_frames % 5 == 0:
                        progress = frame_number / metadata.total_frames
                        self.progress_callback(progress)
                
                frame_number += 1
            
            # =========================================================
            # STAGE 4: Finalize detection
            # =========================================================
            logger.info("Stage 4: Finalizing detection...")
            
            # Handle case where lift type was never detected
            if rep_detector is None:
                logger.warning("Lift type not detected, defaulting to snatch")
                effective_lift_type = "snatch"
                result.detected_lift_type = effective_lift_type
                result.lift_detection_confidence = 0.5
                result.lift_detection_reasoning = "Default (insufficient data for detection)"
                
                rep_detector = WindowRepDetector(
                    lift_type=effective_lift_type,
                    fps=actual_fps,
                    camera_angle=metadata.camera_angle
                )
                
                for past_pose in all_poses:
                    window = rep_detector.process_frame(past_pose)
                    if window:
                        all_windows.append(window)
            
            # Finalize rep detector
            all_windows = rep_detector.finalize()
            
            result.frames_processed = processed_frames
            result.total_attempts = len(all_windows)
            
            logger.info(f"Detected {len(all_windows)} rep attempts")
            
            # =========================================================
            # STAGE 5: Validate each rep window
            # =========================================================
            logger.info("Stage 5: Validating reps...")
            
            for i, window in enumerate(all_windows):
                # Get poses for this window
                window_poses = [
                    p for p in all_poses
                    if window.start_frame <= p.frame_number <= window.end_frame
                ]
                
                # Validate
                validation = validator.validate_window(
                    window,
                    window_poses,
                    camera_angle=metadata.camera_angle,
                    fatigue_score=fatigue_handler.fatigue_indicators.overall_fatigue_score
                )
                
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
                    "timestamp_start": window.start_timestamp,
                    "timestamp_end": window.end_timestamp,
                    "frame_start": window.start_frame,
                    "frame_end": window.end_frame,
                    "duration_seconds": window.duration_seconds,
                    "classification": validation.classification,
                    "failure_reasons": validation.failure_reasons,
                    "confidence_score": validation.confidence_score,
                    "pose_confidence_avg": window.avg_confidence,
                    "peak_wrist_height": window.peak_wrist_height,
                    "fixation_frames": window.fixation_frames,
                    "reached_overhead": window.reached_overhead,
                    "metrics": validation.metrics
                }
                result.rep_results.append(rep_result)
            
            # =========================================================
            # STAGE 6: Compute analytics
            # =========================================================
            logger.info("Stage 6: Computing analytics...")
            
            result.analytics = self._compute_analytics(result.rep_results, all_windows)
            
            # Add fatigue data
            fatigue_summary = fatigue_handler.get_fatigue_summary()
            result.fatigue_score = fatigue_summary["overall_fatigue_score"]
            result.baselines_created = fatigue_summary["baselines_created"]
            result.analytics["fatigue"] = fatigue_summary
            
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
            f"Processing complete in {result.processing_time_seconds:.1f}s: "
            f"{result.total_attempts} attempts, {result.valid_reps} valid, "
            f"{result.no_reps} no-reps, {result.ambiguous_reps} ambiguous"
        )
        
        # Convert all numpy types to native Python types for JSON serialization
        result.analytics = convert_numpy_types(result.analytics)
        result.rep_results = convert_numpy_types(result.rep_results)
        
        return result
    
    def _compute_analytics(
        self,
        rep_results: List[Dict[str, Any]],
        windows: List[RepWindow]
    ) -> Dict[str, Any]:
        """
        Compute form and technique analytics.
        
        All analytics are EXPLAINABLE - traced to specific pose data.
        No black-box "form scores".
        """
        if not rep_results:
            return {}
        
        analytics = {}
        
        # =============================================
        # Valid rep rate
        # =============================================
        valid_count = sum(1 for r in rep_results if r["classification"] == RepClassification.VALID)
        analytics["valid_rep_rate"] = valid_count / len(rep_results) if rep_results else 0
        
        # =============================================
        # Lockout consistency
        # =============================================
        peak_heights = [w.peak_wrist_height for w in windows if w.peak_wrist_height > 0]
        if peak_heights:
            analytics["lockout"] = {
                "avg_peak_height": float(np.mean(peak_heights)),
                "std_peak_height": float(np.std(peak_heights)),
                "min_peak_height": float(np.min(peak_heights)),
                "max_peak_height": float(np.max(peak_heights)),
                "consistency_score": float(1 - (np.std(peak_heights) / (np.mean(peak_heights) + 0.01)))
            }
        
        # =============================================
        # Fixation analysis
        # =============================================
        fixation_frames = [w.fixation_frames for w in windows]
        if fixation_frames:
            analytics["fixation"] = {
                "avg_frames": float(np.mean(fixation_frames)),
                "min_frames": int(np.min(fixation_frames)),
                "max_frames": int(np.max(fixation_frames)),
                "reps_with_fixation": sum(1 for f in fixation_frames if f >= 3)
            }
        
        # =============================================
        # Tempo analysis
        # =============================================
        tempos = [r["duration_seconds"] for r in rep_results if r["duration_seconds"] > 0]
        if tempos:
            analytics["tempo"] = {
                "avg_seconds": float(np.mean(tempos)),
                "std_seconds": float(np.std(tempos)),
                "min_seconds": float(np.min(tempos)),
                "max_seconds": float(np.max(tempos)),
                "consistency": float(1 - (np.std(tempos) / (np.mean(tempos) + 0.01)))
            }
            
            # Tempo trend (first half vs second half)
            if len(tempos) >= 10:
                first_half = tempos[:len(tempos)//2]
                second_half = tempos[len(tempos)//2:]
                tempo_change = np.mean(second_half) - np.mean(first_half)
                analytics["tempo"]["trend_seconds"] = float(tempo_change)
                analytics["tempo"]["trend_description"] = "slowing" if tempo_change > 0.1 else ("speeding" if tempo_change < -0.1 else "stable")
        
        # =============================================
        # No-rep breakdown
        # =============================================
        no_rep_reasons = {}
        for r in rep_results:
            if r["classification"] == RepClassification.NO_REP:
                for reason in r.get("failure_reasons", []):
                    reason_str = reason.value if hasattr(reason, 'value') else str(reason)
                    no_rep_reasons[reason_str] = no_rep_reasons.get(reason_str, 0) + 1
        
        analytics["no_rep_breakdown"] = no_rep_reasons
        
        # =============================================
        # Efficiency degradation over time
        # =============================================
        if len(rep_results) >= 20:
            first_10 = rep_results[:10]
            last_10 = rep_results[-10:]
            
            first_valid = sum(1 for r in first_10 if r["classification"] == RepClassification.VALID)
            last_valid = sum(1 for r in last_10 if r["classification"] == RepClassification.VALID)
            
            analytics["degradation"] = {
                "first_10_valid_rate": first_valid / 10,
                "last_10_valid_rate": last_valid / 10,
                "efficiency_drop": (first_valid - last_valid) / 10
            }
        
        return analytics
