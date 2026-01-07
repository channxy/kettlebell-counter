"""
Computer Vision pipeline for kettlebell rep detection.

PIPELINE COMPONENTS:
1. VideoAnalyzer: Pre-analysis, auto-crop, camera angle detection, quality validation
2. HybridEstimator: MoveNet Thunder + MediaPipe Hands pose estimation
3. KeypointSmoother: Savitzky-Golay + EMA temporal smoothing
4. LiftClassifier: Auto-detect lift type from movement patterns
5. WindowRepDetector: Window-based rep detection state machine
6. RepValidator: Rule-based rep validation
7. TemporalClassifier: 1D temporal analysis for VALID/NO-REP/UNCLEAR
8. FatigueHandler: Dynamic baseline adjustment for long sets
9. VideoProcessor: Main orchestration pipeline
10. MovementClassifier: Kinematic state machines for formal rep validation

KINEMATIC CLASSIFIER:
The MovementClassifier provides formal state machine logic for Olympic
weightlifting and kettlebell sport movements. It implements:
- Velocity-based state transitions (V_y tracking)
- Explicit phase validation (segmented pull detection, rack pause enforcement)
- Body stillness detection for fixation (variance < ε)
- Automatic snatch/long-cycle disambiguation

TEMPORAL CLASSIFIER:
The TemporalClassifier provides 1D temporal analysis of rep trajectories:
- Wrist Y trajectory analysis
- Elbow angle sequence validation
- Velocity profile checking (detects segmented pulls)
- Fixation quality scoring

Usage:
    from app.cv import MovementClassifier, create_movement_classifier
    
    classifier = create_movement_classifier("snatch", fps=30.0)
    for pose in poses:
        result = classifier.process_pose(pose)
        if result:
            print(f"Rep: {result.validity.value}")
"""

from app.cv.video_analyzer import (
    VideoAnalyzer, VideoMetadata, CropRegion, CameraAngle, 
    VideoQuality, AnalysisResult
)
from app.cv.hybrid_estimator import HybridEstimator, HybridPose
from app.cv.keypoint_smoother import KeypointSmoother, KeypointFrame, SmoothedKeypoint
from app.cv.lift_classifier import LiftClassifier, LiftClassification
from app.cv.window_rep_detector import WindowRepDetector, RepWindow, RepPhase
from app.cv.rep_validator import RepValidator, ValidationResult
from app.cv.temporal_classifier import (
    TemporalClassifier, TemporalClassification, TemporalValidity, 
    TemporalFeatures, classify_rep_temporal
)
from app.cv.fatigue_handler import FatigueHandler, BaselineMetrics, FatigueIndicators
from app.cv.video_processor import VideoProcessor, ProcessingResult

# Kinematic Classifier (formal state machine logic)
from app.cv.kinematic_classifier import (
    MovementClassifier,
    create_movement_classifier,
    KinematicFrame,
    RepResult,
    RepValidity,
    LiftType,
    NoLiftReason,
    BodyStillnessDetector,
    SnatchStateMachine,
    JerkStateMachine,
    LongCycleStateMachine,
)

__all__ = [
    # Pre-analysis & Video Quality
    "VideoAnalyzer",
    "VideoMetadata",
    "CropRegion",
    "CameraAngle",
    "VideoQuality",
    "AnalysisResult",
    
    # Pose estimation
    "HybridEstimator",
    "HybridPose",
    
    # Keypoint smoothing (Savitzky-Golay + EMA)
    "KeypointSmoother",
    "KeypointFrame",
    "SmoothedKeypoint",
    
    # Lift classification
    "LiftClassifier",
    "LiftClassification",
    
    # Rep detection
    "WindowRepDetector",
    "RepWindow",
    "RepPhase",
    
    # Validation
    "RepValidator",
    "ValidationResult",
    
    # Temporal Classifier (1D trajectory analysis)
    "TemporalClassifier",
    "TemporalClassification",
    "TemporalValidity",
    "TemporalFeatures",
    "classify_rep_temporal",
    
    # Fatigue handling
    "FatigueHandler",
    "BaselineMetrics",
    "FatigueIndicators",
    
    # Main pipeline
    "VideoProcessor",
    "ProcessingResult",
    
    # Kinematic Classifier (formal state machines)
    "MovementClassifier",
    "create_movement_classifier",
    "KinematicFrame",
    "RepResult",
    "RepValidity",
    "LiftType",
    "NoLiftReason",
    "BodyStillnessDetector",
    "SnatchStateMachine",
    "JerkStateMachine",
    "LongCycleStateMachine",
]
