"""
Rep validation for kettlebell lifts - COMPETITION ACCURATE.

Validates detected rep WINDOWS to classify as:
- VALID: Competition-compliant
- NO_REP: Failed one or more criteria (with reasons)
- AMBIGUOUS: Low confidence, cannot determine

VALIDATION STRATEGY:
1. Rule-based instantaneous checks (elbow angle, height, fixation)
2. Temporal trajectory analysis (1D classifier for movement quality)
3. Combined scoring for final classification

VALIDATION CRITERIA:
1. Elbow extension ≥ 165-170° (camera angle dependent)
2. Wrist clearly above head (75%+ of body height)
3. Fixation duration ≥ 300-500ms with visible stability
4. Pose confidence stable throughout
5. No excessive torso lean (jerk/LC only)
6. Smooth trajectory (no segmented pulls)
7. Proper velocity profile (continuous ascent to overhead)

All decisions are auditable and explainable.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import logging

from app.cv.window_rep_detector import RepWindow
from app.cv.video_analyzer import CameraAngle
from app.models.rep_attempt import RepClassification, FailureReason

try:
    from app.cv.hybrid_estimator import HybridPose
except ImportError:
    HybridPose = None

try:
    from app.cv.temporal_classifier import TemporalClassifier, TemporalValidity
except ImportError:
    TemporalClassifier = None
    TemporalValidity = None

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of rep validation."""
    classification: RepClassification
    confidence_score: float
    failure_reasons: List[FailureReason] = field(default_factory=list)
    pose_confidence_avg: float = 0.0
    
    # Detailed metrics for explainability
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Check results
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)


@dataclass
class ValidationThresholds:
    """Thresholds for validation checks - COMPETITION ACCURATE."""
    # Elbow angle (degrees) - strict for competition
    min_lockout_angle: float = 165.0
    
    # Wrist height (ratio of body height) - must be clearly overhead
    min_overhead_height: float = 0.75
    
    # Fixation (frames) - visible pause required
    min_fixation_frames: int = 4
    
    # Pose confidence
    min_pose_confidence: float = 0.40
    ambiguous_confidence: float = 0.30
    
    # Torso lean (degrees) - only for jerk/LC
    max_torso_lean: float = 18.0
    
    # Temporal classifier weight (0-1)
    temporal_weight: float = 0.3  # 30% of final score from temporal analysis
    
    @classmethod
    def for_lift_and_angle(
        cls, 
        lift_type: str, 
        camera_angle: CameraAngle,
        fps: float = 15.0
    ) -> "ValidationThresholds":
        """Create thresholds adapted to lift type and camera angle."""
        thresholds = cls()
        
        # Camera angle adjustments
        if camera_angle == CameraAngle.SIDE:
            # Side view: elbow angle most reliable
            thresholds.min_lockout_angle = 165.0
            thresholds.min_fixation_frames = max(4, int(0.30 * fps))  # ~300ms
        elif camera_angle == CameraAngle.FRONT:
            # Front view: elbow unreliable, require longer fixation
            thresholds.min_lockout_angle = 155.0  # More lenient
            thresholds.min_fixation_frames = max(6, int(0.45 * fps))  # ~450ms
        else:
            # Diagonal or unknown
            thresholds.min_lockout_angle = 160.0
            thresholds.min_fixation_frames = max(5, int(0.35 * fps))  # ~350ms
        
        # Lift type adjustments
        if lift_type == "snatch":
            # Snatch: no torso lean check, stricter overhead
            thresholds.max_torso_lean = 999.0  # Disabled
            thresholds.min_overhead_height = 0.78  # Must be clearly overhead
        elif lift_type == "jerk":
            thresholds.max_torso_lean = 15.0  # Strict for jerk
            thresholds.min_overhead_height = 0.75
        else:  # long_cycle
            thresholds.max_torso_lean = 18.0
            thresholds.min_overhead_height = 0.75
        
        return thresholds


class RepValidator:
    """
    Validates rep windows using rule-based biomechanical checks.
    
    All checks are:
    1. Explainable (clear criteria)
    2. Auditable (traced to pose data)
    3. Configurable (per lift type and camera angle)
    """
    
    def __init__(self):
        self.lift_type: Optional[str] = None
        self.camera_angle: CameraAngle = CameraAngle.UNKNOWN
    
    def validate_window(
        self,
        window: RepWindow,
        poses: List["HybridPose"],
        camera_angle: CameraAngle = CameraAngle.UNKNOWN,
        fatigue_score: float = 0.0
    ) -> ValidationResult:
        """
        Validate a rep window.
        
        Args:
            window: The detected rep window
            poses: HybridPose list within the window
            camera_angle: Detected camera angle
            fatigue_score: Current fatigue level (0-1)
            
        Returns:
            ValidationResult with classification and reasons
        """
        # Get thresholds for this lift and camera angle
        fps = len(poses) / window.duration_seconds if window.duration_seconds > 0 else 10.0
        thresholds = ValidationThresholds.for_lift_and_angle(
            window.lift_type, camera_angle, fps
        )
        
        # Apply fatigue adjustment (relax slightly as fatigue increases)
        if fatigue_score > 0.3:
            thresholds.min_lockout_angle -= fatigue_score * 5  # Up to 5 degree relaxation
            thresholds.min_overhead_height -= fatigue_score * 0.05  # Slight height relaxation
        
        result = ValidationResult(
            classification=RepClassification.VALID,
            confidence_score=1.0,
            pose_confidence_avg=window.avg_confidence,
            metrics={}
        )
        
        # =========================================================
        # CHECK 1: Pose Confidence
        # =========================================================
        if window.avg_confidence < thresholds.ambiguous_confidence:
            result.classification = RepClassification.AMBIGUOUS
            result.confidence_score = window.avg_confidence  # Reflect actual pose confidence
            result.failure_reasons.append(FailureReason.LOW_POSE_CONFIDENCE)
            result.checks_failed.append("pose_confidence")
            result.metrics["pose_confidence"] = {
                "value": window.avg_confidence,
                "threshold": thresholds.ambiguous_confidence,
                "note": "Pose confidence too low to judge"
            }
            return result  # Early return for ambiguous
        
        if window.avg_confidence < thresholds.min_pose_confidence:
            result.confidence_score *= 0.7
            result.checks_failed.append("pose_confidence_warning")
        else:
            result.checks_passed.append("pose_confidence")
        
        result.metrics["pose_confidence"] = {
            "value": window.avg_confidence,
            "threshold": thresholds.min_pose_confidence,
            "passed": window.avg_confidence >= thresholds.min_pose_confidence
        }
        
        # =========================================================
        # CHECK 2: Overhead Position (Did wrist reach overhead?)
        # =========================================================
        reached_overhead = window.peak_wrist_height >= thresholds.min_overhead_height
        
        if not reached_overhead:
            result.classification = RepClassification.NO_REP
            result.failure_reasons.append(FailureReason.INCOMPLETE_LOCKOUT)
            result.checks_failed.append("overhead_position")
        else:
            result.checks_passed.append("overhead_position")
        
        result.metrics["overhead_position"] = {
            "peak_height": window.peak_wrist_height,
            "threshold": thresholds.min_overhead_height,
            "passed": reached_overhead
        }
        
        # =========================================================
        # CHECK 3: Fixation Duration
        # =========================================================
        had_fixation = window.fixation_frames >= thresholds.min_fixation_frames
        
        if not had_fixation:
            if result.classification == RepClassification.VALID:
                result.classification = RepClassification.NO_REP
            result.failure_reasons.append(FailureReason.INSUFFICIENT_FIXATION)
            result.checks_failed.append("fixation_duration")
        else:
            result.checks_passed.append("fixation_duration")
        
        result.metrics["fixation"] = {
            "frames": window.fixation_frames,
            "threshold": thresholds.min_fixation_frames,
            "passed": had_fixation
        }
        
        # =========================================================
        # CHECK 4: Elbow Extension (at peak)
        # =========================================================
        if window.max_elbow_angle > 0:
            good_lockout = window.max_elbow_angle >= thresholds.min_lockout_angle
            
            if not good_lockout:
                if result.classification == RepClassification.VALID:
                    result.classification = RepClassification.NO_REP
                result.failure_reasons.append(FailureReason.ELBOWS_NOT_EXTENDED)
                result.checks_failed.append("elbow_extension")
            else:
                result.checks_passed.append("elbow_extension")
            
            result.metrics["elbow_angle"] = {
                "max_angle": window.max_elbow_angle,
                "threshold": thresholds.min_lockout_angle,
                "passed": good_lockout
            }
        else:
            # No elbow data - reduce confidence but don't fail
            result.confidence_score *= 0.8
            result.metrics["elbow_angle"] = {
                "note": "Elbow angle not available",
                "passed": None
            }
        
        # =========================================================
        # CHECK 5: Torso Lean (jerk/LC only)
        # =========================================================
        if window.lift_type in ["jerk", "long_cycle"] and poses:
            torso_lean = self._calculate_max_torso_lean(poses)
            
            if torso_lean is not None:
                good_torso = torso_lean <= thresholds.max_torso_lean
                
                if not good_torso:
                    if result.classification == RepClassification.VALID:
                        result.classification = RepClassification.NO_REP
                    result.failure_reasons.append(FailureReason.EXCESSIVE_TORSO_LEAN)
                    result.checks_failed.append("torso_lean")
                else:
                    result.checks_passed.append("torso_lean")
                
                result.metrics["torso_lean"] = {
                    "max_angle": torso_lean,
                    "threshold": thresholds.max_torso_lean,
                    "passed": good_torso
                }
        
        # =========================================================
        # CHECK 6: Temporal Trajectory Analysis (if classifier available)
        # =========================================================
        temporal_score = 1.0
        if TemporalClassifier is not None and window.wrist_heights:
            try:
                temporal_classifier = TemporalClassifier(
                    fps=fps,
                    lift_type=window.lift_type
                )
                temporal_result = temporal_classifier.classify(
                    wrist_heights=window.wrist_heights,
                    elbow_angles=window.elbow_angles if window.elbow_angles else None
                )
                
                temporal_score = temporal_result.confidence
                
                if temporal_result.validity == TemporalValidity.VALID:
                    result.checks_passed.append("temporal_trajectory")
                elif temporal_result.validity == TemporalValidity.NO_REP:
                    result.checks_failed.append("temporal_trajectory")
                    # Add specific temporal failures
                    for reason in temporal_result.reasons:
                        if "Segmented pull" in reason:
                            result.failure_reasons.append(FailureReason.SEGMENTED_PULL)
                        elif "range of motion" in reason.lower():
                            result.failure_reasons.append(FailureReason.INCOMPLETE_LOCKOUT)
                else:
                    # UNCLEAR - don't add to passed or failed
                    pass
                
                result.metrics["temporal_analysis"] = {
                    "validity": temporal_result.validity.value,
                    "confidence": temporal_result.confidence,
                    "checks_passed": temporal_result.checks_passed,
                    "checks_failed": temporal_result.checks_failed,
                    "reasons": temporal_result.reasons
                }
                
            except Exception as e:
                logger.warning(f"Temporal classification failed: {e}")
                temporal_score = 0.5  # Neutral
        
        # =========================================================
        # FINAL: Calculate Combined Score
        # =========================================================
        total_checks = len(result.checks_passed) + len(result.checks_failed)
        if total_checks > 0:
            rule_based_score = len(result.checks_passed) / total_checks
        else:
            rule_based_score = 0.5
        
        # Combine rule-based and temporal scores
        combined_score = (
            rule_based_score * (1 - thresholds.temporal_weight) +
            temporal_score * thresholds.temporal_weight
        )
        result.confidence_score = combined_score
        
        # Final classification based on combined score
        if combined_score >= 0.75:
            if result.classification == RepClassification.NO_REP and combined_score >= 0.85:
                # Temporal analysis overrides minor rule failures
                result.classification = RepClassification.VALID
        elif combined_score < 0.45:
            result.classification = RepClassification.NO_REP
        elif result.classification == RepClassification.VALID and combined_score < 0.6:
            result.classification = RepClassification.AMBIGUOUS
        
        return result
    
    def _calculate_max_torso_lean(self, poses: List["HybridPose"]) -> Optional[float]:
        """Calculate maximum torso lean angle from poses."""
        lean_angles = []
        
        for pose in poses:
            left_hip = pose.left_hip
            right_hip = pose.right_hip
            left_shoulder = pose.left_shoulder
            right_shoulder = pose.right_shoulder
            
            if not all([left_hip, right_hip, left_shoulder, right_shoulder]):
                continue
            
            # Midpoints
            hip_mid = np.array([(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2])
            shoulder_mid = np.array([(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2])
            
            # Torso vector
            torso_vec = shoulder_mid - hip_mid
            
            # Vertical vector (up in image coordinates)
            vertical = np.array([0, -1])
            
            # Angle
            cos_angle = np.dot(torso_vec, vertical) / (np.linalg.norm(torso_vec) * np.linalg.norm(vertical) + 1e-6)
            angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            
            lean_angles.append(float(np.degrees(angle_rad)))
        
        return max(lean_angles) if lean_angles else None
    
    # Legacy compatibility method
    def validate(self, cycle, poses) -> ValidationResult:
        """Legacy validation method for compatibility."""
        # Convert old cycle to new window format
        from app.cv.window_rep_detector import RepWindow
        
        if isinstance(cycle, RepWindow):
            window = cycle
        else:
            # Create minimal window from old cycle
            window = RepWindow(
                start_frame=cycle.start_frame,
                end_frame=cycle.end_frame,
                start_timestamp=cycle.start_timestamp,
                end_timestamp=cycle.end_timestamp,
                lift_type=cycle.lift_type,
                peak_wrist_height=getattr(cycle, 'peak_wrist_height', 0.7),
                fixation_frames=getattr(cycle, 'fixation_frames', 5),
                avg_confidence=getattr(cycle, 'detection_confidence', 0.8)
            )
        
        # Create empty poses list for legacy compatibility
        poses = []
        
        return self.validate_window(window, poses)
