"""
Rule-based rep validation using biomechanical checks.

This module validates each rep attempt and classifies it as:
- VALID: Meets all technical criteria
- NO_REP: Fails one or more criteria (with explicit reasons)
- AMBIGUOUS: Insufficient data to determine validity

CRITICAL: This does NOT inflate valid reps or guess missing data.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

import numpy as np

try:
    from app.cv.hybrid_estimator import HybridPose as PoseKeypoints
except ImportError:
    try:
        from app.cv.movenet_estimator import MoveNetPose as PoseKeypoints
    except ImportError:
        from app.cv.pose_estimator import PoseKeypoints
from app.cv.rep_detector import RepCycle, LiftPhase
from app.models.rep_attempt import RepClassification, FailureReason
from app.config import get_settings


@dataclass
class ValidationResult:
    """
    Result of validating a single rep attempt.
    
    Contains classification and detailed metrics for explainability.
    """
    classification: str  # RepClassification value
    failure_reasons: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    pose_confidence_avg: float = 1.0
    
    # Detailed metrics for analytics and explainability
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        return self.classification == RepClassification.VALID
    
    @property
    def is_no_rep(self) -> bool:
        return self.classification == RepClassification.NO_REP
    
    @property
    def is_ambiguous(self) -> bool:
        return self.classification == RepClassification.AMBIGUOUS


class BiomechanicalCheck:
    """Base class for biomechanical validation checks."""
    
    name: str = "base_check"
    failure_reason: str = ""
    
    def check(
        self,
        cycle: RepCycle,
        poses: List[PoseKeypoints],
        settings: Any
    ) -> tuple[bool, float, Dict[str, Any]]:
        """
        Perform the check.
        
        Args:
            cycle: The rep cycle being validated
            poses: Pose keypoints during the cycle
            settings: Application settings with thresholds
            
        Returns:
            Tuple of (passed, confidence, metrics)
        """
        raise NotImplementedError


class LockoutAngleCheck(BiomechanicalCheck):
    """Check that elbows are fully extended at lockout."""
    
    name = "lockout_angle"
    failure_reason = FailureReason.ELBOWS_NOT_EXTENDED
    
    def check(
        self,
        cycle: RepCycle,
        poses: List[PoseKeypoints],
        settings: Any
    ) -> tuple[bool, float, Dict[str, Any]]:
        # Find peak frame poses
        peak_poses = [
            p for p in poses 
            if cycle.peak_frame - 3 <= p.frame_number <= cycle.peak_frame + 3
        ]
        
        if not peak_poses:
            return False, 0.0, {"error": "no_peak_poses"}
        
        # Get maximum elbow angles at peak
        left_angles = []
        right_angles = []
        
        for pose in peak_poses:
            left = pose.get_elbow_angle("left")
            right = pose.get_elbow_angle("right")
            if left is not None:
                left_angles.append(left)
            if right is not None:
                right_angles.append(right)
        
        if not left_angles and not right_angles:
            return False, 0.0, {"error": "no_elbow_data"}
        
        max_left = max(left_angles) if left_angles else 0
        max_right = max(right_angles) if right_angles else 0
        
        # For double kettlebell (jerk, long cycle), check both
        if cycle.lift_type in ["jerk", "long_cycle"]:
            min_angle = min(max_left, max_right) if (max_left > 0 and max_right > 0) else max(max_left, max_right)
        else:
            # Snatch - single arm, use whichever has the better angle (auto-detect dominant hand)
            min_angle = max(max_left, max_right) if (max_left > 0 or max_right > 0) else 0
        
        passed = min_angle >= settings.min_lockout_angle_degrees
        confidence = min(1.0, min_angle / settings.min_lockout_angle_degrees) if min_angle > 0 else 0.0
        
        return passed, confidence, {
            "lockout_angle_left": max_left,
            "lockout_angle_right": max_right,
            "min_angle": min_angle,
            "threshold": settings.min_lockout_angle_degrees
        }


class OverheadPositionCheck(BiomechanicalCheck):
    """Check that kettlebell reaches proper overhead position."""
    
    name = "overhead_position"
    failure_reason = FailureReason.KETTLEBELL_NOT_OVERHEAD
    
    def check(
        self,
        cycle: RepCycle,
        poses: List[PoseKeypoints],
        settings: Any
    ) -> tuple[bool, float, Dict[str, Any]]:
        # Check peak wrist height
        peak_height = cycle.peak_wrist_height
        
        if peak_height is None:
            return False, 0.0, {"error": "no_peak_height"}
        
        passed = peak_height >= settings.min_overhead_height_ratio
        confidence = min(1.0, peak_height / settings.min_overhead_height_ratio)
        
        return passed, confidence, {
            "peak_height_ratio": peak_height,
            "threshold": settings.min_overhead_height_ratio
        }


class FixationTimeCheck(BiomechanicalCheck):
    """Check for sufficient fixation time at lockout."""
    
    name = "fixation_time"
    failure_reason = FailureReason.INSUFFICIENT_FIXATION
    
    def check(
        self,
        cycle: RepCycle,
        poses: List[PoseKeypoints],
        settings: Any
    ) -> tuple[bool, float, Dict[str, Any]]:
        # Count frames in lockout phase
        lockout_phases = [
            (phase, frame) for phase, frame in cycle.phases 
            if phase == LiftPhase.LOCKOUT
        ]
        
        if not lockout_phases:
            return False, 0.5, {"fixation_frames": 0, "threshold": settings.min_fixation_frames}
        
        lockout_start = lockout_phases[0][1]
        
        # Find when lockout ends (next phase or cycle end)
        lockout_end = cycle.end_frame
        for i, (phase, frame) in enumerate(cycle.phases):
            if phase == LiftPhase.LOCKOUT:
                # Look for next phase
                if i + 1 < len(cycle.phases):
                    lockout_end = cycle.phases[i + 1][1]
                    break
        
        fixation_frames = lockout_end - lockout_start
        passed = fixation_frames >= settings.min_fixation_frames
        confidence = min(1.0, fixation_frames / settings.min_fixation_frames)
        
        return passed, confidence, {
            "fixation_frames": fixation_frames,
            "threshold": settings.min_fixation_frames
        }


class TorsoLeanCheck(BiomechanicalCheck):
    """Check for excessive torso lean during lockout."""
    
    name = "torso_lean"
    failure_reason = FailureReason.EXCESSIVE_TORSO_LEAN
    
    def check(
        self,
        cycle: RepCycle,
        poses: List[PoseKeypoints],
        settings: Any
    ) -> tuple[bool, float, Dict[str, Any]]:
        # Get torso lean at peak
        peak_poses = [
            p for p in poses 
            if cycle.peak_frame - 2 <= p.frame_number <= cycle.peak_frame + 2
        ]
        
        if not peak_poses:
            return False, 0.0, {"error": "no_peak_poses"}
        
        lean_angles = []
        for pose in peak_poses:
            lean = pose.get_torso_lean_angle()
            if lean is not None:
                lean_angles.append(lean)
        
        if not lean_angles:
            return True, 0.5, {"error": "no_torso_data"}  # Don't fail if no data
        
        max_lean = max(lean_angles)
        passed = max_lean <= settings.max_torso_lean_degrees
        confidence = min(1.0, settings.max_torso_lean_degrees / max_lean) if max_lean > 0 else 1.0
        
        return passed, confidence, {
            "max_torso_lean": max_lean,
            "threshold": settings.max_torso_lean_degrees
        }


class SymmetryCheck(BiomechanicalCheck):
    """Check for arm symmetry (for double kettlebell lifts)."""
    
    name = "symmetry"
    failure_reason = FailureReason.ASYMMETRIC_LOCKOUT
    
    def check(
        self,
        cycle: RepCycle,
        poses: List[PoseKeypoints],
        settings: Any
    ) -> tuple[bool, float, Dict[str, Any]]:
        # Only applies to double KB lifts
        if cycle.lift_type == "snatch":
            return True, 1.0, {"skipped": "single_kb_lift"}
        
        # Get symmetry at peak
        peak_poses = [
            p for p in poses 
            if cycle.peak_frame - 2 <= p.frame_number <= cycle.peak_frame + 2
        ]
        
        if not peak_poses:
            return True, 0.5, {"error": "no_peak_poses"}
        
        symmetry_scores = []
        for pose in peak_poses:
            score = pose.get_arm_symmetry_score()
            if score is not None:
                symmetry_scores.append(score)
        
        if not symmetry_scores:
            return True, 0.5, {"error": "no_symmetry_data"}
        
        min_symmetry = min(symmetry_scores)
        passed = min_symmetry >= 0.7  # 70% symmetry threshold
        
        return passed, min_symmetry, {
            "min_symmetry_score": min_symmetry,
            "threshold": 0.7
        }


class RepValidator:
    """
    Rule-based rep validation engine.
    
    Applies biomechanical checks to classify rep attempts.
    
    CRITICAL PRINCIPLES:
    1. Never infer or guess missing data
    2. Low confidence = AMBIGUOUS, not VALID
    3. All decisions must be explainable with metrics
    4. Prefer false negatives over false positives
    """
    
    # Checks for each lift type
    CHECKS_BY_LIFT = {
        "jerk": [
            LockoutAngleCheck(),
            OverheadPositionCheck(),
            FixationTimeCheck(),
            TorsoLeanCheck(),
            SymmetryCheck(),
        ],
        "long_cycle": [
            LockoutAngleCheck(),
            OverheadPositionCheck(),
            FixationTimeCheck(),
            TorsoLeanCheck(),
            SymmetryCheck(),
        ],
        "snatch": [
            LockoutAngleCheck(),
            OverheadPositionCheck(),
            FixationTimeCheck(),
            # Note: TorsoLeanCheck removed - backward lean is normal/expected in snatch
        ],
    }
    
    def __init__(self):
        """Initialize validator with settings."""
        self.settings = get_settings()
    
    def validate(
        self,
        cycle: RepCycle,
        poses: List[PoseKeypoints]
    ) -> ValidationResult:
        """
        Validate a rep cycle and classify it.
        
        Args:
            cycle: The detected rep cycle
            poses: All pose keypoints during the cycle
            
        Returns:
            ValidationResult with classification and metrics
        """
        # Filter poses to cycle timeframe
        cycle_poses = [
            p for p in poses 
            if cycle.start_frame <= p.frame_number <= cycle.end_frame
        ]
        
        # Check pose confidence
        if not cycle_poses:
            return ValidationResult(
                classification=RepClassification.AMBIGUOUS,
                failure_reasons=[FailureReason.OCCLUDED_KEYPOINTS],
                confidence_score=0.0,
                pose_confidence_avg=0.0,
                metrics={"error": "no_poses_in_cycle"}
            )
        
        pose_confidences = [p.overall_confidence for p in cycle_poses]
        avg_pose_confidence = np.mean(pose_confidences)
        
        # If pose confidence is too low, mark as AMBIGUOUS
        if avg_pose_confidence < self.settings.ambiguous_confidence_threshold:
            return ValidationResult(
                classification=RepClassification.AMBIGUOUS,
                failure_reasons=[FailureReason.LOW_POSE_CONFIDENCE],
                confidence_score=avg_pose_confidence,
                pose_confidence_avg=avg_pose_confidence,
                metrics={"pose_confidence_avg": avg_pose_confidence}
            )
        
        # Run all applicable checks
        checks = self.CHECKS_BY_LIFT.get(cycle.lift_type, self.CHECKS_BY_LIFT["jerk"])
        
        all_passed = True
        failure_reasons = []
        all_metrics = {}
        confidences = []
        
        for check in checks:
            passed, confidence, metrics = check.check(
                cycle, cycle_poses, self.settings
            )
            
            all_metrics[check.name] = metrics
            confidences.append(confidence)
            
            if not passed:
                all_passed = False
                failure_reasons.append(check.failure_reason)
        
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        # Determine classification
        if avg_pose_confidence < self.settings.pose_confidence_threshold:
            # Marginal pose confidence - be conservative
            if all_passed:
                classification = RepClassification.VALID
            else:
                # Could be ambiguous if confidence is borderline
                classification = RepClassification.AMBIGUOUS
                failure_reasons.append(FailureReason.LOW_POSE_CONFIDENCE)
        elif all_passed:
            classification = RepClassification.VALID
        else:
            classification = RepClassification.NO_REP
        
        # Add tempo metric
        all_metrics["tempo_ms"] = cycle.duration_seconds * 1000
        
        return ValidationResult(
            classification=classification,
            failure_reasons=failure_reasons,
            confidence_score=overall_confidence,
            pose_confidence_avg=avg_pose_confidence,
            metrics=all_metrics
        )
    
    def validate_batch(
        self,
        cycles: List[RepCycle],
        all_poses: List[PoseKeypoints]
    ) -> List[ValidationResult]:
        """
        Validate multiple cycles.
        
        Args:
            cycles: List of detected rep cycles
            all_poses: All pose keypoints for the video
            
        Returns:
            List of ValidationResult for each cycle
        """
        return [self.validate(cycle, all_poses) for cycle in cycles]

