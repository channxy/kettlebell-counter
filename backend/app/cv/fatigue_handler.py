"""
Fatigue handling with dynamic re-baseline.

Long kettlebell sets (10-60 minutes) cause fatigue-related posture changes:
- Shoulder height drops
- Torso angle increases
- Lockout quality degrades

This module dynamically adjusts baseline metrics every 3-5 minutes
to maintain accurate detection throughout the set.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Deque
from collections import deque
import logging

try:
    from app.cv.hybrid_estimator import HybridPose
except ImportError:
    HybridPose = None

logger = logging.getLogger(__name__)


@dataclass
class BaselineMetrics:
    """Baseline posture metrics for a time window."""
    timestamp_start: float
    timestamp_end: float
    
    # Shoulder baseline (normalized Y position)
    shoulder_height_avg: float = 0.0
    shoulder_height_std: float = 0.0
    
    # Hip baseline (normalized Y position)
    hip_height_avg: float = 0.0
    hip_height_std: float = 0.0
    
    # Torso angle baseline (degrees from vertical)
    torso_angle_avg: float = 0.0
    torso_angle_std: float = 0.0
    
    # Body height ratio (for wrist height calculations)
    body_height_avg: float = 0.0
    
    # Number of frames used
    frame_count: int = 0


@dataclass
class FatigueIndicators:
    """Detected fatigue indicators."""
    shoulder_drop_percent: float = 0.0  # How much shoulders have dropped
    torso_lean_increase: float = 0.0    # Increase in torso lean
    lockout_degradation: float = 0.0    # Degradation in lockout quality
    tempo_slowdown: float = 0.0         # Increase in rep duration
    
    @property
    def overall_fatigue_score(self) -> float:
        """Combined fatigue score (0-1)."""
        return min(1.0, (
            self.shoulder_drop_percent * 0.3 +
            self.torso_lean_increase * 0.3 +
            self.lockout_degradation * 0.2 +
            self.tempo_slowdown * 0.2
        ))


class FatigueHandler:
    """
    Dynamic baseline adjustment for fatigue handling.
    
    Every 3-5 minutes:
    - Re-calculate shoulder height baseline
    - Re-calculate torso angle baseline
    - Adjust neutral posture dynamically
    
    This prevents static thresholds from causing false positives/negatives
    as the athlete fatigues.
    """
    
    # Re-baseline interval (3 minutes = 180 seconds)
    REBASELINE_INTERVAL_SECONDS = 180.0
    
    # Minimum frames for reliable baseline
    MIN_BASELINE_FRAMES = 30
    
    # Maximum allowed baseline drift (beyond this, flag as significant fatigue)
    MAX_SHOULDER_DROP = 0.10  # 10% of body height
    MAX_TORSO_LEAN_INCREASE = 15.0  # 15 degrees
    
    def __init__(self, fps: float):
        self.fps = fps
        
        # Baseline history
        self.baselines: List[BaselineMetrics] = []
        self.current_baseline: Optional[BaselineMetrics] = None
        
        # Current window data collection
        self._current_window_start: float = 0.0
        self._shoulder_heights: Deque[float] = deque(maxlen=500)
        self._hip_heights: Deque[float] = deque(maxlen=500)
        self._torso_angles: Deque[float] = deque(maxlen=500)
        self._body_heights: Deque[float] = deque(maxlen=500)
        
        # Fatigue tracking
        self.fatigue_indicators = FatigueIndicators()
        
        # Rep timing for tempo tracking
        self._rep_durations: List[float] = []
        
        logger.info(f"FatigueHandler initialized: rebaseline every {self.REBASELINE_INTERVAL_SECONDS}s")
    
    def process_pose(
        self,
        pose: "HybridPose",
        timestamp: float
    ) -> Optional[BaselineMetrics]:
        """
        Process a pose and update baseline tracking.
        
        Returns new baseline if re-baseline occurred.
        """
        # Extract posture metrics using HybridPose methods
        shoulder_height = self._get_shoulder_height(pose)
        hip_height = self._get_hip_height(pose)
        torso_angle = self._get_torso_angle(pose)
        body_height = self._get_body_height(pose)
        
        if shoulder_height is not None:
            self._shoulder_heights.append(shoulder_height)
        if hip_height is not None:
            self._hip_heights.append(hip_height)
        if torso_angle is not None:
            self._torso_angles.append(torso_angle)
        if body_height is not None:
            self._body_heights.append(body_height)
        
        # Check if we need to create initial baseline or re-baseline
        if self.current_baseline is None:
            if len(self._shoulder_heights) >= self.MIN_BASELINE_FRAMES:
                self.current_baseline = self._create_baseline(
                    0.0, timestamp
                )
                self.baselines.append(self.current_baseline)
                self._current_window_start = timestamp
                logger.info(f"Initial baseline created at {timestamp:.1f}s")
                return self.current_baseline
        else:
            # Check for re-baseline
            if timestamp - self._current_window_start >= self.REBASELINE_INTERVAL_SECONDS:
                if len(self._shoulder_heights) >= self.MIN_BASELINE_FRAMES:
                    new_baseline = self._create_baseline(
                        self._current_window_start, timestamp
                    )
                    
                    # Calculate fatigue indicators
                    self._update_fatigue_indicators(new_baseline)
                    
                    self.current_baseline = new_baseline
                    self.baselines.append(new_baseline)
                    
                    # Reset for next window
                    self._shoulder_heights.clear()
                    self._hip_heights.clear()
                    self._torso_angles.clear()
                    self._body_heights.clear()
                    self._current_window_start = timestamp
                    
                    logger.info(f"Re-baseline at {timestamp:.1f}s - "
                               f"fatigue score: {self.fatigue_indicators.overall_fatigue_score:.2%}")
                    
                    return new_baseline
        
        return None
    
    def add_rep_duration(self, duration_seconds: float):
        """Add a rep duration for tempo tracking."""
        self._rep_durations.append(duration_seconds)
    
    def get_adjusted_thresholds(self, base_thresholds: dict) -> dict:
        """
        Get fatigue-adjusted thresholds.
        
        As fatigue increases, slightly relax certain thresholds
        to account for natural degradation.
        """
        if self.current_baseline is None:
            return base_thresholds
        
        adjusted = base_thresholds.copy()
        fatigue = self.fatigue_indicators.overall_fatigue_score
        
        # Slightly relax lockout angle as fatigue increases (max 5 degrees)
        if "lockout_angle" in adjusted:
            adjusted["lockout_angle"] = adjusted["lockout_angle"] - (fatigue * 5)
        
        # Slightly increase overhead threshold tolerance
        if "overhead_height" in adjusted:
            adjusted["overhead_height"] = adjusted["overhead_height"] * (1 - fatigue * 0.1)
        
        return adjusted
    
    def _create_baseline(
        self,
        start_time: float,
        end_time: float
    ) -> BaselineMetrics:
        """Create baseline metrics from collected data."""
        return BaselineMetrics(
            timestamp_start=start_time,
            timestamp_end=end_time,
            shoulder_height_avg=float(np.mean(self._shoulder_heights)) if self._shoulder_heights else 0.0,
            shoulder_height_std=float(np.std(self._shoulder_heights)) if self._shoulder_heights else 0.0,
            hip_height_avg=float(np.mean(self._hip_heights)) if self._hip_heights else 0.0,
            hip_height_std=float(np.std(self._hip_heights)) if self._hip_heights else 0.0,
            torso_angle_avg=float(np.mean(self._torso_angles)) if self._torso_angles else 0.0,
            torso_angle_std=float(np.std(self._torso_angles)) if self._torso_angles else 0.0,
            body_height_avg=float(np.mean(self._body_heights)) if self._body_heights else 0.0,
            frame_count=len(self._shoulder_heights)
        )
    
    def _update_fatigue_indicators(self, new_baseline: BaselineMetrics):
        """Update fatigue indicators by comparing to initial baseline."""
        if len(self.baselines) == 0:
            return
        
        initial = self.baselines[0]
        
        # Shoulder drop (as percentage of body height)
        if initial.body_height_avg > 0:
            shoulder_change = new_baseline.shoulder_height_avg - initial.shoulder_height_avg
            self.fatigue_indicators.shoulder_drop_percent = max(0, shoulder_change / initial.body_height_avg)
        
        # Torso lean increase (degrees)
        torso_change = new_baseline.torso_angle_avg - initial.torso_angle_avg
        self.fatigue_indicators.torso_lean_increase = max(0, torso_change / self.MAX_TORSO_LEAN_INCREASE)
        
        # Tempo slowdown
        if len(self._rep_durations) >= 10:
            first_10_avg = np.mean(self._rep_durations[:10])
            last_10_avg = np.mean(self._rep_durations[-10:])
            if first_10_avg > 0:
                slowdown = (last_10_avg - first_10_avg) / first_10_avg
                self.fatigue_indicators.tempo_slowdown = max(0, slowdown)
    
    def _get_shoulder_height(self, pose: "HybridPose") -> Optional[float]:
        """Get average shoulder Y position."""
        left = pose.left_shoulder
        right = pose.right_shoulder
        
        if left and right:
            return (left.y + right.y) / 2
        return left.y if left else (right.y if right else None)
    
    def _get_hip_height(self, pose: "HybridPose") -> Optional[float]:
        """Get average hip Y position."""
        left = pose.left_hip
        right = pose.right_hip
        
        if left and right:
            return (left.y + right.y) / 2
        return left.y if left else (right.y if right else None)
    
    def _get_body_height(self, pose: "HybridPose") -> Optional[float]:
        """Get body height (hip to nose)."""
        nose = pose.nose
        left_hip = pose.left_hip
        right_hip = pose.right_hip
        
        if not nose or not (left_hip or right_hip):
            return None
        
        hip_y = (left_hip.y + right_hip.y) / 2 if (left_hip and right_hip) else (left_hip.y if left_hip else right_hip.y)
        return hip_y - nose.y
    
    def _get_torso_angle(self, pose: "HybridPose") -> Optional[float]:
        """Get torso angle from vertical (degrees)."""
        left_hip = pose.left_hip
        right_hip = pose.right_hip
        left_shoulder = pose.left_shoulder
        right_shoulder = pose.right_shoulder
        
        if not all([left_hip, right_hip, left_shoulder, right_shoulder]):
            return None
        
        # Midpoints
        hip_mid = np.array([(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2])
        shoulder_mid = np.array([(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2])
        
        # Torso vector
        torso_vec = shoulder_mid - hip_mid
        
        # Vertical vector (up)
        vertical = np.array([0, -1])
        
        # Angle
        cos_angle = np.dot(torso_vec, vertical) / (np.linalg.norm(torso_vec) * np.linalg.norm(vertical) + 1e-6)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return float(np.degrees(angle_rad))
    
    def get_fatigue_summary(self) -> dict:
        """Get summary of fatigue analysis."""
        return {
            "baselines_created": len(self.baselines),
            "shoulder_drop_percent": self.fatigue_indicators.shoulder_drop_percent,
            "torso_lean_increase": self.fatigue_indicators.torso_lean_increase,
            "lockout_degradation": self.fatigue_indicators.lockout_degradation,
            "tempo_slowdown": self.fatigue_indicators.tempo_slowdown,
            "overall_fatigue_score": self.fatigue_indicators.overall_fatigue_score
        }

