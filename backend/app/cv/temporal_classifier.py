"""
1D Temporal Classifier for Rep Validation.

Classifies rep attempts as VALID / NO-REP / UNCLEAR based on temporal features.

This is the "optional but powerful" component that improves accuracy by analyzing:
1. Wrist Y trajectory (height over time)
2. Elbow angle sequence
3. Bell velocity profile

APPROACH: Simple statistical/rule-based temporal analysis
- Does NOT require training data
- Uses domain knowledge of kettlebell sport biomechanics
- Provides explainable classifications

This can be upgraded to a learned 1D CNN/LSTM in the future if training data
is collected, but the rule-based approach works well for competition-standard lifts.
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TemporalValidity(Enum):
    """Temporal classification result."""
    VALID = "valid"           # Clear valid rep pattern
    NO_REP = "no_rep"         # Clear invalid pattern
    UNCLEAR = "unclear"       # Cannot determine with confidence


@dataclass
class TemporalFeatures:
    """Extracted temporal features from a rep window."""
    # Wrist trajectory features
    peak_height: float = 0.0
    min_height: float = 0.0
    height_range: float = 0.0
    time_to_peak_ratio: float = 0.0      # Time to reach peak / total duration
    height_at_25pct: float = 0.0         # Height at 25% of duration
    height_at_75pct: float = 0.0         # Height at 75% of duration
    
    # Velocity features
    peak_ascent_velocity: float = 0.0
    peak_descent_velocity: float = 0.0
    velocity_reversal_count: int = 0     # Should be 1 for clean rep
    time_above_threshold: float = 0.0    # Time in overhead zone
    
    # Elbow angle features
    max_elbow_angle: float = 0.0
    min_elbow_angle: float = 0.0
    elbow_range: float = 0.0
    elbow_at_peak: float = 0.0           # Elbow angle at peak height
    
    # Fixation features
    max_stability_duration: int = 0      # Longest stable period (frames)
    stable_at_peak: bool = False         # Was stable when at peak?
    fixation_quality: float = 0.0        # 0-1 quality score
    
    # Symmetry features
    ascent_descent_ratio: float = 0.0    # Ascent time / descent time
    trajectory_smoothness: float = 0.0   # Inverse of jerk (3rd derivative)


@dataclass
class TemporalClassification:
    """Result of temporal classification."""
    validity: TemporalValidity
    confidence: float
    features: TemporalFeatures
    reasons: List[str] = field(default_factory=list)
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)


class TemporalClassifier:
    """
    Classifies rep attempts using temporal feature analysis.
    
    Analyzes the TRAJECTORY of movement over time, not just instantaneous values.
    This catches issues that frame-by-frame analysis misses:
    - Jerky/unstable movements
    - Incomplete range of motion
    - Missing fixation pause
    - Segmented pulls (velocity drops before overhead)
    """
    
    # Thresholds for valid rep
    MIN_HEIGHT_RANGE = 0.50           # Must have significant vertical movement
    MIN_PEAK_HEIGHT = 0.70            # Must reach clearly overhead
    MIN_OVERHEAD_TIME_RATIO = 0.10    # At least 10% of rep in overhead zone
    MIN_LOCKOUT_ANGLE = 160.0         # Elbow angle for lockout
    MIN_FIXATION_QUALITY = 0.5        # Minimum fixation quality
    
    # Velocity thresholds
    MIN_PEAK_VELOCITY = 0.3           # Minimum ascent velocity
    MAX_VELOCITY_REVERSALS = 2        # Allow minor reversals, not segmented pulls
    
    # Trajectory quality
    MIN_SMOOTHNESS = 0.3              # Minimum trajectory smoothness
    ASCENT_DESCENT_MIN = 0.3          # Min ratio (not too lopsided)
    ASCENT_DESCENT_MAX = 3.0          # Max ratio
    
    def __init__(self, fps: float = 15.0, lift_type: str = "snatch"):
        self.fps = fps
        self.lift_type = lift_type
        self.frame_time = 1.0 / fps
        
    def classify(
        self,
        wrist_heights: List[float],
        elbow_angles: Optional[List[float]] = None,
        timestamps: Optional[List[float]] = None
    ) -> TemporalClassification:
        """
        Classify a rep attempt based on temporal features.
        
        Args:
            wrist_heights: Wrist height values over the rep window (normalized)
            elbow_angles: Optional elbow angle values (degrees)
            timestamps: Optional timestamps (seconds)
            
        Returns:
            TemporalClassification with validity and reasoning
        """
        if len(wrist_heights) < 5:
            return TemporalClassification(
                validity=TemporalValidity.UNCLEAR,
                confidence=0.0,
                features=TemporalFeatures(),
                reasons=["Insufficient data for temporal analysis"]
            )
        
        # Convert to numpy arrays
        heights = np.array(wrist_heights)
        angles = np.array(elbow_angles) if elbow_angles else None
        
        # Extract features
        features = self._extract_features(heights, angles)
        
        # Run classification checks
        checks_passed = []
        checks_failed = []
        reasons = []
        score = 0.0
        max_score = 0.0
        
        # =====================================================
        # CHECK 1: Height Range (must have full range of motion)
        # =====================================================
        max_score += 1.0
        if features.height_range >= self.MIN_HEIGHT_RANGE:
            checks_passed.append("height_range")
            score += 1.0
        else:
            checks_failed.append("height_range")
            reasons.append(f"Insufficient range of motion: {features.height_range:.2f} < {self.MIN_HEIGHT_RANGE}")
        
        # =====================================================
        # CHECK 2: Peak Height (must reach overhead)
        # =====================================================
        max_score += 1.0
        if features.peak_height >= self.MIN_PEAK_HEIGHT:
            checks_passed.append("peak_height")
            score += 1.0
        else:
            checks_failed.append("peak_height")
            reasons.append(f"Did not reach overhead: {features.peak_height:.2f} < {self.MIN_PEAK_HEIGHT}")
        
        # =====================================================
        # CHECK 3: Time in Overhead Zone
        # =====================================================
        max_score += 1.0
        if features.time_above_threshold >= self.MIN_OVERHEAD_TIME_RATIO:
            checks_passed.append("overhead_time")
            score += 1.0
        else:
            checks_failed.append("overhead_time")
            reasons.append(f"Insufficient time overhead: {features.time_above_threshold:.1%}")
        
        # =====================================================
        # CHECK 4: Velocity Profile (no segmented pulls)
        # =====================================================
        max_score += 1.0
        if features.velocity_reversal_count <= self.MAX_VELOCITY_REVERSALS:
            checks_passed.append("velocity_profile")
            score += 1.0
        else:
            checks_failed.append("velocity_profile")
            reasons.append(f"Segmented pull detected: {features.velocity_reversal_count} velocity reversals")
        
        # =====================================================
        # CHECK 5: Elbow Lockout (if available)
        # =====================================================
        if angles is not None and len(angles) > 0:
            max_score += 1.0
            if features.elbow_at_peak >= self.MIN_LOCKOUT_ANGLE:
                checks_passed.append("elbow_lockout")
                score += 1.0
            else:
                checks_failed.append("elbow_lockout")
                reasons.append(f"Incomplete lockout: {features.elbow_at_peak:.0f}° < {self.MIN_LOCKOUT_ANGLE}°")
        
        # =====================================================
        # CHECK 6: Fixation Quality
        # =====================================================
        max_score += 1.0
        if features.fixation_quality >= self.MIN_FIXATION_QUALITY:
            checks_passed.append("fixation")
            score += 1.0
        else:
            checks_failed.append("fixation")
            reasons.append(f"Insufficient fixation: quality {features.fixation_quality:.2f}")
        
        # =====================================================
        # CHECK 7: Trajectory Smoothness
        # =====================================================
        max_score += 0.5  # Lower weight - more of a form issue
        if features.trajectory_smoothness >= self.MIN_SMOOTHNESS:
            checks_passed.append("smoothness")
            score += 0.5
        else:
            checks_failed.append("smoothness")
            reasons.append(f"Jerky movement detected")
        
        # =====================================================
        # CHECK 8: Ascent/Descent Balance
        # =====================================================
        max_score += 0.5
        if self.ASCENT_DESCENT_MIN <= features.ascent_descent_ratio <= self.ASCENT_DESCENT_MAX:
            checks_passed.append("balance")
            score += 0.5
        else:
            checks_failed.append("balance")
            if features.ascent_descent_ratio < self.ASCENT_DESCENT_MIN:
                reasons.append("Descent much longer than ascent - possible catch issue")
            else:
                reasons.append("Ascent much longer than descent - possible struggle")
        
        # =====================================================
        # FINAL: Determine Validity
        # =====================================================
        confidence = score / max_score if max_score > 0 else 0.0
        
        if confidence >= 0.8:
            validity = TemporalValidity.VALID
        elif confidence <= 0.4:
            validity = TemporalValidity.NO_REP
        else:
            validity = TemporalValidity.UNCLEAR
        
        return TemporalClassification(
            validity=validity,
            confidence=confidence,
            features=features,
            reasons=reasons,
            checks_passed=checks_passed,
            checks_failed=checks_failed
        )
    
    def _extract_features(
        self,
        heights: np.ndarray,
        angles: Optional[np.ndarray]
    ) -> TemporalFeatures:
        """Extract temporal features from trajectory data."""
        features = TemporalFeatures()
        n_frames = len(heights)
        
        # Smooth the data for cleaner analysis
        if n_frames >= 7:
            heights_smooth = savgol_filter(heights, min(7, n_frames if n_frames % 2 == 1 else n_frames - 1), 2)
        else:
            heights_smooth = gaussian_filter1d(heights, sigma=1)
        
        # Basic height features
        features.peak_height = float(np.max(heights_smooth))
        features.min_height = float(np.min(heights_smooth))
        features.height_range = features.peak_height - features.min_height
        
        # Time to peak
        peak_idx = int(np.argmax(heights_smooth))
        features.time_to_peak_ratio = peak_idx / n_frames if n_frames > 0 else 0
        
        # Heights at specific points
        idx_25 = int(n_frames * 0.25)
        idx_75 = int(n_frames * 0.75)
        features.height_at_25pct = float(heights_smooth[idx_25]) if idx_25 < n_frames else 0
        features.height_at_75pct = float(heights_smooth[idx_75]) if idx_75 < n_frames else 0
        
        # Velocity analysis
        velocity = np.diff(heights_smooth) * self.fps
        if len(velocity) > 0:
            features.peak_ascent_velocity = float(np.max(velocity))
            features.peak_descent_velocity = float(np.min(velocity))
            
            # Count significant velocity reversals
            # A reversal is when velocity changes sign
            signs = np.sign(velocity)
            sign_changes = np.diff(signs)
            # Only count significant reversals (not noise)
            significant_reversals = np.sum(np.abs(sign_changes) > 0.5)
            features.velocity_reversal_count = int(significant_reversals)
        
        # Time above overhead threshold
        overhead_mask = heights_smooth >= self.MIN_PEAK_HEIGHT * 0.9
        features.time_above_threshold = float(np.sum(overhead_mask)) / n_frames
        
        # Elbow features
        if angles is not None and len(angles) > 0:
            if len(angles) >= 7:
                angles_smooth = savgol_filter(angles, min(7, len(angles) if len(angles) % 2 == 1 else len(angles) - 1), 2)
            else:
                angles_smooth = angles
            
            features.max_elbow_angle = float(np.max(angles_smooth))
            features.min_elbow_angle = float(np.min(angles_smooth))
            features.elbow_range = features.max_elbow_angle - features.min_elbow_angle
            
            # Elbow at peak height
            if peak_idx < len(angles_smooth):
                features.elbow_at_peak = float(angles_smooth[peak_idx])
            else:
                features.elbow_at_peak = features.max_elbow_angle
        
        # Fixation analysis (stability at peak)
        features.fixation_quality, features.max_stability_duration, features.stable_at_peak = \
            self._analyze_fixation(heights_smooth, peak_idx)
        
        # Trajectory smoothness (inverse of mean absolute jerk)
        if len(heights_smooth) >= 4:
            jerk = np.diff(heights_smooth, n=3)
            mean_abs_jerk = np.mean(np.abs(jerk)) if len(jerk) > 0 else 1.0
            features.trajectory_smoothness = 1.0 / (1.0 + mean_abs_jerk * 100)
        
        # Ascent/descent ratio
        if peak_idx > 0 and peak_idx < n_frames - 1:
            ascent_time = peak_idx
            descent_time = n_frames - peak_idx
            features.ascent_descent_ratio = ascent_time / descent_time if descent_time > 0 else 1.0
        
        return features
    
    def _analyze_fixation(
        self,
        heights: np.ndarray,
        peak_idx: int
    ) -> Tuple[float, int, bool]:
        """
        Analyze fixation quality at the peak.
        
        Returns:
            (fixation_quality, max_stable_frames, stable_at_peak)
        """
        n_frames = len(heights)
        
        # Calculate velocity
        velocity = np.diff(heights)
        
        # Stability = low velocity
        stability_threshold = 0.01  # Normalized height change per frame
        is_stable = np.abs(velocity) < stability_threshold
        
        # Find longest stable period
        max_stable_frames = 0
        current_stable = 0
        stable_at_peak = False
        
        for i, stable in enumerate(is_stable):
            if stable:
                current_stable += 1
                max_stable_frames = max(max_stable_frames, current_stable)
                
                # Check if peak is within this stable period
                if i >= peak_idx - 2 and i <= peak_idx + 2:
                    stable_at_peak = True
            else:
                current_stable = 0
        
        # Calculate fixation quality
        # Higher score for longer stability near the peak
        peak_window_start = max(0, peak_idx - 3)
        peak_window_end = min(len(is_stable), peak_idx + 3)
        
        if peak_window_end > peak_window_start:
            stability_near_peak = np.mean(is_stable[peak_window_start:peak_window_end])
        else:
            stability_near_peak = 0.0
        
        # Combine factors for quality score
        duration_factor = min(1.0, max_stable_frames / 5.0)  # 5 frames = full score
        location_factor = 1.0 if stable_at_peak else 0.5
        
        fixation_quality = stability_near_peak * duration_factor * location_factor
        
        return float(fixation_quality), max_stable_frames, stable_at_peak


def classify_rep_temporal(
    wrist_heights: List[float],
    elbow_angles: Optional[List[float]] = None,
    fps: float = 15.0,
    lift_type: str = "snatch"
) -> TemporalClassification:
    """
    Convenience function to classify a rep temporally.
    
    Args:
        wrist_heights: Wrist height values over the rep window
        elbow_angles: Optional elbow angle values
        fps: Video frame rate
        lift_type: Type of lift
        
    Returns:
        TemporalClassification result
    """
    classifier = TemporalClassifier(fps=fps, lift_type=lift_type)
    return classifier.classify(wrist_heights, elbow_angles)

