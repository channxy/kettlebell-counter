"""
Auto-detection of kettlebell lift types based on movement patterns.

KETTLEBELL SPORT DISCIPLINES:

1. SNATCH (Single KB):
   - ONE kettlebell, one arm at a time
   - Only ONE wrist makes big vertical movements
   - Other arm is for balance (moves independently, smaller range)

2. JERK / Short Cycle (Double KB):
   - TWO kettlebells, both arms move TOGETHER (synchronized)
   - Starts at rack position, never goes to low position between reps

3. LONG CYCLE / Clean & Jerk (Double KB):
   - TWO kettlebells, both arms move TOGETHER (synchronized)
   - Goes to low position for the clean phase

KEY INSIGHT: The main differentiator between single and double KB lifts is
whether the wrists are SYNCHRONIZED (moving together) or not.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
import logging

try:
    from app.cv.hybrid_estimator import HybridPose as PoseKeypoints
except ImportError:
    try:
        from app.cv.movenet_estimator import MoveNetPose as PoseKeypoints
    except ImportError:
        from app.cv.pose_estimator import PoseKeypoints

logger = logging.getLogger(__name__)


@dataclass
class LiftClassification:
    """Result of lift type classification."""
    lift_type: str  # "snatch", "jerk", "long_cycle", "unknown"
    confidence: float
    reasoning: str
    single_arm: bool = False
    dominant_hand: Optional[str] = None


class LiftClassifier:
    """
    Classifies kettlebell lift type from pose data.
    
    Detection Strategy:
    1. Calculate CORRELATION between left and right wrist movements
       - High correlation (>0.7) = Double KB (both move together)
       - Low correlation (<0.5) = Single KB (one moves, other doesn't)
    
    2. For double KB, check if wrists go to LOW position:
       - Significant low time = Long Cycle (has clean phase)
       - Minimal low time = Jerk (stays at rack/overhead)
    """
    
    ANALYSIS_FRAMES = 180  # ~6 seconds at 30fps
    
    # Height thresholds
    LOW_THRESHOLD = 0.40      # Below this = backswing/swing
    RACK_LOW = 0.40
    RACK_HIGH = 0.58
    OVERHEAD_THRESHOLD = 0.68
    
    # Correlation threshold for single vs double
    SYNC_THRESHOLD = 0.6  # Above this = arms synchronized (double KB)
    
    def __init__(self):
        self.poses: List[PoseKeypoints] = []
        self.left_heights: List[float] = []
        self.right_heights: List[float] = []
        
    def add_pose(self, pose: PoseKeypoints):
        """Add a pose observation."""
        if len(self.poses) >= self.ANALYSIS_FRAMES:
            return
            
        self.poses.append(pose)
        
        left_h = pose.get_wrist_height_ratio("left")
        right_h = pose.get_wrist_height_ratio("right")
        
        if left_h is not None:
            self.left_heights.append(left_h)
        if right_h is not None:
            self.right_heights.append(right_h)
    
    def has_enough_data(self) -> bool:
        """Check if we have enough data to classify."""
        return len(self.left_heights) >= 30 and len(self.right_heights) >= 30
    
    def classify(self) -> LiftClassification:
        """Classify the lift type based on collected poses."""
        if not self.has_enough_data():
            return LiftClassification(
                lift_type="unknown",
                confidence=0.0,
                reasoning="Insufficient pose data"
            )
        
        # Make arrays same length
        min_len = min(len(self.left_heights), len(self.right_heights))
        left_arr = np.array(self.left_heights[:min_len])
        right_arr = np.array(self.right_heights[:min_len])
        
        # === 1. Calculate Movement Range for Each Arm ===
        left_range = np.max(left_arr) - np.min(left_arr)
        right_range = np.max(right_arr) - np.min(right_arr)
        
        logger.info(f"Left range: {left_range:.3f}, Right range: {right_range:.3f}")
        
        # === 2. Calculate Correlation Between Arms ===
        # High correlation = arms move together (double KB)
        # Low correlation = one arm active, other passive (single KB)
        if np.std(left_arr) > 0.01 and np.std(right_arr) > 0.01:
            correlation = np.corrcoef(left_arr, right_arr)[0, 1]
        else:
            correlation = 0.0
        
        logger.info(f"Wrist correlation: {correlation:.3f}")
        
        # === 3. Detect Which Arm is Dominant (if single KB) ===
        # Calculate movement velocity (frame-to-frame changes)
        left_velocity = np.abs(np.diff(left_arr))
        right_velocity = np.abs(np.diff(right_arr))
        
        left_total_movement = np.sum(left_velocity)
        right_total_movement = np.sum(right_velocity)
        
        dominant_hand = "left" if left_total_movement > right_total_movement else "right"
        movement_ratio = max(left_total_movement, right_total_movement) / (min(left_total_movement, right_total_movement) + 0.001)
        
        logger.info(f"Left movement: {left_total_movement:.3f}, Right: {right_total_movement:.3f}, Ratio: {movement_ratio:.2f}")
        
        # === 4. Analyze Height Distribution ===
        # Use dominant arm for single KB analysis
        if left_total_movement > right_total_movement:
            active_heights = left_arr
        else:
            active_heights = right_arr
        
        # For double KB, use average
        avg_heights = (left_arr + right_arr) / 2
        
        # Count frames in each zone
        def count_zones(heights):
            low_frames = np.sum(heights < self.LOW_THRESHOLD)
            rack_frames = np.sum((heights >= self.RACK_LOW) & (heights <= self.RACK_HIGH))
            overhead_frames = np.sum(heights > self.OVERHEAD_THRESHOLD)
            return low_frames, rack_frames, overhead_frames
        
        active_low, active_rack, active_overhead = count_zones(active_heights)
        avg_low, avg_rack, avg_overhead = count_zones(avg_heights)
        
        total = len(active_heights)
        active_low_pct = active_low / total
        avg_low_pct = avg_low / total
        
        logger.info(f"Active arm - Low: {active_low_pct:.1%}, Overhead: {active_overhead/total:.1%}")
        logger.info(f"Average - Low: {avg_low_pct:.1%}")
        
        # === 5. Classification Decision ===
        
        # SINGLE KB (Snatch) Detection:
        # - Low correlation between arms (one moves, other doesn't)
        # - OR one arm has much more movement than the other
        is_single_kb = (correlation < self.SYNC_THRESHOLD) or (movement_ratio > 2.0)
        
        if is_single_kb:
            # It's a SNATCH
            confidence = 0.9 if correlation < 0.4 else 0.75
            return LiftClassification(
                lift_type="snatch",
                confidence=confidence,
                reasoning=f"Single KB detected. Correlation: {correlation:.2f} (threshold: {self.SYNC_THRESHOLD}). "
                         f"Movement ratio: {movement_ratio:.1f}x. Dominant: {dominant_hand}",
                single_arm=True,
                dominant_hand=dominant_hand
            )
        
        # DOUBLE KB - Now determine Jerk vs Long Cycle
        # Long Cycle goes to LOW position for the clean phase
        # Jerk stays at rack/overhead (never goes low)
        
        if avg_low_pct > 0.15:
            # Significant time in LOW position = Long Cycle
            return LiftClassification(
                lift_type="long_cycle",
                confidence=0.85,
                reasoning=f"Double KB with clean phase. Correlation: {correlation:.2f}. "
                         f"Low position: {avg_low_pct:.1%} (>15% = has swing/clean)",
                single_arm=False
            )
        else:
            # Minimal low time = Jerk
            return LiftClassification(
                lift_type="jerk",
                confidence=0.85,
                reasoning=f"Double KB jerk pattern. Correlation: {correlation:.2f}. "
                         f"Low position: {avg_low_pct:.1%} (<15% = no clean phase)",
                single_arm=False
            )


def auto_detect_lift_type(poses: List[PoseKeypoints]) -> LiftClassification:
    """Convenience function to classify lift type."""
    classifier = LiftClassifier()
    for pose in poses[:LiftClassifier.ANALYSIS_FRAMES]:
        classifier.add_pose(pose)
    return classifier.classify()
