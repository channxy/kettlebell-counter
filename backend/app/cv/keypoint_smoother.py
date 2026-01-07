"""
Temporal keypoint smoothing using Savitzky-Golay and Exponential Moving Average.

IMPROVED SMOOTHING STRATEGY:
1. Savitzky-Golay filter: Primary smoother - preserves peaks/valleys, no phase lag
2. EMA fallback: For short sequences where SG can't be applied
3. Outlier rejection: Detect and handle sudden jumps

Savitzky-Golay is superior to pure EMA for sports movements because:
- Preserves peak heights (critical for lockout detection)
- No phase lag (EMA delays the signal)
- Better for velocity/acceleration calculations
"""

import numpy as np
from scipy.signal import savgol_filter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class SmoothedKeypoint:
    """A temporally smoothed keypoint."""
    x: float
    y: float
    confidence: float
    raw_x: float  # Original unsmoothed
    raw_y: float
    stability: float = 0.0  # How stable the keypoint is (0-1)
    velocity: float = 0.0  # Movement velocity for this keypoint


@dataclass
class KeypointFrame:
    """All keypoints for a single frame."""
    frame_number: int
    timestamp: float
    keypoints: Dict[int, SmoothedKeypoint] = field(default_factory=dict)
    
    def get(self, idx: int) -> Optional[SmoothedKeypoint]:
        return self.keypoints.get(idx)


class KeypointSmoother:
    """
    Advanced keypoint smoother with Savitzky-Golay filtering.
    
    Features:
    - Savitzky-Golay filter for smooth, phase-accurate tracking
    - Per-keypoint smoothing history
    - Confidence-weighted smoothing
    - Stability and velocity scoring
    - Outlier rejection with adaptive thresholds
    """
    
    # MoveNet keypoint indices
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    NUM_KEYPOINTS = 17
    
    # Savitzky-Golay parameters
    SG_WINDOW_SIZE = 7  # Must be odd, 7 frames = ~230ms at 30fps
    SG_POLY_ORDER = 2   # Quadratic fit - good for smooth movements
    
    def __init__(
        self,
        window_size: int = 11,  # Increased for better SG filtering
        alpha: float = 0.3,     # EMA alpha for fallback
        confidence_threshold: float = 0.35,
        outlier_threshold: float = 0.12,  # Tighter outlier detection
        use_savgol: bool = True  # Enable Savitzky-Golay
    ):
        """
        Initialize smoother.
        
        Args:
            window_size: Number of frames for smoothing history (must be >= SG_WINDOW_SIZE)
            alpha: EMA smoothing factor (0 = no smoothing, 1 = no history) - fallback only
            confidence_threshold: Minimum confidence to use keypoint
            outlier_threshold: Max movement per frame to not be outlier
            use_savgol: Whether to use Savitzky-Golay (True) or just EMA (False)
        """
        self.window_size = max(window_size, self.SG_WINDOW_SIZE)
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold
        self.outlier_threshold = outlier_threshold
        self.use_savgol = use_savgol
        
        # Per-keypoint history: keypoint_idx -> deque of (x, y, conf, timestamp)
        self.history: Dict[int, Deque[Tuple[float, float, float, float]]] = {
            i: deque(maxlen=window_size) for i in range(self.NUM_KEYPOINTS)
        }
        
        # Current smoothed values (for EMA fallback)
        self.smoothed: Dict[int, Tuple[float, float, float]] = {}
        
        # Stability tracking (variance of recent positions)
        self.stability: Dict[int, float] = {i: 0.0 for i in range(self.NUM_KEYPOINTS)}
        
        # Velocity tracking
        self.velocity: Dict[int, float] = {i: 0.0 for i in range(self.NUM_KEYPOINTS)}
        
        # Adaptive outlier thresholds per keypoint
        self._outlier_thresholds: Dict[int, float] = {
            i: outlier_threshold for i in range(self.NUM_KEYPOINTS)
        }
        
    def process(
        self,
        keypoints: np.ndarray,
        frame_number: int,
        timestamp: float
    ) -> KeypointFrame:
        """
        Process raw keypoints and return smoothed frame.
        
        Args:
            keypoints: Array of shape (17, 3) with (y, x, confidence) per keypoint
            frame_number: Current frame index
            timestamp: Current timestamp in seconds
            
        Returns:
            KeypointFrame with smoothed keypoints
        """
        result = KeypointFrame(
            frame_number=frame_number,
            timestamp=timestamp
        )
        
        for i in range(self.NUM_KEYPOINTS):
            y, x, conf = keypoints[i]
            
            # Skip low confidence keypoints
            if conf < self.confidence_threshold:
                # Use last known good value if available
                if i in self.smoothed:
                    prev_x, prev_y, prev_conf = self.smoothed[i]
                    result.keypoints[i] = SmoothedKeypoint(
                        x=prev_x,
                        y=prev_y,
                        confidence=prev_conf * 0.9,  # Decay confidence
                        raw_x=x,
                        raw_y=y,
                        stability=self.stability[i],
                        velocity=self.velocity[i]
                    )
                continue
            
            # Check for outliers (sudden jumps)
            is_outlier = False
            if i in self.smoothed:
                prev_x, prev_y, _ = self.smoothed[i]
                movement = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                
                # Use adaptive threshold
                threshold = self._outlier_thresholds[i]
                if movement > threshold:
                    is_outlier = True
                    logger.debug(f"Outlier detected: keypoint {i}, movement {movement:.4f} > {threshold:.4f}")
                
                # Update adaptive threshold (slowly adapt to movement patterns)
                self._outlier_thresholds[i] = threshold * 0.95 + movement * 0.05
            
            if is_outlier:
                # Blend with previous (don't fully trust outlier)
                prev_x, prev_y, prev_conf = self.smoothed[i]
                x = prev_x * 0.7 + x * 0.3
                y = prev_y * 0.7 + y * 0.3
            
            # Add to history
            self.history[i].append((x, y, conf, timestamp))
            
            # Apply smoothing
            if self.use_savgol and len(self.history[i]) >= self.SG_WINDOW_SIZE:
                smooth_x, smooth_y = self._apply_savgol(i)
                smooth_conf = conf
            else:
                # Fallback to EMA for short sequences
                smooth_x, smooth_y, smooth_conf = self._apply_ema(i, x, y, conf)
            
            # Update smoothed value
            self.smoothed[i] = (smooth_x, smooth_y, smooth_conf)
            
            # Calculate velocity and stability
            self._update_velocity(i, timestamp)
            self._update_stability(i)
            
            result.keypoints[i] = SmoothedKeypoint(
                x=smooth_x,
                y=smooth_y,
                confidence=smooth_conf,
                raw_x=keypoints[i][1],
                raw_y=keypoints[i][0],
                stability=self.stability[i],
                velocity=self.velocity[i]
            )
        
        return result
    
    def _apply_savgol(self, keypoint_idx: int) -> Tuple[float, float]:
        """Apply Savitzky-Golay filter to keypoint history."""
        history = list(self.history[keypoint_idx])
        
        if len(history) < self.SG_WINDOW_SIZE:
            # Not enough data for SG filter
            return history[-1][0], history[-1][1]
        
        # Extract x and y sequences
        xs = np.array([h[0] for h in history])
        ys = np.array([h[1] for h in history])
        
        # Apply Savitzky-Golay filter
        # Window size must be odd and <= len(data)
        window = min(self.SG_WINDOW_SIZE, len(xs))
        if window % 2 == 0:
            window -= 1
        
        if window >= 3:  # Minimum window for SG
            try:
                xs_smooth = savgol_filter(xs, window, self.SG_POLY_ORDER)
                ys_smooth = savgol_filter(ys, window, self.SG_POLY_ORDER)
                return float(xs_smooth[-1]), float(ys_smooth[-1])
            except Exception as e:
                logger.warning(f"Savgol filter failed: {e}")
        
        # Fallback to latest value
        return history[-1][0], history[-1][1]
    
    def _apply_ema(
        self, 
        keypoint_idx: int, 
        x: float, 
        y: float, 
        conf: float
    ) -> Tuple[float, float, float]:
        """Apply EMA smoothing (fallback when SG can't be used)."""
        if keypoint_idx in self.smoothed:
            prev_x, prev_y, prev_conf = self.smoothed[keypoint_idx]
            # Confidence-weighted alpha
            effective_alpha = self.alpha * (conf / (conf + prev_conf + 0.001))
            smooth_x = prev_x * (1 - effective_alpha) + x * effective_alpha
            smooth_y = prev_y * (1 - effective_alpha) + y * effective_alpha
            smooth_conf = max(conf, prev_conf * 0.95)
        else:
            smooth_x, smooth_y, smooth_conf = x, y, conf
        
        return smooth_x, smooth_y, smooth_conf
    
    def _update_velocity(self, keypoint_idx: int, current_timestamp: float):
        """Calculate keypoint velocity."""
        history = self.history[keypoint_idx]
        if len(history) < 2:
            self.velocity[keypoint_idx] = 0.0
            return
        
        # Use last two points
        x1, y1, _, t1 = history[-2]
        x2, y2, _, t2 = history[-1]
        
        dt = t2 - t1
        if dt > 0:
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            self.velocity[keypoint_idx] = distance / dt
        else:
            self.velocity[keypoint_idx] = 0.0
    
    def _update_stability(self, keypoint_idx: int):
        """Update stability score for a keypoint."""
        history = self.history[keypoint_idx]
        if len(history) < 3:
            self.stability[keypoint_idx] = 0.5
            return
        
        # Calculate variance of recent positions
        xs = [h[0] for h in history]
        ys = [h[1] for h in history]
        
        variance = np.var(xs) + np.var(ys)
        
        # Convert to stability score (lower variance = higher stability)
        self.stability[keypoint_idx] = 1.0 / (1.0 + variance * 100)
    
    def get_wrist_velocity(self, side: str = "left") -> float:
        """Get recent wrist velocity for fixation detection."""
        idx = self.LEFT_WRIST if side == "left" else self.RIGHT_WRIST
        return self.velocity.get(idx, 0.0)
    
    def get_wrist_stability(self, side: str = "left") -> float:
        """Get wrist stability score (for fixation)."""
        idx = self.LEFT_WRIST if side == "left" else self.RIGHT_WRIST
        return self.stability.get(idx, 0.0)
    
    def get_smoothed_trajectory(
        self, 
        keypoint_idx: int, 
        last_n_frames: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get smoothed trajectory for a keypoint.
        
        Returns:
            Tuple of (x_positions, y_positions, timestamps)
        """
        history = list(self.history[keypoint_idx])
        if last_n_frames:
            history = history[-last_n_frames:]
        
        if len(history) < self.SG_WINDOW_SIZE:
            xs = np.array([h[0] for h in history])
            ys = np.array([h[1] for h in history])
            ts = np.array([h[3] for h in history])
            return xs, ys, ts
        
        xs = np.array([h[0] for h in history])
        ys = np.array([h[1] for h in history])
        ts = np.array([h[3] for h in history])
        
        # Apply SG filter to full trajectory
        window = min(self.SG_WINDOW_SIZE, len(xs))
        if window % 2 == 0:
            window -= 1
        
        if window >= 3:
            xs = savgol_filter(xs, window, self.SG_POLY_ORDER)
            ys = savgol_filter(ys, window, self.SG_POLY_ORDER)
        
        return xs, ys, ts
    
    def reset(self):
        """Reset all smoothing history."""
        for i in range(self.NUM_KEYPOINTS):
            self.history[i].clear()
        self.smoothed.clear()
        self.stability = {i: 0.0 for i in range(self.NUM_KEYPOINTS)}
        self.velocity = {i: 0.0 for i in range(self.NUM_KEYPOINTS)}
        self._outlier_thresholds = {
            i: self.outlier_threshold for i in range(self.NUM_KEYPOINTS)
        }
