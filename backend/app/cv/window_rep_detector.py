"""
Window-based rep detector for kettlebell lifts.

Detects rep WINDOWS (start to end) rather than discrete events.
Every detected window = 1 Total Attempt.

REP WINDOW DEFINITION:
- START: Bell leaves rack OR backswing begins
- END: Bell returns to rack OR fixation attempt ends

This is the CORE detection logic - validation happens separately.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Deque, Tuple
from collections import deque
from enum import Enum
import logging

from app.cv.video_analyzer import CameraAngle

# Use HybridPose directly for better accuracy
try:
    from app.cv.hybrid_estimator import HybridPose
except ImportError:
    HybridPose = None

logger = logging.getLogger(__name__)


class RepPhase(Enum):
    """Phases within a rep window."""
    IDLE = "idle"           # Waiting for rep to start
    BACKSWING = "backswing" # KB below hip (snatch/LC)
    ASCENT = "ascent"       # KB rising
    OVERHEAD = "overhead"   # KB at lockout position
    FIXATION = "fixation"   # Stable overhead hold
    DESCENT = "descent"     # KB dropping
    RACK = "rack"           # KB at rack position (jerk/LC)


@dataclass
class RepWindow:
    """A detected rep attempt window."""
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    lift_type: str
    
    # Phase sequence within window
    phases: List[Tuple[RepPhase, int, float]] = field(default_factory=list)
    
    # Metrics collected during window
    peak_wrist_height: float = 0.0
    peak_frame: int = 0
    min_wrist_height: float = 1.0
    fixation_frames: int = 0
    max_elbow_angle: float = 0.0
    avg_confidence: float = 0.0
    
    # Raw data for validation
    wrist_heights: List[float] = field(default_factory=list)
    elbow_angles: List[float] = field(default_factory=list)
    wrist_velocities: List[float] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        return self.end_timestamp - self.start_timestamp
    
    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame
    
    @property
    def reached_overhead(self) -> bool:
        return self.peak_wrist_height > 0.65
    
    @property
    def had_fixation(self) -> bool:
        return self.fixation_frames >= 3


@dataclass
class AngleAdaptiveThresholds:
    """
    Camera angle-specific thresholds - COMPETITION ACCURATE.
    
    Different camera angles require different validation thresholds
    because keypoint visibility and accuracy varies.
    
    TIMING CALIBRATION (for snatch at ~2-3 seconds per rep):
    - Minimum rep: 1.2 seconds (prevents false positives from partial swings)
    - Maximum rep: 6 seconds (allows for slow, controlled reps)
    
    COMPETITION STANDARDS:
    - Elbow must be near fully extended (165-175°)
    - Wrist must be clearly overhead (above head level)
    - Visible fixation pause required
    """
    # Wrist height thresholds (normalized to body height)
    backswing_height: float = 0.35       # Below this = backswing/swing (slightly raised)
    rack_height_low: float = 0.45        # Rack zone lower bound
    rack_height_high: float = 0.60       # Rack zone upper bound
    overhead_height: float = 0.75        # Competition standard: clearly above head
    
    # Elbow angle thresholds - COMPETITION STRICT
    lockout_angle: float = 165.0         # Near full extension required
    
    # Fixation requirements - COMPETITION STRICT
    min_fixation_frames: int = 5         # ~333ms at 15fps - visible pause
    fixation_velocity_threshold: float = 0.008  # Tighter stability requirement
    
    # Timing - CALIBRATED FOR REAL KETTLEBELL REPS
    min_rep_duration_ms: float = 1200.0  # 1.2 seconds minimum
    max_rep_duration_ms: float = 6000.0  # 6 seconds maximum
    
    # Minimum gap between reps (prevent detecting the same cycle twice)
    min_gap_between_reps_ms: float = 400.0
    
    @classmethod
    def for_angle(cls, angle: CameraAngle, fps: float) -> "AngleAdaptiveThresholds":
        """Create thresholds adapted to camera angle."""
        thresholds = cls()
        
        # Convert fixation time to frames
        thresholds.min_fixation_frames = max(3, int(0.5 * fps))  # 500ms minimum
        
        if angle == CameraAngle.SIDE:
            # Side view: elbow angle is reliable
            thresholds.lockout_angle = 165.0
            thresholds.min_fixation_frames = max(3, int(0.4 * fps))  # Less strict
            
        elif angle == CameraAngle.FRONT:
            # Front view: elbow angle unreliable, require longer fixation
            thresholds.lockout_angle = 155.0  # More lenient
            thresholds.min_fixation_frames = max(5, int(0.7 * fps))  # More strict
            
        elif angle == CameraAngle.DIAGONAL:
            # Diagonal: mixed reliability
            thresholds.lockout_angle = 160.0
            thresholds.min_fixation_frames = max(4, int(0.5 * fps))
        
        return thresholds
    
    @classmethod
    def for_lift_type(cls, lift_type: str, angle: CameraAngle, fps: float) -> "AngleAdaptiveThresholds":
        """Create thresholds adapted to lift type and camera angle."""
        thresholds = cls.for_angle(angle, fps)
        
        # Lift-specific timing adjustments
        if lift_type == "snatch":
            # Snatch: typically 2-3 seconds per rep
            thresholds.min_rep_duration_ms = 1500.0
            thresholds.max_rep_duration_ms = 5000.0
        elif lift_type == "jerk":
            # Jerk: faster, typically 1.5-2.5 seconds per rep
            thresholds.min_rep_duration_ms = 1000.0
            thresholds.max_rep_duration_ms = 4000.0
        elif lift_type == "long_cycle":
            # Long cycle: slowest, typically 3-4.5 seconds per rep
            thresholds.min_rep_duration_ms = 2000.0
            thresholds.max_rep_duration_ms = 7000.0
        
        return thresholds


class WindowRepDetector:
    """
    Window-based rep detector.
    
    Detects rep attempt WINDOWS based on wrist movement patterns.
    Does not make validity judgments - just detects cycles.
    
    Uses a simple but robust state machine:
    1. IDLE: Waiting for movement to start
    2. ACTIVE: In a rep attempt (from low/rack → overhead → returning)
    """
    
    def __init__(
        self,
        lift_type: str,
        fps: float,
        camera_angle: CameraAngle = CameraAngle.UNKNOWN,
        dominant_hand: Optional[str] = None
    ):
        self.lift_type = lift_type
        self.fps = fps
        self.camera_angle = camera_angle
        self.dominant_hand = dominant_hand
        
        # Get lift-type and angle-adaptive thresholds
        self.thresholds = AngleAdaptiveThresholds.for_lift_type(lift_type, camera_angle, fps)
        
        # Frame timing
        self.frame_time_ms = 1000.0 / fps
        self.min_rep_frames = max(3, int(self.thresholds.min_rep_duration_ms / self.frame_time_ms))
        self.max_rep_frames = int(self.thresholds.max_rep_duration_ms / self.frame_time_ms)
        
        # State
        self.current_phase = RepPhase.IDLE
        self.detected_windows: List[RepWindow] = []
        
        # Current window tracking
        self._window_start_frame: Optional[int] = None
        self._window_start_timestamp: Optional[float] = None
        self._current_window_data: Dict = {}
        self._phase_history: List[Tuple[RepPhase, int, float]] = []
        
        # Height tracking for rep detection
        self._height_history: Deque[Tuple[float, int, float]] = deque(maxlen=30)
        self._velocity_history: Deque[float] = deque(maxlen=10)
        
        # Fixation tracking
        self._fixation_frame_count = 0
        self._peak_height = 0.0
        self._peak_frame = 0
        
        # Dominant hand detection
        self._left_movement_sum = 0.0
        self._right_movement_sum = 0.0
        self._prev_left_height: Optional[float] = None
        self._prev_right_height: Optional[float] = None
        
        logger.info(f"WindowRepDetector initialized: {lift_type}, {fps} fps, "
                   f"angle={camera_angle.value}, dominant={dominant_hand}")
    
    def process_frame(
        self,
        pose: "HybridPose",
        smoother=None  # Optional, not used anymore
    ) -> Optional[RepWindow]:
        """
        Process a single frame and detect rep windows.
        
        Args:
            pose: HybridPose from the estimator
            smoother: Deprecated, not used
            
        Returns:
            RepWindow if a window just completed, None otherwise
        """
        frame_num = pose.frame_number
        timestamp = pose.timestamp
        
        # Get wrist heights directly from HybridPose
        left_height = pose.get_wrist_height_ratio("left")
        right_height = pose.get_wrist_height_ratio("right")
        
        # Track dominant hand
        self._track_dominant_hand(left_height, right_height)
        
        # Determine which side to track
        wrist_side = self.dominant_hand or ("left" if left_height and (not right_height or left_height > right_height) else "right")
        
        # Get active wrist height based on lift type
        active_height = self._get_active_height(left_height, right_height)
        if active_height is None:
            return None
        
        # Calculate velocity
        velocity = self._calculate_velocity(active_height, frame_num, timestamp)
        
        # Store in history
        self._height_history.append((active_height, frame_num, timestamp))
        self._velocity_history.append(velocity)
        
        # Get elbow angle directly from pose
        elbow_angle = pose.get_elbow_angle(wrist_side)
        
        # Calculate wrist velocity from recent history
        wrist_velocity = self._calculate_wrist_velocity()
        
        # Debug logging
        if frame_num % 5 == 0:
            phase_str = self.current_phase.value
            logger.info(f"Frame {frame_num}: h={active_height:.3f}, v={velocity:.4f}, "
                       f"phase={phase_str}, peak={self._peak_height:.3f}, "
                       f"fix={self._fixation_frame_count}")
        
        # Get confidence from pose
        confidence = pose.overall_confidence if hasattr(pose, 'overall_confidence') else 0.8
        
        # Process state machine
        completed_window = self._process_state(
            height=active_height,
            velocity=velocity,
            wrist_velocity=wrist_velocity,
            elbow_angle=elbow_angle,
            frame=frame_num,
            timestamp=timestamp,
            confidence=confidence
        )
        
        return completed_window
    
    def _track_dominant_hand(
        self,
        left_height: Optional[float],
        right_height: Optional[float]
    ):
        """Track which hand is dominant (for single KB lifts)."""
        if self.dominant_hand is not None:
            return
        
        if left_height is not None and self._prev_left_height is not None:
            self._left_movement_sum += abs(left_height - self._prev_left_height)
        if right_height is not None and self._prev_right_height is not None:
            self._right_movement_sum += abs(right_height - self._prev_right_height)
        
        self._prev_left_height = left_height
        self._prev_right_height = right_height
        
        # Determine after some data
        if len(self._height_history) > 20:
            if self._left_movement_sum > self._right_movement_sum * 1.5:
                self.dominant_hand = "left"
                logger.info(f"Detected dominant hand: left (L={self._left_movement_sum:.2f}, R={self._right_movement_sum:.2f})")
            elif self._right_movement_sum > self._left_movement_sum * 1.5:
                self.dominant_hand = "right"
                logger.info(f"Detected dominant hand: right (L={self._left_movement_sum:.2f}, R={self._right_movement_sum:.2f})")
    
    def _get_active_height(
        self,
        left_height: Optional[float],
        right_height: Optional[float]
    ) -> Optional[float]:
        """Get the active (kettlebell-holding) wrist height."""
        if self.lift_type == "snatch":
            # Single KB - use dominant or higher wrist
            if self.dominant_hand == "left" and left_height is not None:
                return left_height
            elif self.dominant_hand == "right" and right_height is not None:
                return right_height
            elif left_height is not None and right_height is not None:
                return max(left_height, right_height)
            return left_height or right_height
        else:
            # Double KB - average both wrists
            if left_height is not None and right_height is not None:
                return (left_height + right_height) / 2
            return left_height or right_height
    
    def _calculate_velocity(
        self,
        height: float,
        frame: int,
        timestamp: float
    ) -> float:
        """Calculate vertical velocity (positive = ascending)."""
        if len(self._height_history) < 1:
            return 0.0
        
        prev_height, prev_frame, prev_ts = self._height_history[-1]
        dt = timestamp - prev_ts
        if dt <= 0:
            return 0.0
        
        return (height - prev_height) / dt
    
    def _calculate_wrist_velocity(self) -> float:
        """Calculate recent wrist velocity for fixation detection."""
        if len(self._height_history) < 3:
            return 1.0  # High velocity = not stable
        
        # Use last 5 frames
        recent = list(self._height_history)[-5:]
        heights = [h for h, f, t in recent]
        
        # Calculate variance as proxy for velocity
        variance = float(np.var(heights)) if len(heights) > 1 else 0.0
        
        # Convert variance to velocity-like metric
        return float(np.sqrt(variance))
    
    def _process_state(
        self,
        height: float,
        velocity: float,
        wrist_velocity: float,
        elbow_angle: Optional[float],
        frame: int,
        timestamp: float,
        confidence: float
    ) -> Optional[RepWindow]:
        """
        Process state machine for rep window detection.
        
        Simplified robust approach:
        1. IDLE: Wait for height to go LOW (backswing) or high from RACK
        2. ACTIVE: Track until we see LOW again (cycle complete)
        """
        completed_window = None
        
        # Zone detection
        is_low = height < self.thresholds.backswing_height
        is_rack = self.thresholds.rack_height_low <= height <= self.thresholds.rack_height_high
        is_overhead = height >= self.thresholds.overhead_height
        is_stable = wrist_velocity < self.thresholds.fixation_velocity_threshold
        
        # Track peak during active window
        if self._window_start_frame is not None:
            if height > self._peak_height:
                self._peak_height = height
                self._peak_frame = frame
            
            # Track fixation
            if is_overhead and is_stable:
                self._fixation_frame_count += 1
            else:
                self._fixation_frame_count = 0
            
            # Store metrics
            self._current_window_data.setdefault("heights", []).append(height)
            if elbow_angle is not None:
                self._current_window_data.setdefault("elbows", []).append(elbow_angle)
            self._current_window_data.setdefault("velocities", []).append(wrist_velocity)
            self._current_window_data.setdefault("confidences", []).append(confidence)
        
        # State transitions
        if self.current_phase == RepPhase.IDLE:
            # Start rep when we see LOW position (snatch/LC) or leaving RACK (jerk)
            if self.lift_type == "snatch":
                if is_low:
                    self._start_window(frame, timestamp, RepPhase.BACKSWING)
            elif self.lift_type == "jerk":
                if is_rack:
                    self._start_window(frame, timestamp, RepPhase.RACK)
            else:  # long_cycle
                if is_low:
                    self._start_window(frame, timestamp, RepPhase.BACKSWING)
        
        elif self.current_phase == RepPhase.BACKSWING:
            if height > self.thresholds.rack_height_low:
                self.current_phase = RepPhase.ASCENT
                self._phase_history.append((RepPhase.ASCENT, frame, timestamp))
        
        elif self.current_phase == RepPhase.RACK:
            if height > self.thresholds.rack_height_high:
                self.current_phase = RepPhase.ASCENT
                self._phase_history.append((RepPhase.ASCENT, frame, timestamp))
        
        elif self.current_phase == RepPhase.ASCENT:
            if is_overhead:
                self.current_phase = RepPhase.OVERHEAD
                self._phase_history.append((RepPhase.OVERHEAD, frame, timestamp))
            elif is_low and self.lift_type == "snatch":
                # Dropped back without reaching overhead - still counts as attempt
                completed_window = self._complete_window(frame, timestamp)
        
        elif self.current_phase == RepPhase.OVERHEAD:
            if is_stable and self._fixation_frame_count >= self.thresholds.min_fixation_frames:
                self.current_phase = RepPhase.FIXATION
                self._phase_history.append((RepPhase.FIXATION, frame, timestamp))
            elif height < self.thresholds.overhead_height:
                # Started dropping
                self.current_phase = RepPhase.DESCENT
                self._phase_history.append((RepPhase.DESCENT, frame, timestamp))
        
        elif self.current_phase == RepPhase.FIXATION:
            if height < self.thresholds.overhead_height:
                self.current_phase = RepPhase.DESCENT
                self._phase_history.append((RepPhase.DESCENT, frame, timestamp))
        
        elif self.current_phase == RepPhase.DESCENT:
            # Rep complete when we return to start position
            if self.lift_type == "snatch":
                if is_low:
                    completed_window = self._complete_window(frame, timestamp)
            elif self.lift_type == "jerk":
                if is_rack:
                    completed_window = self._complete_window(frame, timestamp)
            else:  # long_cycle
                if is_low:
                    completed_window = self._complete_window(frame, timestamp)
        
        # Timeout check
        if self._window_start_frame is not None:
            window_frames = frame - self._window_start_frame
            if window_frames > self.max_rep_frames:
                logger.warning(f"Frame {frame}: Rep window timeout, resetting")
                self._reset_window()
        
        return completed_window
    
    def _start_window(self, frame: int, timestamp: float, phase: RepPhase):
        """Start a new rep window."""
        self._window_start_frame = frame
        self._window_start_timestamp = timestamp
        self.current_phase = phase
        self._phase_history = [(phase, frame, timestamp)]
        self._peak_height = 0.0
        self._peak_frame = frame
        self._fixation_frame_count = 0
        self._current_window_data = {}
        
        logger.info(f"Frame {frame}: Started rep window ({phase.value})")
    
    def _complete_window(self, frame: int, timestamp: float) -> Optional[RepWindow]:
        """Complete the current rep window."""
        if self._window_start_frame is None:
            self._reset_window()
            return None
        
        # Check minimum duration
        window_frames = frame - self._window_start_frame
        if window_frames < self.min_rep_frames:
            logger.info(f"Frame {frame}: Window too short ({window_frames} < {self.min_rep_frames}), discarding")
            self._reset_window()
            return None
        
        # Check minimum gap from previous rep (prevent double-counting)
        min_gap_frames = int(self.thresholds.min_gap_between_reps_ms / self.frame_time_ms)
        if self.detected_windows:
            last_window = self.detected_windows[-1]
            gap_frames = self._window_start_frame - last_window.end_frame
            if gap_frames < min_gap_frames:
                logger.info(f"Frame {frame}: Window too close to previous "
                           f"(gap={gap_frames} < {min_gap_frames}), discarding")
                self._reset_window()
                return None
        
        # Check if this window reached a meaningful peak height
        if self._peak_height < self.thresholds.overhead_height * 0.8:
            logger.info(f"Frame {frame}: Window peak too low "
                       f"({self._peak_height:.3f} < {self.thresholds.overhead_height * 0.8:.3f}), discarding")
            self._reset_window()
            return None
        
        # Create window object
        heights = self._current_window_data.get("heights", [])
        elbows = self._current_window_data.get("elbows", [])
        velocities = self._current_window_data.get("velocities", [])
        confidences = self._current_window_data.get("confidences", [])
        
        window = RepWindow(
            start_frame=self._window_start_frame,
            end_frame=frame,
            start_timestamp=self._window_start_timestamp,
            end_timestamp=timestamp,
            lift_type=self.lift_type,
            phases=list(self._phase_history),
            peak_wrist_height=self._peak_height,
            peak_frame=self._peak_frame,
            min_wrist_height=min(heights) if heights else 0.0,
            fixation_frames=self._fixation_frame_count,
            max_elbow_angle=max(elbows) if elbows else 0.0,
            avg_confidence=float(np.mean(confidences)) if confidences else 0.0,
            wrist_heights=heights,
            elbow_angles=elbows,
            wrist_velocities=velocities
        )
        
        self.detected_windows.append(window)
        
        logger.info(f"Frame {frame}: Completed rep #{len(self.detected_windows)} "
                   f"(peak={self._peak_height:.3f}, fix={self._fixation_frame_count}, "
                   f"duration={window.duration_seconds:.2f}s)")
        
        self._reset_window()
        return window
    
    def _reset_window(self):
        """Reset window tracking for next rep."""
        self._window_start_frame = None
        self._window_start_timestamp = None
        self.current_phase = RepPhase.IDLE
        self._phase_history = []
        self._peak_height = 0.0
        self._peak_frame = 0
        self._fixation_frame_count = 0
        self._current_window_data = {}
    
    def finalize(self) -> List[RepWindow]:
        """Finalize detection and return all windows."""
        # Complete any in-progress window
        if self._window_start_frame is not None and len(self._height_history) > 0:
            last_height, last_frame, last_ts = self._height_history[-1]
            # Only count if it reached overhead
            if self._peak_height >= self.thresholds.overhead_height:
                self._complete_window(last_frame, last_ts)
        
        return self.detected_windows
    
    def get_total_attempts(self) -> int:
        """Get total detected rep attempts."""
        return len(self.detected_windows)

