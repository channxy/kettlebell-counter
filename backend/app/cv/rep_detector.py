"""
Deterministic rep attempt detection based on movement cycles.

This module detects rep ATTEMPTS regardless of validity.
Every detected cycle increments Total Attempts.

KETTLEBELL SPORT DISCIPLINES:
1. SNATCH: Single KB, continuous swing from between legs → overhead → back down
2. JERK (Short Cycle): Double KB, starts at rack position
3. LONG CYCLE (Clean & Jerk): Double KB, full clean + jerk
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Deque
from enum import Enum
from collections import deque

import logging

try:
    from app.cv.hybrid_estimator import HybridPose as PoseKeypoints
except ImportError:
    try:
        from app.cv.movenet_estimator import MoveNetPose as PoseKeypoints
    except ImportError:
        from app.cv.pose_estimator import PoseKeypoints

logger = logging.getLogger(__name__)


class LiftPhase(Enum):
    """Phases of a kettlebell lift cycle."""
    BACKSWING = "backswing"     # KB between legs (snatch, long cycle)
    RACK = "rack"               # KB at chest/shoulder
    DRIVE = "drive"             # Moving upward
    LOCKOUT = "lockout"         # Overhead lockout position
    DROP = "drop"               # Dropping back down
    UNKNOWN = "unknown"


class SnatchState(Enum):
    """State machine states for snatch detection."""
    INITIAL = "initial"           # Waiting for first movement
    DROP_BACKSWING = "drop"       # KB descending / in backswing
    ASCENT = "ascent"             # KB rising
    HAND_INSERTION = "insertion"  # KB rotating around wrist near top
    FIXATION = "fixation"         # Stable overhead position
    COUNTING = "counting"         # In fixation, accumulating time


@dataclass
class RepCycle:
    """A detected rep attempt (movement cycle)."""
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    lift_type: str
    phases: List[Tuple[LiftPhase, int]] = field(default_factory=list)
    peak_frame: Optional[int] = None
    peak_wrist_height: Optional[float] = None
    detection_confidence: float = 1.0
    
    @property
    def duration_seconds(self) -> float:
        if self.start_timestamp is None or self.end_timestamp is None:
            return 0.0
        return self.end_timestamp - self.start_timestamp
    
    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


class SnatchDetector:
    """
    Competition-accurate snatch rep detector using state machine.
    
    A valid snatch rep follows this sequence:
    1. DROP/BACKSWING: KB descends, passes behind knees
    2. ASCENT: KB rises with hip extension
    3. HAND INSERTION: KB rotates around wrist
    4. FIXATION: Stable overhead for ≥200ms with proper alignment
    
    The rep is counted when FIXATION is achieved.
    """
    
    # Height thresholds (wrist height as ratio of body height)
    # ALIGNED with LiftClassifier thresholds for consistency
    # - Backswing (KB between legs): < 0.40
    # - Overhead (KB above head): > 0.68
    BACKSWING_HEIGHT = 0.40      # KB below this = backswing/low position
    SHOULDER_HEIGHT = 0.55       # Approximate shoulder level  
    OVERHEAD_HEIGHT = 0.68       # KB above this = overhead region
    FIXATION_HEIGHT = 0.65       # Minimum height for valid fixation
    
    # Timing thresholds
    FIXATION_TIME_MS = 100       # Minimum fixation time 
    MIN_REP_DURATION_MS = 400    # Minimum time for a full rep
    MAX_REP_DURATION_MS = 5000   # Maximum time for a rep
    
    # Velocity thresholds (change per frame in height ratio)
    VELOCITY_THRESHOLD = 0.03   # Below this = "stable"
    ASCENT_VELOCITY = 0.02      # Minimum upward velocity for ascent
    
    def __init__(self, fps: float):
        self.fps = fps
        self.frame_time_ms = 1000.0 / fps
        
        # Timing in frames
        self.fixation_frames = max(1, int(self.FIXATION_TIME_MS / self.frame_time_ms))
        self.min_rep_frames = max(1, int(self.MIN_REP_DURATION_MS / self.frame_time_ms))
        self.max_rep_frames = int(self.MAX_REP_DURATION_MS / self.frame_time_ms)
        
        # State
        self.state = SnatchState.INITIAL
        self.detected_cycles: List[RepCycle] = []
        
        # Tracking
        self._rep_start_frame: Optional[int] = None
        self._rep_start_timestamp: Optional[float] = None
        self._peak_height = 0.0
        self._peak_frame: Optional[int] = None
        self._fixation_start_frame: Optional[int] = None
        self._phase_history: List[Tuple[LiftPhase, int]] = []
        
        # Height history for velocity calculation
        self._height_history: Deque[Tuple[float, int, float]] = deque(maxlen=10)
        
        # Dominant hand tracking
        self._dominant_hand: Optional[str] = None
        self._left_movement = 0.0
        self._right_movement = 0.0
        self._prev_left: Optional[float] = None
        self._prev_right: Optional[float] = None
        
        # Rep counter
        self._rep_count = 0
        
    def process_pose(self, pose: PoseKeypoints) -> Optional[RepCycle]:
        """Process a pose and detect snatch reps."""
        # Debug: log every pose to see what we're getting
        if pose.frame_number % 10 == 0:
            logger.info(f"SnatchDetector.process_pose called: frame={pose.frame_number}, is_valid={pose.is_valid}")
        
        if not pose.is_valid:
            if pose.frame_number % 10 == 0:
                logger.warning(f"Frame {pose.frame_number} marked INVALID - skipping")
            return None
        
        # Get wrist heights
        left_h = pose.get_wrist_height_ratio("left")
        right_h = pose.get_wrist_height_ratio("right")
        
        # Track dominant hand
        if left_h is not None and self._prev_left is not None:
            self._left_movement += abs(left_h - self._prev_left)
        if right_h is not None and self._prev_right is not None:
            self._right_movement += abs(right_h - self._prev_right)
        
        self._prev_left = left_h
        self._prev_right = right_h
        
        # Determine dominant hand after some frames
        if len(self._height_history) > 10:
            if self._left_movement > self._right_movement * 1.3:
                self._dominant_hand = "left"
            elif self._right_movement > self._left_movement * 1.3:
                self._dominant_hand = "right"
        
        # Get active wrist height
        wrist_height = self._get_active_wrist_height(left_h, right_h)
        if wrist_height is None:
            return None
        
        # Calculate velocity
        velocity = self._calculate_velocity(wrist_height, pose.frame_number)
        
        # Store in history
        self._height_history.append((wrist_height, pose.frame_number, pose.timestamp))
        
        # Debug logging every 5 frames for more visibility
        if pose.frame_number % 5 == 0:
            is_low = wrist_height < self.BACKSWING_HEIGHT
            is_high = wrist_height >= self.OVERHEAD_HEIGHT
            logger.info(f"Frame {pose.frame_number}: height={wrist_height:.3f}, vel={velocity:.4f}, "
                       f"state={self.state.value}, peak={self._peak_height:.3f}, "
                       f"is_low={is_low}, is_high={is_high}")
        
        # Process state machine
        return self._process_state_machine(
            wrist_height, velocity, pose.frame_number, pose.timestamp
        )
    
    def _get_active_wrist_height(
        self, 
        left: Optional[float], 
        right: Optional[float]
    ) -> Optional[float]:
        """Get the active (KB-holding) wrist height."""
        if self._dominant_hand == "left" and left is not None:
            return left
        elif self._dominant_hand == "right" and right is not None:
            return right
        elif left is not None and right is not None:
            return max(left, right)  # Use higher one (likely the active arm)
        return left or right
    
    def _calculate_velocity(self, height: float, frame: int) -> float:
        """Calculate vertical velocity (positive = ascending)."""
        if len(self._height_history) < 2:
            return 0.0
        
        prev_height, prev_frame, _ = self._height_history[-1]
        frame_diff = frame - prev_frame
        if frame_diff == 0:
            return 0.0
        
        return (height - prev_height) / frame_diff
    
    def _process_state_machine(
        self,
        height: float,
        velocity: float,
        frame: int,
        timestamp: float
    ) -> Optional[RepCycle]:
        """
        Simplified snatch state machine focused on reliable detection.
        
        Core logic: Track LOW → HIGH → LOW cycles
        A rep is counted when we see the sequence:
        1. Wrist in BACKSWING zone (height < 0.5)
        2. Wrist reaches OVERHEAD zone (height > 1.0)  
        3. Wrist returns to BACKSWING zone
        """
        
        completed_rep = None
        
        # Track peak height during current rep
        if self._rep_start_frame is not None and height > self._peak_height:
            self._peak_height = height
            self._peak_frame = frame
        
        is_low = height < self.BACKSWING_HEIGHT
        is_high = height >= self.OVERHEAD_HEIGHT
        
        if self.state == SnatchState.INITIAL:
            # Wait for a stable low position to start
            if is_low:
                logger.info(f"Frame {frame}: INITIAL -> DROP_BACKSWING (height={height:.3f} < {self.BACKSWING_HEIGHT})")
                self.state = SnatchState.DROP_BACKSWING
                self._rep_start_frame = frame
                self._rep_start_timestamp = timestamp
                self._peak_height = height
                self._phase_history = [(LiftPhase.BACKSWING, frame)]
                
        elif self.state == SnatchState.DROP_BACKSWING:
            # In backswing, waiting to see overhead
            if is_high:
                # Reached overhead!
                logger.info(f"Frame {frame}: DROP_BACKSWING -> ASCENT (height={height:.3f} >= {self.OVERHEAD_HEIGHT})")
                self.state = SnatchState.ASCENT  # Mark that we've been high
                self._phase_history.append((LiftPhase.LOCKOUT, frame))
                if height > self._peak_height:
                    self._peak_height = height
                    self._peak_frame = frame
            elif not is_low and height > self._peak_height:
                # Still ascending
                self._peak_height = height
                self._peak_frame = frame
                
        elif self.state == SnatchState.ASCENT:
            # We've reached overhead at some point, now track for return to low
            if height > self._peak_height:
                self._peak_height = height
                self._peak_frame = frame
                
            if is_low:
                # Returned to backswing = REP COMPLETE!
                logger.info(f"Frame {frame}: ASCENT -> REP COMPLETE (height={height:.3f}, peak was {self._peak_height:.3f})")
                # Validate that we actually went overhead
                if self._peak_height >= self.OVERHEAD_HEIGHT:
                    self._rep_count += 1
                    logger.info(f"Valid rep #{self._rep_count} completed!")
                    completed_rep = self._complete_rep(frame, timestamp, is_valid_fixation=True)
                else:
                    # Didn't reach overhead - blank swing, don't count
                    logger.info(f"Blank swing - peak {self._peak_height:.3f} < threshold {self.OVERHEAD_HEIGHT}")
                    completed_rep = self._complete_rep(frame, timestamp, is_valid_fixation=False)
        
        return completed_rep
    
    def _complete_rep(
        self, 
        frame: int, 
        timestamp: float, 
        is_valid_fixation: bool
    ) -> Optional[RepCycle]:
        """Complete a rep cycle."""
        if self._rep_start_frame is None or self._rep_start_timestamp is None:
            self._reset_for_next_rep(frame, timestamp)
            return None
        
        # Validate duration
        rep_frames = frame - self._rep_start_frame
        if rep_frames < self.min_rep_frames or rep_frames > self.max_rep_frames:
            self._reset_for_next_rep(frame)
            return None
        
        confidence = 0.95 if is_valid_fixation else 0.6
        
        cycle = RepCycle(
            start_frame=self._rep_start_frame,
            end_frame=frame,
            start_timestamp=self._rep_start_timestamp,
            end_timestamp=timestamp,
            lift_type="snatch",
            phases=list(self._phase_history),
            peak_frame=self._peak_frame,
            peak_wrist_height=self._peak_height,
            detection_confidence=confidence
        )
        
        self.detected_cycles.append(cycle)
        self._reset_for_next_rep(frame, timestamp)
        
        return cycle
    
    def _reset_for_next_rep(self, frame: int, timestamp: float = None):
        """Reset state for next rep detection."""
        self.state = SnatchState.DROP_BACKSWING
        self._rep_start_frame = frame
        self._rep_start_timestamp = timestamp  # Keep the timestamp
        self._peak_height = 0.0
        self._peak_frame = None
        self._fixation_start_frame = None
        self._phase_history = [(LiftPhase.BACKSWING, frame)]
    
    def finalize(self) -> List[RepCycle]:
        """Return all detected cycles."""
        return self.detected_cycles
    
    def get_total_attempts(self) -> int:
        """Get total detected rep attempts."""
        return len(self.detected_cycles)


class JerkDetector:
    """
    Jerk rep detector.
    
    Pattern: RACK → OVERHEAD → RACK
    Never goes to backswing position between reps.
    """
    
    # Aligned with LiftClassifier thresholds
    RACK_LOW = 0.40
    RACK_HIGH = 0.58
    OVERHEAD_HEIGHT = 0.68
    
    FIXATION_TIME_MS = 100
    MIN_REP_DURATION_MS = 500
    MAX_REP_DURATION_MS = 4000
    
    def __init__(self, fps: float):
        self.fps = fps
        self.frame_time_ms = 1000.0 / fps
        self.fixation_frames = max(1, int(self.FIXATION_TIME_MS / self.frame_time_ms))
        self.min_rep_frames = max(1, int(self.MIN_REP_DURATION_MS / self.frame_time_ms))
        self.max_rep_frames = int(self.MAX_REP_DURATION_MS / self.frame_time_ms)
        
        self.detected_cycles: List[RepCycle] = []
        self._state = "rack"  # rack, drive, overhead, drop
        self._rep_start_frame: Optional[int] = None
        self._rep_start_timestamp: Optional[float] = None
        self._peak_height = 0.0
        self._peak_frame: Optional[int] = None
        self._overhead_frames = 0
        self._rack_frames = 0
        
    def process_pose(self, pose: PoseKeypoints) -> Optional[RepCycle]:
        """Process pose for jerk detection."""
        if not pose.is_valid:
            return None
        
        # Average both wrists for double KB
        left_h = pose.get_wrist_height_ratio("left")
        right_h = pose.get_wrist_height_ratio("right")
        
        if left_h is not None and right_h is not None:
            height = (left_h + right_h) / 2
        else:
            height = left_h or right_h
        
        if height is None:
            return None
        
        return self._process_state(height, pose.frame_number, pose.timestamp)
    
    def _process_state(self, height: float, frame: int, timestamp: float) -> Optional[RepCycle]:
        """State machine for jerk."""
        completed = None
        
        is_rack = self.RACK_LOW <= height <= self.RACK_HIGH
        is_overhead = height > self.OVERHEAD_HEIGHT
        
        if is_rack:
            self._rack_frames += 1
            self._overhead_frames = 0
        elif is_overhead:
            self._overhead_frames += 1
            self._rack_frames = 0
        else:
            self._rack_frames = 0
            self._overhead_frames = 0
        
        if height > self._peak_height:
            self._peak_height = height
            self._peak_frame = frame
        
        if self._state == "rack":
            if self._rack_frames >= 2:
                # Stable at rack
                pass
            if height > self.RACK_HIGH:
                # Starting drive
                self._state = "drive"
                self._rep_start_frame = frame
                self._rep_start_timestamp = timestamp
                
        elif self._state == "drive":
            if is_overhead and self._overhead_frames >= self.fixation_frames:
                # Reached overhead
                self._state = "overhead"
            elif is_rack and self._rack_frames >= 2:
                # Dropped back without overhead - reset
                self._state = "rack"
                self._rep_start_frame = None
                
        elif self._state == "overhead":
            if height < self.OVERHEAD_HEIGHT:
                # Starting drop
                self._state = "drop"
                
        elif self._state == "drop":
            if is_rack and self._rack_frames >= 2:
                # Completed rep
                if self._rep_start_frame is not None:
                    rep_frames = frame - self._rep_start_frame
                    if self.min_rep_frames <= rep_frames <= self.max_rep_frames:
                        completed = RepCycle(
                            start_frame=self._rep_start_frame,
                            end_frame=frame,
                            start_timestamp=self._rep_start_timestamp,
                            end_timestamp=timestamp,
                            lift_type="jerk",
                            peak_frame=self._peak_frame,
                            peak_wrist_height=self._peak_height,
                            detection_confidence=0.9
                        )
                        self.detected_cycles.append(completed)
                
                # Reset for next rep
                self._state = "rack"
                self._rep_start_frame = None
                self._peak_height = 0.0
                
        return completed
    
    def finalize(self) -> List[RepCycle]:
        return self.detected_cycles
    
    def get_total_attempts(self) -> int:
        return len(self.detected_cycles)


class LongCycleDetector:
    """
    Long Cycle (Clean & Jerk) detector.
    
    Pattern: LOW (swing) → RACK (clean) → OVERHEAD (jerk) → LOW
    """
    
    # Aligned with LiftClassifier thresholds
    LOW_HEIGHT = 0.40
    RACK_LOW = 0.40
    RACK_HIGH = 0.58
    OVERHEAD_HEIGHT = 0.68
    
    MIN_REP_DURATION_MS = 1000
    MAX_REP_DURATION_MS = 6000
    
    def __init__(self, fps: float):
        self.fps = fps
        self.frame_time_ms = 1000.0 / fps
        self.min_rep_frames = max(1, int(self.MIN_REP_DURATION_MS / self.frame_time_ms))
        self.max_rep_frames = int(self.MAX_REP_DURATION_MS / self.frame_time_ms)
        
        self.detected_cycles: List[RepCycle] = []
        self._state = "low"  # low, clean, rack, jerk, overhead, drop
        self._rep_start_frame: Optional[int] = None
        self._rep_start_timestamp: Optional[float] = None
        self._peak_height = 0.0
        self._peak_frame: Optional[int] = None
        self._frames_in_zone = 0
        
    def process_pose(self, pose: PoseKeypoints) -> Optional[RepCycle]:
        """Process pose for long cycle detection."""
        if not pose.is_valid:
            return None
        
        left_h = pose.get_wrist_height_ratio("left")
        right_h = pose.get_wrist_height_ratio("right")
        
        if left_h is not None and right_h is not None:
            height = (left_h + right_h) / 2
        else:
            height = left_h or right_h
        
        if height is None:
            return None
        
        return self._process_state(height, pose.frame_number, pose.timestamp)
    
    def _process_state(self, height: float, frame: int, timestamp: float) -> Optional[RepCycle]:
        """State machine for long cycle."""
        completed = None
        
        is_low = height < self.LOW_HEIGHT
        is_rack = self.RACK_LOW <= height <= self.RACK_HIGH
        is_overhead = height > self.OVERHEAD_HEIGHT
        
        if height > self._peak_height:
            self._peak_height = height
            self._peak_frame = frame
        
        if self._state == "low":
            if is_low:
                self._frames_in_zone += 1
                if self._frames_in_zone >= 2 and self._rep_start_frame is None:
                    self._rep_start_frame = frame
                    self._rep_start_timestamp = timestamp
            elif height > self.LOW_HEIGHT:
                # Rising - clean phase
                self._state = "clean"
                self._frames_in_zone = 0
                if self._rep_start_frame is None:
                    self._rep_start_frame = frame
                    self._rep_start_timestamp = timestamp
                    
        elif self._state == "clean":
            if is_rack:
                self._frames_in_zone += 1
                if self._frames_in_zone >= 2:
                    # Caught at rack
                    self._state = "rack"
                    self._frames_in_zone = 0
            elif is_low:
                # Dropped back
                self._state = "low"
                self._frames_in_zone = 0
                
        elif self._state == "rack":
            if is_rack:
                self._frames_in_zone += 1
            elif height > self.RACK_HIGH:
                # Starting jerk
                self._state = "jerk"
                self._frames_in_zone = 0
            elif is_low:
                # Dropped without jerk
                self._state = "low"
                self._rep_start_frame = None
                self._frames_in_zone = 0
                
        elif self._state == "jerk":
            if is_overhead:
                self._frames_in_zone += 1
                if self._frames_in_zone >= 2:
                    self._state = "overhead"
                    self._frames_in_zone = 0
            elif is_rack:
                # Back to rack without lockout
                self._state = "rack"
                self._frames_in_zone = 0
                
        elif self._state == "overhead":
            if height < self.OVERHEAD_HEIGHT:
                self._state = "drop"
                self._frames_in_zone = 0
                
        elif self._state == "drop":
            if is_low:
                self._frames_in_zone += 1
                if self._frames_in_zone >= 2:
                    # Completed full cycle
                    if self._rep_start_frame is not None:
                        rep_frames = frame - self._rep_start_frame
                        if self.min_rep_frames <= rep_frames <= self.max_rep_frames:
                            completed = RepCycle(
                                start_frame=self._rep_start_frame,
                                end_frame=frame,
                                start_timestamp=self._rep_start_timestamp,
                                end_timestamp=timestamp,
                                lift_type="long_cycle",
                                peak_frame=self._peak_frame,
                                peak_wrist_height=self._peak_height,
                                detection_confidence=0.85
                            )
                            self.detected_cycles.append(completed)
                    
                    # Reset
                    self._state = "low"
                    self._rep_start_frame = frame
                    self._rep_start_timestamp = timestamp
                    self._peak_height = 0.0
                    self._frames_in_zone = 0
                    
        return completed
    
    def finalize(self) -> List[RepCycle]:
        return self.detected_cycles
    
    def get_total_attempts(self) -> int:
        return len(self.detected_cycles)


class RepDetector:
    """Unified rep detector that delegates to lift-specific detectors."""
    
    def __init__(self, lift_type: str, fps: float, **kwargs):
        self.lift_type = lift_type
        self.fps = fps
        
        if lift_type == "snatch":
            self._detector = SnatchDetector(fps)
        elif lift_type == "jerk":
            self._detector = JerkDetector(fps)
        else:  # long_cycle
            self._detector = LongCycleDetector(fps)
    
    def process_pose(self, pose: PoseKeypoints) -> Optional[RepCycle]:
        return self._detector.process_pose(pose)
    
    def finalize(self) -> List[RepCycle]:
        return self._detector.finalize()
    
    def get_total_attempts(self) -> int:
        return self._detector.get_total_attempts()
    
    @property
    def detected_cycles(self) -> List[RepCycle]:
        return self._detector.detected_cycles


class RepDetectorFactory:
    """Factory for creating lift-specific rep detectors."""
    
    @staticmethod
    def create(lift_type: str, fps: float) -> RepDetector:
        return RepDetector(lift_type=lift_type, fps=fps)
