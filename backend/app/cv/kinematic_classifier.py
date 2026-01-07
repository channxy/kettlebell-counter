"""
Athletic Movement Classifier (Weightlifting)

A Computer Vision Logic Engine for Kinematic Analysis of Olympic Weightlifting
and Kettlebell Sport movements.

OBJECTIVE: Analyze skeletal coordinates + weight position to identify and 
validate one (1) repetition of specific movements.

DATA INPUT CONTEXT (30fps video):
- V_y: Vertical velocity of the weight (positive = ascending)
- Y_weight: Vertical coordinate of the weight (normalized 0-1, 0=top)
- θ_elbow: Elbow joint angle in degrees
- S_state: Current movement phase

SUPPORTED LIFTS:
1. Snatch: Continuous positive V_y with terminal overhead lockout
2. Jerk: Discontinuous V_y from mid-level stationary point
3. Long Cycle: Compound state machine [Clean → Rack → Jerk → Drop]
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Deque, Tuple, Any
from collections import deque
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: Core Data Structures
# =============================================================================

class LiftType(Enum):
    """Supported lift types."""
    SNATCH = "snatch"
    JERK = "jerk"
    LONG_CYCLE = "long_cycle"
    UNKNOWN = "unknown"


class RepValidity(Enum):
    """Rep validation result."""
    VALID = "valid"
    NO_LIFT = "no_lift"
    AMBIGUOUS = "ambiguous"


@dataclass
class KinematicFrame:
    """
    Single frame of kinematic data.
    
    All measurements are normalized to body height where appropriate.
    """
    frame_number: int
    timestamp: float
    fps: float = 30.0
    
    # Weight position (normalized Y, 0 = top of frame, 1 = bottom)
    y_weight: float = 0.5
    
    # Vertical velocity (positive = ascending, units: normalized height / second)
    v_y: float = 0.0
    
    # Joint angles (degrees)
    elbow_angle_left: float = 0.0
    elbow_angle_right: float = 0.0
    
    # Hip/knee extension (degrees, 180 = fully extended)
    hip_angle_left: float = 180.0
    hip_angle_right: float = 180.0
    knee_angle_left: float = 180.0
    knee_angle_right: float = 180.0
    
    # Wrist positions (normalized height ratios, >1.0 = above head)
    wrist_height_left: float = 0.5
    wrist_height_right: float = 0.5
    
    # Foot positions (normalized X coordinates for stance detection)
    ankle_x_left: float = 0.4
    ankle_x_right: float = 0.6
    
    # Confidence
    pose_confidence: float = 0.8
    
    @property
    def avg_elbow_angle(self) -> float:
        return (self.elbow_angle_left + self.elbow_angle_right) / 2
    
    @property
    def active_wrist_height(self) -> float:
        """Use the higher wrist (likely the active one)."""
        return max(self.wrist_height_left, self.wrist_height_right)
    
    @property
    def ankle_spread(self) -> float:
        """Distance between ankles (for stance detection)."""
        return abs(self.ankle_x_right - self.ankle_x_left)


@dataclass
class NoLiftReason:
    """Detailed reason for a no-lift classification."""
    code: str
    description: str
    frame: int
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class RepResult:
    """Result of rep detection and validation."""
    start_frame: int
    end_frame: int
    start_timestamp: float
    end_timestamp: float
    
    lift_type: LiftType
    validity: RepValidity
    
    # Detailed metrics
    peak_height: float = 0.0
    peak_frame: int = 0
    peak_velocity: float = 0.0
    fixation_frames: int = 0
    max_elbow_angle: float = 0.0
    
    # Validation details
    no_lift_reasons: List[NoLiftReason] = field(default_factory=list)
    phase_history: List[Tuple[str, int]] = field(default_factory=list)
    
    # Confidence
    confidence_score: float = 1.0
    
    @property
    def duration_seconds(self) -> float:
        return self.end_timestamp - self.start_timestamp


# =============================================================================
# SECTION 2: Body Stillness Detection (for Fixation)
# =============================================================================

class BodyStillnessDetector:
    """
    Detects total body stillness for fixation validation.
    
    In Kettlebell Sport, "Fixation" requires total body stillness:
    Variance of all major joint positions < ε
    """
    
    # Epsilon threshold for stillness (normalized position variance)
    STILLNESS_EPSILON = 0.002
    
    # Minimum frames of stillness for valid fixation
    MIN_STILLNESS_FRAMES = 5  # ~166ms at 30fps
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self._position_history: Deque[List[float]] = deque(maxlen=window_size)
        self._stillness_frame_count = 0
    
    def update(self, joint_positions: List[Tuple[float, float]]) -> bool:
        """
        Update with new joint positions and check stillness.
        
        Args:
            joint_positions: List of (x, y) tuples for major joints
            
        Returns:
            True if body is currently still (variance < ε)
        """
        # Flatten to 1D array of all coordinates
        flat_positions = []
        for x, y in joint_positions:
            flat_positions.extend([x, y])
        
        self._position_history.append(flat_positions)
        
        if len(self._position_history) < 3:
            return False
        
        # Calculate variance across window
        positions_array = np.array(list(self._position_history))
        variance = np.var(positions_array, axis=0).mean()
        
        is_still = variance < self.STILLNESS_EPSILON
        
        if is_still:
            self._stillness_frame_count += 1
        else:
            self._stillness_frame_count = 0
        
        return is_still
    
    @property
    def stillness_frames(self) -> int:
        return self._stillness_frame_count
    
    @property
    def is_fixed(self) -> bool:
        """True if stillness has been maintained for minimum duration."""
        return self._stillness_frame_count >= self.MIN_STILLNESS_FRAMES
    
    def reset(self):
        self._position_history.clear()
        self._stillness_frame_count = 0


# =============================================================================
# SECTION 3: Snatch State Machine
# =============================================================================

class SnatchState(Enum):
    """State machine states for snatch detection."""
    FLOOR = auto()        # Weight at floor/backswing position
    INITIATION = auto()   # V_y > threshold, starting from Y_floor
    ASCENT = auto()       # Continuous positive V_y
    CATCH = auto()        # Rapid deceleration, elbow snap
    FIXATION = auto()     # Stable overhead, body still
    COMPLETE = auto()     # Rep complete
    NO_LIFT = auto()      # Failed validation


class SnatchStateMachine:
    """
    Snatch Rep State Machine.
    
    Logic: Continuous positive V_y with a terminal overhead lockout.
    
    States:
    - State 0 (FLOOR): Weight at floor position
    - State 1 (INITIATION): V_y > threshold starting from Y_floor
    - State 2 (ASCENT): Continuous positive velocity
    - State 3 (CATCH): Rapid deceleration + elbow snap to 180°
    - State 4 (FIXATION): V_y ≈ 0 for >15 frames, Y_weight at max, full extension
    
    Validation:
    - If V_y ≤ 0 before head height → "No Lift (Segmented Pull)"
    """
    
    # Thresholds (normalized height ratios)
    FLOOR_HEIGHT = 0.35          # Below this = floor/backswing
    HEAD_HEIGHT = 0.90           # Must clear this before any deceleration
    OVERHEAD_HEIGHT = 1.0        # True overhead lockout position
    
    # Velocity thresholds (normalized height / second)
    INITIATION_VELOCITY = 0.3   # Minimum V_y to start rep
    STABLE_VELOCITY = 0.05       # V_y below this = stable
    
    # Timing (frames at 30fps)
    MIN_FIXATION_FRAMES = 15     # ~500ms required fixation
    MIN_REP_FRAMES = 12          # ~400ms minimum rep duration
    MAX_REP_FRAMES = 150         # ~5s maximum rep duration
    
    # Elbow angle
    LOCKOUT_ANGLE = 170.0        # Minimum elbow angle for lockout
    
    def __init__(self, fps: float = 30.0, dominant_hand: Optional[str] = None):
        self.fps = fps
        self.dominant_hand = dominant_hand
        
        # Adjust thresholds for actual FPS
        frame_ratio = fps / 30.0
        self.min_fixation_frames = max(3, int(self.MIN_FIXATION_FRAMES * frame_ratio))
        self.min_rep_frames = max(3, int(self.MIN_REP_FRAMES * frame_ratio))
        self.max_rep_frames = int(self.MAX_REP_FRAMES * frame_ratio)
        
        # State
        self.state = SnatchState.FLOOR
        self.detected_reps: List[RepResult] = []
        
        # Tracking
        self._rep_start_frame: Optional[int] = None
        self._rep_start_timestamp: Optional[float] = None
        self._peak_height = 0.0
        self._peak_frame = 0
        self._peak_velocity = 0.0
        self._max_elbow_angle = 0.0
        self._fixation_frame_count = 0
        self._phase_history: List[Tuple[str, int]] = []
        
        # History for velocity calculation
        self._frame_history: Deque[KinematicFrame] = deque(maxlen=30)
        
        # Body stillness detector
        self._stillness_detector = BodyStillnessDetector()
        
        # No-lift tracking
        self._no_lift_reasons: List[NoLiftReason] = []
        
        logger.info(f"SnatchStateMachine initialized: {fps} fps, "
                   f"fixation={self.min_fixation_frames} frames")
    
    def process_frame(self, frame: KinematicFrame) -> Optional[RepResult]:
        """
        Process a kinematic frame through the snatch state machine.
        
        Returns:
            RepResult if a rep just completed, None otherwise
        """
        self._frame_history.append(frame)
        
        # Get relevant measurements
        height = self._get_active_height(frame)
        velocity = frame.v_y
        elbow_angle = self._get_active_elbow(frame)
        
        # Track peak during active rep
        if self._rep_start_frame is not None:
            if height > self._peak_height:
                self._peak_height = height
                self._peak_frame = frame.frame_number
            if abs(velocity) > abs(self._peak_velocity):
                self._peak_velocity = velocity
            if elbow_angle > self._max_elbow_angle:
                self._max_elbow_angle = elbow_angle
        
        # Update stillness detector
        joint_positions = self._extract_joint_positions(frame)
        is_body_still = self._stillness_detector.update(joint_positions)
        
        # Debug logging
        if frame.frame_number % 10 == 0:
            logger.debug(f"Frame {frame.frame_number}: state={self.state.name}, "
                        f"h={height:.3f}, v={velocity:.3f}, still={is_body_still}")
        
        # Process state machine
        return self._process_state(frame, height, velocity, elbow_angle, is_body_still)
    
    def _get_active_height(self, frame: KinematicFrame) -> float:
        """Get height of active (KB-holding) wrist."""
        if self.dominant_hand == "left":
            return frame.wrist_height_left
        elif self.dominant_hand == "right":
            return frame.wrist_height_right
        else:
            return frame.active_wrist_height
    
    def _get_active_elbow(self, frame: KinematicFrame) -> float:
        """Get elbow angle of active arm."""
        if self.dominant_hand == "left":
            return frame.elbow_angle_left
        elif self.dominant_hand == "right":
            return frame.elbow_angle_right
        else:
            return max(frame.elbow_angle_left, frame.elbow_angle_right)
    
    def _extract_joint_positions(self, frame: KinematicFrame) -> List[Tuple[float, float]]:
        """Extract major joint positions for stillness detection."""
        # Use wrist and ankle positions as proxies
        return [
            (frame.wrist_height_left, 0.5),
            (frame.wrist_height_right, 0.5),
            (frame.ankle_x_left, 1.0),
            (frame.ankle_x_right, 1.0),
        ]
    
    def _process_state(
        self,
        frame: KinematicFrame,
        height: float,
        velocity: float,
        elbow_angle: float,
        is_body_still: bool
    ) -> Optional[RepResult]:
        """Process state machine transitions."""
        completed_rep = None
        
        if self.state == SnatchState.FLOOR:
            # Waiting at floor position
            if height < self.FLOOR_HEIGHT and velocity > self.INITIATION_VELOCITY:
                # Starting rep - positive velocity from floor
                self.state = SnatchState.INITIATION
                self._start_rep(frame)
                self._phase_history.append(("initiation", frame.frame_number))
                logger.info(f"Frame {frame.frame_number}: FLOOR → INITIATION "
                           f"(h={height:.3f}, v={velocity:.3f})")
        
        elif self.state == SnatchState.INITIATION:
            if velocity > 0:
                # Still ascending - transition to full ascent
                if height > self.FLOOR_HEIGHT:
                    self.state = SnatchState.ASCENT
                    self._phase_history.append(("ascent", frame.frame_number))
                    logger.info(f"Frame {frame.frame_number}: INITIATION → ASCENT")
            else:
                # VALIDATION CHECK: Velocity stopped before leaving floor zone
                # This is a failed swing, not a rep attempt
                self._no_lift_reasons.append(NoLiftReason(
                    code="EARLY_DECEL",
                    description="Velocity stopped before clearing floor zone",
                    frame=frame.frame_number,
                    metrics={"height": height, "velocity": velocity}
                ))
                self._reset_rep()
        
        elif self.state == SnatchState.ASCENT:
            # CRITICAL VALIDATION: Check for segmented pull
            if velocity <= 0 and height < self.HEAD_HEIGHT:
                # V_y ≤ 0 before clearing head height = "Segmented Pull"
                logger.warning(f"Frame {frame.frame_number}: SEGMENTED PULL detected "
                              f"(h={height:.3f} < {self.HEAD_HEIGHT})")
                self._no_lift_reasons.append(NoLiftReason(
                    code="SEGMENTED_PULL",
                    description="Velocity stopped before weight cleared head height",
                    frame=frame.frame_number,
                    metrics={"height": height, "velocity": velocity, "threshold": self.HEAD_HEIGHT}
                ))
                completed_rep = self._complete_rep(
                    frame, validity=RepValidity.NO_LIFT
                )
            
            elif height >= self.OVERHEAD_HEIGHT and velocity < self.STABLE_VELOCITY:
                # Reached overhead and decelerating - transition to catch
                self.state = SnatchState.CATCH
                self._phase_history.append(("catch", frame.frame_number))
                logger.info(f"Frame {frame.frame_number}: ASCENT → CATCH "
                           f"(h={height:.3f}, v={velocity:.3f})")
        
        elif self.state == SnatchState.CATCH:
            # Check for elbow lockout
            has_lockout = elbow_angle >= self.LOCKOUT_ANGLE
            is_stable = abs(velocity) < self.STABLE_VELOCITY
            
            if has_lockout and is_stable:
                self.state = SnatchState.FIXATION
                self._phase_history.append(("fixation", frame.frame_number))
                self._fixation_frame_count = 0
                logger.info(f"Frame {frame.frame_number}: CATCH → FIXATION "
                           f"(elbow={elbow_angle:.1f}°)")
            
            elif height < self.HEAD_HEIGHT:
                # Dropped without lockout
                self._no_lift_reasons.append(NoLiftReason(
                    code="NO_LOCKOUT",
                    description="Weight dropped before achieving elbow lockout",
                    frame=frame.frame_number,
                    metrics={"elbow_angle": elbow_angle, "threshold": self.LOCKOUT_ANGLE}
                ))
                completed_rep = self._complete_rep(frame, validity=RepValidity.NO_LIFT)
        
        elif self.state == SnatchState.FIXATION:
            is_stable = abs(velocity) < self.STABLE_VELOCITY
            is_overhead = height >= self.OVERHEAD_HEIGHT * 0.95  # Small tolerance
            
            if is_stable and is_overhead and is_body_still:
                self._fixation_frame_count += 1
                
                if self._fixation_frame_count >= self.min_fixation_frames:
                    # VALID REP - sufficient fixation achieved
                    logger.info(f"Frame {frame.frame_number}: FIXATION complete "
                               f"({self._fixation_frame_count} frames)")
                    completed_rep = self._complete_rep(frame, validity=RepValidity.VALID)
            else:
                # Not stable - check if dropping
                if height < self.OVERHEAD_HEIGHT * 0.9:
                    if self._fixation_frame_count < self.min_fixation_frames:
                        self._no_lift_reasons.append(NoLiftReason(
                            code="INSUFFICIENT_FIXATION",
                            description=f"Fixation held for only {self._fixation_frame_count} frames "
                                       f"(minimum: {self.min_fixation_frames})",
                            frame=frame.frame_number,
                            metrics={"frames": self._fixation_frame_count}
                        ))
                    completed_rep = self._complete_rep(
                        frame, 
                        validity=RepValidity.VALID if self._fixation_frame_count >= self.min_fixation_frames 
                                 else RepValidity.NO_LIFT
                    )
        
        # Timeout check
        if self._rep_start_frame is not None:
            rep_frames = frame.frame_number - self._rep_start_frame
            if rep_frames > self.max_rep_frames:
                logger.warning(f"Frame {frame.frame_number}: Rep timeout")
                self._reset_rep()
        
        return completed_rep
    
    def _start_rep(self, frame: KinematicFrame):
        """Initialize tracking for a new rep."""
        self._rep_start_frame = frame.frame_number
        self._rep_start_timestamp = frame.timestamp
        self._peak_height = 0.0
        self._peak_frame = frame.frame_number
        self._peak_velocity = 0.0
        self._max_elbow_angle = 0.0
        self._fixation_frame_count = 0
        self._phase_history = []
        self._no_lift_reasons = []
        self._stillness_detector.reset()
    
    def _complete_rep(
        self, 
        frame: KinematicFrame, 
        validity: RepValidity
    ) -> RepResult:
        """Complete current rep and return result."""
        result = RepResult(
            start_frame=self._rep_start_frame or frame.frame_number,
            end_frame=frame.frame_number,
            start_timestamp=self._rep_start_timestamp or frame.timestamp,
            end_timestamp=frame.timestamp,
            lift_type=LiftType.SNATCH,
            validity=validity,
            peak_height=self._peak_height,
            peak_frame=self._peak_frame,
            peak_velocity=self._peak_velocity,
            fixation_frames=self._fixation_frame_count,
            max_elbow_angle=self._max_elbow_angle,
            no_lift_reasons=list(self._no_lift_reasons),
            phase_history=list(self._phase_history),
            confidence_score=0.95 if validity == RepValidity.VALID else 0.7
        )
        
        self.detected_reps.append(result)
        self._reset_rep()
        
        return result
    
    def _reset_rep(self):
        """Reset state for next rep."""
        self.state = SnatchState.FLOOR
        self._rep_start_frame = None
        self._rep_start_timestamp = None
        self._peak_height = 0.0
        self._peak_frame = 0
        self._peak_velocity = 0.0
        self._max_elbow_angle = 0.0
        self._fixation_frame_count = 0
        self._phase_history = []
        self._no_lift_reasons = []
        self._stillness_detector.reset()
    
    def finalize(self) -> List[RepResult]:
        """Finalize and return all detected reps."""
        return self.detected_reps


# =============================================================================
# SECTION 4: Jerk State Machine  
# =============================================================================

class JerkState(Enum):
    """State machine states for jerk detection."""
    RACK = auto()         # V_y ≈ 0 at shoulder level (>10 frames)
    DIP = auto()          # Brief V_y < 0 (negative velocity)
    DRIVE = auto()        # Maximum V_y (positive)
    OVERHEAD = auto()     # Weight at max height
    RECOVERY = auto()     # Feet returning to parallel stance
    FIXATION = auto()     # Stationary overhead, feet aligned
    COMPLETE = auto()     # Rep complete
    NO_LIFT = auto()      # Failed validation


class JerkStateMachine:
    """
    Jerk Rep State Machine.
    
    Logic: Discontinuous V_y starting from a mid-level stationary point.
    
    States:
    - State 0 (RACK): V_y ≈ 0 at shoulder-level for >10 frames
    - State 1 (DIP): Brief V_y < 0 (negative velocity transition)
    - State 2 (DRIVE): Immediate transition to maximum V_y (positive)
    - State 3 (RECOVERY): Weight at Y_max, feet moving from split to parallel
    - State 4 (FIXATION): Stationary weight overhead with feet aligned
    """
    
    # Height thresholds (normalized ratios)
    RACK_HEIGHT_LOW = 0.45
    RACK_HEIGHT_HIGH = 0.60
    OVERHEAD_HEIGHT = 0.95
    
    # Velocity thresholds
    STABLE_VELOCITY = 0.05
    DIP_VELOCITY = -0.15         # Negative velocity for dip
    DRIVE_VELOCITY = 0.3         # Minimum positive velocity for drive
    
    # Stance thresholds
    PARALLEL_STANCE_TOLERANCE = 0.15  # Max ankle spread for "parallel"
    
    # Timing
    MIN_RACK_FRAMES = 10
    MIN_FIXATION_FRAMES = 10
    MIN_REP_FRAMES = 15
    MAX_REP_FRAMES = 120
    
    # Elbow
    LOCKOUT_ANGLE = 168.0
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        
        # Adjust thresholds for FPS
        frame_ratio = fps / 30.0
        self.min_rack_frames = max(3, int(self.MIN_RACK_FRAMES * frame_ratio))
        self.min_fixation_frames = max(3, int(self.MIN_FIXATION_FRAMES * frame_ratio))
        self.min_rep_frames = max(5, int(self.MIN_REP_FRAMES * frame_ratio))
        self.max_rep_frames = int(self.MAX_REP_FRAMES * frame_ratio)
        
        # State
        self.state = JerkState.RACK
        self.detected_reps: List[RepResult] = []
        
        # Tracking
        self._rack_frame_count = 0
        self._rep_start_frame: Optional[int] = None
        self._rep_start_timestamp: Optional[float] = None
        self._peak_height = 0.0
        self._peak_frame = 0
        self._peak_velocity = 0.0
        self._max_elbow_angle = 0.0
        self._fixation_frame_count = 0
        self._initial_stance: Optional[float] = None
        self._phase_history: List[Tuple[str, int]] = []
        self._no_lift_reasons: List[NoLiftReason] = []
        
        # History
        self._frame_history: Deque[KinematicFrame] = deque(maxlen=30)
        self._stillness_detector = BodyStillnessDetector()
        
        logger.info(f"JerkStateMachine initialized: {fps} fps")
    
    def process_frame(self, frame: KinematicFrame) -> Optional[RepResult]:
        """Process a kinematic frame through the jerk state machine."""
        self._frame_history.append(frame)
        
        # Average both wrists for double KB
        height = (frame.wrist_height_left + frame.wrist_height_right) / 2
        velocity = frame.v_y
        elbow_angle = frame.avg_elbow_angle
        stance = frame.ankle_spread
        
        # Track during active rep
        if self._rep_start_frame is not None:
            if height > self._peak_height:
                self._peak_height = height
                self._peak_frame = frame.frame_number
            if abs(velocity) > abs(self._peak_velocity):
                self._peak_velocity = velocity
            if elbow_angle > self._max_elbow_angle:
                self._max_elbow_angle = elbow_angle
        
        # Update stillness
        joint_positions = [
            (frame.wrist_height_left, 0.5),
            (frame.wrist_height_right, 0.5),
            (frame.ankle_x_left, 1.0),
            (frame.ankle_x_right, 1.0),
        ]
        is_body_still = self._stillness_detector.update(joint_positions)
        
        return self._process_state(frame, height, velocity, elbow_angle, stance, is_body_still)
    
    def _process_state(
        self,
        frame: KinematicFrame,
        height: float,
        velocity: float,
        elbow_angle: float,
        stance: float,
        is_body_still: bool
    ) -> Optional[RepResult]:
        """Process jerk state machine transitions."""
        completed_rep = None
        is_at_rack = self.RACK_HEIGHT_LOW <= height <= self.RACK_HEIGHT_HIGH
        is_stable = abs(velocity) < self.STABLE_VELOCITY
        
        if self.state == JerkState.RACK:
            if is_at_rack and is_stable:
                self._rack_frame_count += 1
                
                if self._rack_frame_count >= self.min_rack_frames:
                    # Stable at rack - ready for dip
                    if velocity < self.DIP_VELOCITY:
                        # Starting dip
                        self.state = JerkState.DIP
                        self._start_rep(frame)
                        self._initial_stance = stance
                        self._phase_history.append(("dip", frame.frame_number))
                        logger.info(f"Frame {frame.frame_number}: RACK → DIP "
                                   f"(v={velocity:.3f})")
            else:
                self._rack_frame_count = 0
        
        elif self.state == JerkState.DIP:
            if velocity > self.DRIVE_VELOCITY:
                # Transitioning to drive
                self.state = JerkState.DRIVE
                self._phase_history.append(("drive", frame.frame_number))
                logger.info(f"Frame {frame.frame_number}: DIP → DRIVE "
                           f"(v={velocity:.3f})")
            elif is_at_rack and is_stable:
                # Aborted dip - back to rack
                self._no_lift_reasons.append(NoLiftReason(
                    code="ABORTED_DIP",
                    description="Dip aborted before drive phase",
                    frame=frame.frame_number
                ))
                completed_rep = self._complete_rep(frame, validity=RepValidity.NO_LIFT)
        
        elif self.state == JerkState.DRIVE:
            if height >= self.OVERHEAD_HEIGHT:
                # Reached overhead
                self.state = JerkState.OVERHEAD
                self._phase_history.append(("overhead", frame.frame_number))
                logger.info(f"Frame {frame.frame_number}: DRIVE → OVERHEAD")
            elif velocity < 0 and height < self.RACK_HEIGHT_HIGH:
                # Failed to reach overhead
                self._no_lift_reasons.append(NoLiftReason(
                    code="DRIVE_FAILED",
                    description="Drive failed to reach overhead position",
                    frame=frame.frame_number,
                    metrics={"height": height, "target": self.OVERHEAD_HEIGHT}
                ))
                completed_rep = self._complete_rep(frame, validity=RepValidity.NO_LIFT)
        
        elif self.state == JerkState.OVERHEAD:
            has_lockout = elbow_angle >= self.LOCKOUT_ANGLE
            is_parallel = stance <= self.PARALLEL_STANCE_TOLERANCE
            
            if not is_parallel:
                # In split stance - need recovery
                self.state = JerkState.RECOVERY
                self._phase_history.append(("recovery", frame.frame_number))
                logger.info(f"Frame {frame.frame_number}: OVERHEAD → RECOVERY "
                           f"(stance={stance:.3f})")
            elif has_lockout and is_stable:
                # Already parallel with lockout - go to fixation
                self.state = JerkState.FIXATION
                self._phase_history.append(("fixation", frame.frame_number))
                self._fixation_frame_count = 0
        
        elif self.state == JerkState.RECOVERY:
            is_parallel = stance <= self.PARALLEL_STANCE_TOLERANCE
            has_lockout = elbow_angle >= self.LOCKOUT_ANGLE
            
            if is_parallel and has_lockout:
                # Recovered to parallel with lockout
                self.state = JerkState.FIXATION
                self._phase_history.append(("fixation", frame.frame_number))
                self._fixation_frame_count = 0
                logger.info(f"Frame {frame.frame_number}: RECOVERY → FIXATION")
            elif height < self.RACK_HEIGHT_HIGH:
                # Dropped during recovery
                self._no_lift_reasons.append(NoLiftReason(
                    code="DROPPED_IN_RECOVERY",
                    description="Weight dropped before completing recovery to parallel stance",
                    frame=frame.frame_number
                ))
                completed_rep = self._complete_rep(frame, validity=RepValidity.NO_LIFT)
        
        elif self.state == JerkState.FIXATION:
            is_overhead = height >= self.OVERHEAD_HEIGHT * 0.95
            is_parallel = stance <= self.PARALLEL_STANCE_TOLERANCE
            
            if is_overhead and is_stable and is_parallel and is_body_still:
                self._fixation_frame_count += 1
                
                if self._fixation_frame_count >= self.min_fixation_frames:
                    # VALID REP
                    logger.info(f"Frame {frame.frame_number}: FIXATION complete")
                    completed_rep = self._complete_rep(frame, validity=RepValidity.VALID)
            else:
                if height < self.RACK_HEIGHT_HIGH:
                    # Returned to rack
                    if self._fixation_frame_count >= self.min_fixation_frames:
                        completed_rep = self._complete_rep(frame, validity=RepValidity.VALID)
                    else:
                        self._no_lift_reasons.append(NoLiftReason(
                            code="INSUFFICIENT_FIXATION",
                            description=f"Fixation: {self._fixation_frame_count}/{self.min_fixation_frames} frames",
                            frame=frame.frame_number
                        ))
                        completed_rep = self._complete_rep(frame, validity=RepValidity.NO_LIFT)
        
        # Timeout
        if self._rep_start_frame is not None:
            if frame.frame_number - self._rep_start_frame > self.max_rep_frames:
                logger.warning(f"Frame {frame.frame_number}: Jerk rep timeout")
                completed_rep = self._complete_rep(frame, validity=RepValidity.AMBIGUOUS)
        
        return completed_rep
    
    def _start_rep(self, frame: KinematicFrame):
        """Initialize tracking for new rep."""
        self._rep_start_frame = frame.frame_number
        self._rep_start_timestamp = frame.timestamp
        self._peak_height = 0.0
        self._peak_frame = frame.frame_number
        self._peak_velocity = 0.0
        self._max_elbow_angle = 0.0
        self._fixation_frame_count = 0
        self._phase_history = []
        self._no_lift_reasons = []
        self._stillness_detector.reset()
    
    def _complete_rep(self, frame: KinematicFrame, validity: RepValidity) -> RepResult:
        """Complete current rep."""
        result = RepResult(
            start_frame=self._rep_start_frame or frame.frame_number,
            end_frame=frame.frame_number,
            start_timestamp=self._rep_start_timestamp or frame.timestamp,
            end_timestamp=frame.timestamp,
            lift_type=LiftType.JERK,
            validity=validity,
            peak_height=self._peak_height,
            peak_frame=self._peak_frame,
            peak_velocity=self._peak_velocity,
            fixation_frames=self._fixation_frame_count,
            max_elbow_angle=self._max_elbow_angle,
            no_lift_reasons=list(self._no_lift_reasons),
            phase_history=list(self._phase_history),
            confidence_score=0.95 if validity == RepValidity.VALID else 0.7
        )
        
        self.detected_reps.append(result)
        self._reset_rep()
        return result
    
    def _reset_rep(self):
        """Reset for next rep."""
        self.state = JerkState.RACK
        self._rack_frame_count = 0
        self._rep_start_frame = None
        self._peak_height = 0.0
        self._fixation_frame_count = 0
        self._phase_history = []
        self._no_lift_reasons = []
        self._stillness_detector.reset()
    
    def finalize(self) -> List[RepResult]:
        return self.detected_reps


# =============================================================================
# SECTION 5: Long Cycle State Machine
# =============================================================================

class LongCycleState(Enum):
    """State machine states for long cycle (clean & jerk) detection."""
    BACKSWING = auto()       # Weight at low swing position
    CLEAN = auto()           # Weight moving from Y_low_swing to Y_rack
    RACK_PAUSE = auto()      # MANDATORY: V_y = 0 at rack position
    JERK_DIP = auto()        # Jerk dip phase
    JERK_DRIVE = auto()      # Jerk drive phase
    OVERHEAD = auto()        # Weight at overhead
    FIXATION = auto()        # Total body stillness
    DROP = auto()            # Return to backswing
    COMPLETE = auto()
    NO_LIFT = auto()


class LongCycleStateMachine:
    """
    Long Cycle (Clean & Jerk) State Machine.
    
    Logic: Compound state machine [Clean → Rack → Jerk → Drop]
    
    States:
    - State 0 (CLEAN): Weight moves from Y_low_swing to Y_rack
    - State 1 (RACK_PAUSE): MANDATORY - V_y = 0 at rack position
      * If weight bypasses rack and goes straight overhead → classify as SNATCH
    - State 2 (JERK): Execute Jerk Logic (Dip/Drive/Fixation)
    - State 3 (FIXATION): Total body stillness (variance of all joints < ε)
    - Rep Complete: Return of weight to backswing
    """
    
    # Height thresholds
    BACKSWING_HEIGHT = 0.30
    RACK_HEIGHT_LOW = 0.45
    RACK_HEIGHT_HIGH = 0.60
    OVERHEAD_HEIGHT = 0.95
    
    # Velocity thresholds
    STABLE_VELOCITY = 0.05
    DIP_VELOCITY = -0.15
    DRIVE_VELOCITY = 0.3
    
    # Timing
    MIN_RACK_PAUSE_FRAMES = 5    # Mandatory pause at rack
    MIN_FIXATION_FRAMES = 8
    MIN_REP_FRAMES = 30          # LC is longer than snatch
    MAX_REP_FRAMES = 180
    
    LOCKOUT_ANGLE = 168.0
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        
        frame_ratio = fps / 30.0
        self.min_rack_pause_frames = max(2, int(self.MIN_RACK_PAUSE_FRAMES * frame_ratio))
        self.min_fixation_frames = max(3, int(self.MIN_FIXATION_FRAMES * frame_ratio))
        self.min_rep_frames = max(10, int(self.MIN_REP_FRAMES * frame_ratio))
        self.max_rep_frames = int(self.MAX_REP_FRAMES * frame_ratio)
        
        # State
        self.state = LongCycleState.BACKSWING
        self.detected_reps: List[RepResult] = []
        self._misclassified_as_snatch: List[RepResult] = []
        
        # Tracking
        self._rep_start_frame: Optional[int] = None
        self._rep_start_timestamp: Optional[float] = None
        self._peak_height = 0.0
        self._peak_frame = 0
        self._peak_velocity = 0.0
        self._max_elbow_angle = 0.0
        self._rack_pause_frames = 0
        self._fixation_frame_count = 0
        self._had_rack_pause = False
        self._phase_history: List[Tuple[str, int]] = []
        self._no_lift_reasons: List[NoLiftReason] = []
        
        self._frame_history: Deque[KinematicFrame] = deque(maxlen=30)
        self._stillness_detector = BodyStillnessDetector()
        
        logger.info(f"LongCycleStateMachine initialized: {fps} fps, "
                   f"rack_pause={self.min_rack_pause_frames} frames")
    
    def process_frame(self, frame: KinematicFrame) -> Optional[RepResult]:
        """Process a kinematic frame through the long cycle state machine."""
        self._frame_history.append(frame)
        
        # Average both wrists for double KB
        height = (frame.wrist_height_left + frame.wrist_height_right) / 2
        velocity = frame.v_y
        elbow_angle = frame.avg_elbow_angle
        
        # Track during rep
        if self._rep_start_frame is not None:
            if height > self._peak_height:
                self._peak_height = height
                self._peak_frame = frame.frame_number
            if abs(velocity) > abs(self._peak_velocity):
                self._peak_velocity = velocity
            if elbow_angle > self._max_elbow_angle:
                self._max_elbow_angle = elbow_angle
        
        # Stillness
        joint_positions = [
            (frame.wrist_height_left, 0.5),
            (frame.wrist_height_right, 0.5),
            (frame.ankle_x_left, 1.0),
            (frame.ankle_x_right, 1.0),
        ]
        is_body_still = self._stillness_detector.update(joint_positions)
        
        return self._process_state(frame, height, velocity, elbow_angle, is_body_still)
    
    def _process_state(
        self,
        frame: KinematicFrame,
        height: float,
        velocity: float,
        elbow_angle: float,
        is_body_still: bool
    ) -> Optional[RepResult]:
        """Process long cycle state machine."""
        completed_rep = None
        is_backswing = height < self.BACKSWING_HEIGHT
        is_rack = self.RACK_HEIGHT_LOW <= height <= self.RACK_HEIGHT_HIGH
        is_overhead = height >= self.OVERHEAD_HEIGHT
        is_stable = abs(velocity) < self.STABLE_VELOCITY
        
        if self.state == LongCycleState.BACKSWING:
            if is_backswing and velocity > 0:
                # Starting clean
                self.state = LongCycleState.CLEAN
                self._start_rep(frame)
                self._phase_history.append(("clean", frame.frame_number))
                logger.info(f"Frame {frame.frame_number}: BACKSWING → CLEAN")
        
        elif self.state == LongCycleState.CLEAN:
            if is_rack:
                # Reached rack - need to pause
                self.state = LongCycleState.RACK_PAUSE
                self._rack_pause_frames = 0
                self._phase_history.append(("rack_pause", frame.frame_number))
                logger.info(f"Frame {frame.frame_number}: CLEAN → RACK_PAUSE")
            
            elif is_overhead:
                # VALIDATION: Bypassed rack and went straight overhead
                # This is a SNATCH, not a Long Cycle!
                logger.warning(f"Frame {frame.frame_number}: Weight bypassed rack → SNATCH")
                self._no_lift_reasons.append(NoLiftReason(
                    code="BYPASSED_RACK",
                    description="Weight went directly to overhead without stopping at rack - "
                               "classified as SNATCH, not Long Cycle",
                    frame=frame.frame_number
                ))
                
                # Create as snatch instead
                snatch_result = RepResult(
                    start_frame=self._rep_start_frame or frame.frame_number,
                    end_frame=frame.frame_number,
                    start_timestamp=self._rep_start_timestamp or frame.timestamp,
                    end_timestamp=frame.timestamp,
                    lift_type=LiftType.SNATCH,  # Reclassify as snatch
                    validity=RepValidity.VALID,
                    peak_height=self._peak_height,
                    peak_frame=self._peak_frame,
                    no_lift_reasons=list(self._no_lift_reasons),
                    phase_history=list(self._phase_history),
                    confidence_score=0.85
                )
                self._misclassified_as_snatch.append(snatch_result)
                self._reset_rep()
                return snatch_result
            
            elif is_backswing and len(self._phase_history) > 0:
                # Failed clean - returned to backswing
                self._no_lift_reasons.append(NoLiftReason(
                    code="FAILED_CLEAN",
                    description="Failed to complete clean phase",
                    frame=frame.frame_number
                ))
                completed_rep = self._complete_rep(frame, validity=RepValidity.NO_LIFT)
        
        elif self.state == LongCycleState.RACK_PAUSE:
            if is_rack and is_stable:
                self._rack_pause_frames += 1
                
                if self._rack_pause_frames >= self.min_rack_pause_frames:
                    self._had_rack_pause = True
                    
                    # Check for jerk initiation (dip)
                    if velocity < self.DIP_VELOCITY:
                        self.state = LongCycleState.JERK_DIP
                        self._phase_history.append(("jerk_dip", frame.frame_number))
                        logger.info(f"Frame {frame.frame_number}: RACK_PAUSE → JERK_DIP")
            
            elif not is_rack:
                if height > self.RACK_HEIGHT_HIGH:
                    # Starting drive without proper pause
                    if self._rack_pause_frames < self.min_rack_pause_frames:
                        self._no_lift_reasons.append(NoLiftReason(
                            code="INSUFFICIENT_RACK_PAUSE",
                            description=f"Rack pause only {self._rack_pause_frames}/{self.min_rack_pause_frames} frames",
                            frame=frame.frame_number
                        ))
                    # Continue to jerk anyway
                    self.state = LongCycleState.JERK_DRIVE
                    self._phase_history.append(("jerk_drive", frame.frame_number))
                elif is_backswing:
                    # Dropped from rack
                    self._no_lift_reasons.append(NoLiftReason(
                        code="DROPPED_FROM_RACK",
                        description="Weight dropped from rack without jerk attempt",
                        frame=frame.frame_number
                    ))
                    completed_rep = self._complete_rep(frame, validity=RepValidity.NO_LIFT)
        
        elif self.state == LongCycleState.JERK_DIP:
            if velocity > self.DRIVE_VELOCITY:
                self.state = LongCycleState.JERK_DRIVE
                self._phase_history.append(("jerk_drive", frame.frame_number))
                logger.info(f"Frame {frame.frame_number}: JERK_DIP → JERK_DRIVE")
            elif is_rack and is_stable:
                # Aborted dip
                self._rack_pause_frames = 0  # Reset pause counter
                self.state = LongCycleState.RACK_PAUSE
        
        elif self.state == LongCycleState.JERK_DRIVE:
            if is_overhead:
                self.state = LongCycleState.OVERHEAD
                self._phase_history.append(("overhead", frame.frame_number))
                logger.info(f"Frame {frame.frame_number}: JERK_DRIVE → OVERHEAD")
            elif is_rack and velocity <= 0:
                # Failed drive
                self._no_lift_reasons.append(NoLiftReason(
                    code="FAILED_JERK_DRIVE",
                    description="Jerk drive failed to reach overhead",
                    frame=frame.frame_number
                ))
                completed_rep = self._complete_rep(frame, validity=RepValidity.NO_LIFT)
        
        elif self.state == LongCycleState.OVERHEAD:
            has_lockout = elbow_angle >= self.LOCKOUT_ANGLE
            
            if has_lockout and is_stable:
                self.state = LongCycleState.FIXATION
                self._phase_history.append(("fixation", frame.frame_number))
                self._fixation_frame_count = 0
            elif height < self.RACK_HEIGHT_HIGH:
                # Started dropping
                self.state = LongCycleState.DROP
                self._phase_history.append(("drop", frame.frame_number))
        
        elif self.state == LongCycleState.FIXATION:
            if is_overhead and is_stable and is_body_still:
                self._fixation_frame_count += 1
                
                if self._fixation_frame_count >= self.min_fixation_frames:
                    # Valid fixation achieved - start drop
                    self.state = LongCycleState.DROP
                    self._phase_history.append(("drop", frame.frame_number))
                    logger.info(f"Frame {frame.frame_number}: FIXATION → DROP")
            else:
                if height < self.OVERHEAD_HEIGHT * 0.9:
                    self.state = LongCycleState.DROP
                    self._phase_history.append(("drop", frame.frame_number))
                    if self._fixation_frame_count < self.min_fixation_frames:
                        self._no_lift_reasons.append(NoLiftReason(
                            code="INSUFFICIENT_FIXATION",
                            description=f"Fixation: {self._fixation_frame_count}/{self.min_fixation_frames}",
                            frame=frame.frame_number
                        ))
        
        elif self.state == LongCycleState.DROP:
            if is_backswing:
                # Rep complete - returned to backswing
                validity = RepValidity.VALID
                if self._fixation_frame_count < self.min_fixation_frames:
                    validity = RepValidity.NO_LIFT
                if not self._had_rack_pause:
                    validity = RepValidity.NO_LIFT
                
                completed_rep = self._complete_rep(frame, validity=validity)
        
        # Timeout
        if self._rep_start_frame is not None:
            if frame.frame_number - self._rep_start_frame > self.max_rep_frames:
                completed_rep = self._complete_rep(frame, validity=RepValidity.AMBIGUOUS)
        
        return completed_rep
    
    def _start_rep(self, frame: KinematicFrame):
        self._rep_start_frame = frame.frame_number
        self._rep_start_timestamp = frame.timestamp
        self._peak_height = 0.0
        self._peak_frame = frame.frame_number
        self._peak_velocity = 0.0
        self._max_elbow_angle = 0.0
        self._rack_pause_frames = 0
        self._fixation_frame_count = 0
        self._had_rack_pause = False
        self._phase_history = []
        self._no_lift_reasons = []
        self._stillness_detector.reset()
    
    def _complete_rep(self, frame: KinematicFrame, validity: RepValidity) -> RepResult:
        result = RepResult(
            start_frame=self._rep_start_frame or frame.frame_number,
            end_frame=frame.frame_number,
            start_timestamp=self._rep_start_timestamp or frame.timestamp,
            end_timestamp=frame.timestamp,
            lift_type=LiftType.LONG_CYCLE,
            validity=validity,
            peak_height=self._peak_height,
            peak_frame=self._peak_frame,
            peak_velocity=self._peak_velocity,
            fixation_frames=self._fixation_frame_count,
            max_elbow_angle=self._max_elbow_angle,
            no_lift_reasons=list(self._no_lift_reasons),
            phase_history=list(self._phase_history),
            confidence_score=0.95 if validity == RepValidity.VALID else 0.7
        )
        
        self.detected_reps.append(result)
        self._reset_rep()
        return result
    
    def _reset_rep(self):
        self.state = LongCycleState.BACKSWING
        self._rep_start_frame = None
        self._peak_height = 0.0
        self._rack_pause_frames = 0
        self._fixation_frame_count = 0
        self._had_rack_pause = False
        self._phase_history = []
        self._no_lift_reasons = []
        self._stillness_detector.reset()
    
    def finalize(self) -> List[RepResult]:
        return self.detected_reps
    
    def get_reclassified_snatches(self) -> List[RepResult]:
        """Get reps that were reclassified as snatches (bypassed rack)."""
        return self._misclassified_as_snatch


# =============================================================================
# SECTION 6: Unified Movement Classifier
# =============================================================================

class MovementClassifier:
    """
    Unified Athletic Movement Classifier.
    
    Orchestrates lift-specific state machines and provides a clean API
    for processing video frames.
    
    Usage:
        classifier = MovementClassifier(lift_type="snatch", fps=30.0)
        
        for frame in pose_sequence:
            kinematic_frame = classifier.convert_pose(frame)
            result = classifier.process_frame(kinematic_frame)
            if result:
                print(f"Rep detected: {result.validity.value}")
        
        all_reps = classifier.finalize()
    """
    
    def __init__(
        self,
        lift_type: str,
        fps: float = 30.0,
        dominant_hand: Optional[str] = None
    ):
        self.lift_type = LiftType(lift_type) if isinstance(lift_type, str) else lift_type
        self.fps = fps
        self.dominant_hand = dominant_hand
        
        # Create appropriate state machine
        if self.lift_type == LiftType.SNATCH:
            self._state_machine = SnatchStateMachine(fps, dominant_hand)
        elif self.lift_type == LiftType.JERK:
            self._state_machine = JerkStateMachine(fps)
        elif self.lift_type == LiftType.LONG_CYCLE:
            self._state_machine = LongCycleStateMachine(fps)
        else:
            raise ValueError(f"Unsupported lift type: {lift_type}")
        
        # Velocity calculator
        self._prev_height: Optional[float] = None
        self._prev_timestamp: Optional[float] = None
        
        logger.info(f"MovementClassifier initialized: {self.lift_type.value}, {fps} fps")
    
    def convert_pose(self, pose) -> KinematicFrame:
        """
        Convert a HybridPose to KinematicFrame.
        
        Args:
            pose: HybridPose from the estimator
            
        Returns:
            KinematicFrame with kinematic measurements
        """
        # Get wrist heights
        left_h = pose.get_wrist_height_ratio("left") or 0.5
        right_h = pose.get_wrist_height_ratio("right") or 0.5
        
        # Calculate weight height (use appropriate wrist based on lift type)
        if self.lift_type == LiftType.SNATCH:
            if self.dominant_hand == "left":
                y_weight = left_h
            elif self.dominant_hand == "right":
                y_weight = right_h
            else:
                y_weight = max(left_h, right_h)
        else:
            # Double KB - average both
            y_weight = (left_h + right_h) / 2
        
        # Calculate vertical velocity
        v_y = 0.0
        if self._prev_height is not None and self._prev_timestamp is not None:
            dt = pose.timestamp - self._prev_timestamp
            if dt > 0:
                v_y = (y_weight - self._prev_height) / dt
        
        self._prev_height = y_weight
        self._prev_timestamp = pose.timestamp
        
        # Get elbow angles
        left_elbow = pose.get_elbow_angle("left") or 90.0
        right_elbow = pose.get_elbow_angle("right") or 90.0
        
        # Get ankle positions (if available, else use defaults)
        left_ankle_x = 0.4
        right_ankle_x = 0.6
        
        # Try to get from body keypoints
        if hasattr(pose, 'body_keypoints') and len(pose.body_keypoints) > 16:
            left_ankle = pose.body_keypoints[15]  # LEFT_ANKLE
            right_ankle = pose.body_keypoints[16]  # RIGHT_ANKLE
            if left_ankle.is_visible:
                left_ankle_x = left_ankle.x
            if right_ankle.is_visible:
                right_ankle_x = right_ankle.x
        
        return KinematicFrame(
            frame_number=pose.frame_number,
            timestamp=pose.timestamp,
            fps=self.fps,
            y_weight=y_weight,
            v_y=v_y,
            elbow_angle_left=left_elbow,
            elbow_angle_right=right_elbow,
            wrist_height_left=left_h,
            wrist_height_right=right_h,
            ankle_x_left=left_ankle_x,
            ankle_x_right=right_ankle_x,
            pose_confidence=pose.overall_confidence if hasattr(pose, 'overall_confidence') else 0.8
        )
    
    def process_frame(self, frame: KinematicFrame) -> Optional[RepResult]:
        """
        Process a kinematic frame through the state machine.
        
        Args:
            frame: KinematicFrame with kinematic measurements
            
        Returns:
            RepResult if a rep just completed, None otherwise
        """
        return self._state_machine.process_frame(frame)
    
    def process_pose(self, pose) -> Optional[RepResult]:
        """
        Convenience method: convert pose and process in one step.
        
        Args:
            pose: HybridPose from the estimator
            
        Returns:
            RepResult if a rep just completed, None otherwise
        """
        frame = self.convert_pose(pose)
        return self.process_frame(frame)
    
    def finalize(self) -> List[RepResult]:
        """
        Finalize detection and return all detected reps.
        
        Returns:
            List of all RepResult objects
        """
        return self._state_machine.finalize()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with counts and statistics
        """
        reps = self.finalize()
        
        valid_count = sum(1 for r in reps if r.validity == RepValidity.VALID)
        no_lift_count = sum(1 for r in reps if r.validity == RepValidity.NO_LIFT)
        ambiguous_count = sum(1 for r in reps if r.validity == RepValidity.AMBIGUOUS)
        
        # Collect no-lift reasons
        no_lift_reasons_count: Dict[str, int] = {}
        for rep in reps:
            if rep.validity == RepValidity.NO_LIFT:
                for reason in rep.no_lift_reasons:
                    no_lift_reasons_count[reason.code] = no_lift_reasons_count.get(reason.code, 0) + 1
        
        # Reclassified snatches (for long cycle only)
        reclassified_snatches = 0
        if hasattr(self._state_machine, 'get_reclassified_snatches'):
            reclassified_snatches = len(self._state_machine.get_reclassified_snatches())
        
        return {
            "lift_type": self.lift_type.value,
            "total_attempts": len(reps),
            "valid_reps": valid_count,
            "no_reps": no_lift_count,
            "ambiguous_reps": ambiguous_count,
            "no_lift_reasons": no_lift_reasons_count,
            "reclassified_as_snatch": reclassified_snatches
        }


# =============================================================================
# SECTION 7: Factory Function
# =============================================================================

def create_movement_classifier(
    lift_type: str,
    fps: float = 30.0,
    dominant_hand: Optional[str] = None
) -> MovementClassifier:
    """
    Factory function to create a MovementClassifier.
    
    Args:
        lift_type: "snatch", "jerk", or "long_cycle"
        fps: Video frame rate
        dominant_hand: For snatch only - "left" or "right"
        
    Returns:
        MovementClassifier instance
    """
    return MovementClassifier(
        lift_type=lift_type,
        fps=fps,
        dominant_hand=dominant_hand
    )

