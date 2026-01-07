"""
Hybrid Pose Estimator: MoveNet Thunder + MediaPipe Hands

Combines:
- MoveNet Thunder: Body pose (17 keypoints) with temporal stability
- MediaPipe Hands: Precise wrist tracking (21 keypoints per hand)

This hybrid approach provides the best accuracy for kettlebell sport:
- Body tracking for movement phases (backswing, ascent)
- Precise wrist tracking for fixation detection
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Deque
from collections import deque
from enum import IntEnum
import logging

import cv2
import tensorflow as tf
import tensorflow_hub as hub
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logger = logging.getLogger(__name__)


class MoveNetKeypoint(IntEnum):
    """MoveNet keypoint indices."""
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


@dataclass
class Keypoint:
    """Single keypoint with position and confidence."""
    x: float  # Normalized x (0-1)
    y: float  # Normalized y (0-1)
    confidence: float
    
    @property
    def is_visible(self) -> bool:
        return self.confidence > 0.35


@dataclass 
class HandData:
    """Precise hand tracking data from MediaPipe Hands."""
    wrist: Keypoint  # Primary wrist position
    palm_center: Optional[Keypoint] = None
    index_tip: Optional[Keypoint] = None
    
    # Wrist stability metrics
    velocity: float = 0.0  # Wrist movement velocity
    is_stable: bool = False  # True if wrist is stable (for fixation)
    
    # Hand orientation
    palm_facing: str = "unknown"  # "up", "down", "forward", "back"


@dataclass
class HybridPose:
    """
    Combined pose from MoveNet + MediaPipe Hands.
    
    Provides both body tracking and precise hand data.
    """
    frame_number: int
    timestamp: float
    
    # Body keypoints from MoveNet (17 points)
    body_keypoints: List[Keypoint] = field(default_factory=list)
    overall_confidence: float = 0.0
    
    # Precise hand data from MediaPipe Hands
    left_hand: Optional[HandData] = None
    right_hand: Optional[HandData] = None
    
    # Wrist stability tracking (for fixation)
    left_wrist_stable_frames: int = 0
    right_wrist_stable_frames: int = 0
    
    @property
    def is_valid(self) -> bool:
        """Check if pose has minimum required keypoints."""
        if len(self.body_keypoints) < 17:
            return False
        required = [
            MoveNetKeypoint.LEFT_WRIST, MoveNetKeypoint.RIGHT_WRIST,
            MoveNetKeypoint.LEFT_HIP, MoveNetKeypoint.RIGHT_HIP
        ]
        return all(self.body_keypoints[i].is_visible for i in required)
    
    # Body keypoint accessors
    @property
    def nose(self) -> Optional[Keypoint]:
        return self.body_keypoints[MoveNetKeypoint.NOSE] if len(self.body_keypoints) > 0 else None
    
    @property
    def left_shoulder(self) -> Optional[Keypoint]:
        return self.body_keypoints[MoveNetKeypoint.LEFT_SHOULDER] if len(self.body_keypoints) > 5 else None
    
    @property
    def right_shoulder(self) -> Optional[Keypoint]:
        return self.body_keypoints[MoveNetKeypoint.RIGHT_SHOULDER] if len(self.body_keypoints) > 6 else None
    
    @property
    def left_elbow(self) -> Optional[Keypoint]:
        return self.body_keypoints[MoveNetKeypoint.LEFT_ELBOW] if len(self.body_keypoints) > 7 else None
    
    @property
    def right_elbow(self) -> Optional[Keypoint]:
        return self.body_keypoints[MoveNetKeypoint.RIGHT_ELBOW] if len(self.body_keypoints) > 8 else None
    
    @property
    def left_hip(self) -> Optional[Keypoint]:
        return self.body_keypoints[MoveNetKeypoint.LEFT_HIP] if len(self.body_keypoints) > 11 else None
    
    @property
    def right_hip(self) -> Optional[Keypoint]:
        return self.body_keypoints[MoveNetKeypoint.RIGHT_HIP] if len(self.body_keypoints) > 12 else None
    
    @property
    def left_wrist(self) -> Optional[Keypoint]:
        """Get left wrist - use MediaPipe Hands if available, else MoveNet."""
        if self.left_hand and self.left_hand.wrist:
            return self.left_hand.wrist
        return self.body_keypoints[MoveNetKeypoint.LEFT_WRIST] if len(self.body_keypoints) > 9 else None
    
    @property
    def right_wrist(self) -> Optional[Keypoint]:
        """Get right wrist - use MediaPipe Hands if available, else MoveNet."""
        if self.right_hand and self.right_hand.wrist:
            return self.right_hand.wrist
        return self.body_keypoints[MoveNetKeypoint.RIGHT_WRIST] if len(self.body_keypoints) > 10 else None
    
    def get_wrist_height_ratio(self, side: str = "left") -> Optional[float]:
        """
        Get wrist height as ratio of body height.
        
        Returns value where 1.0 = wrist at nose level.
        Higher values = wrist above head (overhead).
        """
        wrist = self.left_wrist if side == "left" else self.right_wrist
        if not wrist or not self.nose or not self.left_hip or not self.right_hip:
            return None
        
        hip_y = (self.left_hip.y + self.right_hip.y) / 2
        nose_y = self.nose.y
        
        body_height = hip_y - nose_y
        if abs(body_height) < 0.01:
            return None
        
        wrist_from_hip = hip_y - wrist.y
        ratio = wrist_from_hip / body_height
        
        return ratio
    
    def get_elbow_angle(self, side: str = "left") -> Optional[float]:
        """Calculate elbow angle in degrees (180 = fully extended)."""
        if side == "left":
            shoulder = self.left_shoulder
            elbow = self.left_elbow
            wrist = self.left_wrist
        else:
            shoulder = self.right_shoulder
            elbow = self.right_elbow
            wrist = self.right_wrist
        
        if not all([shoulder, elbow, wrist]):
            return None
        if not all([shoulder.is_visible, elbow.is_visible, wrist.is_visible]):
            return None
        
        v1 = np.array([shoulder.x - elbow.x, shoulder.y - elbow.y])
        v2 = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return float(np.degrees(angle_rad))
    
    def is_in_fixation(self, side: str = "left", min_stable_frames: int = 3) -> bool:
        """Check if wrist is in stable fixation position."""
        stable_frames = self.left_wrist_stable_frames if side == "left" else self.right_wrist_stable_frames
        hand_data = self.left_hand if side == "left" else self.right_hand
        
        if hand_data and hand_data.is_stable:
            return stable_frames >= min_stable_frames
        return False


class HybridEstimator:
    """
    Hybrid pose estimator combining MoveNet Thunder + MediaPipe Hands.
    
    Strategy:
    1. Run MoveNet on every frame for body tracking
    2. Run MediaPipe Hands when wrist is in overhead zone
    3. Track wrist stability for fixation detection
    """
    
    MOVENET_INPUT_SIZE = 256
    OVERHEAD_THRESHOLD = 0.9  # Wrist height ratio to trigger hand tracking
    WRIST_STABILITY_THRESHOLD = 0.015  # Movement threshold for "stable"
    SMOOTHING_WINDOW = 7
    
    def __init__(self):
        """Initialize both models."""
        logger.info("Loading Hybrid Estimator (MoveNet Thunder + MediaPipe Hands)...")
        
        # Load MoveNet Thunder
        self.movenet_model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/4')
        self.movenet = self.movenet_model.signatures['serving_default']
        logger.info("MoveNet Thunder loaded")
        
        # Load MediaPipe Hands using new Tasks API
        # Download model if needed
        import os
        import urllib.request
        
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'hand_landmarker.task')
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            logger.info("Downloading Hand Landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            logger.info("Hand Landmarker model downloaded")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        logger.info("MediaPipe Hands loaded")
        
        # Temporal smoothing
        self._body_history: Deque[np.ndarray] = deque(maxlen=self.SMOOTHING_WINDOW)
        
        # Wrist tracking for stability
        self._left_wrist_history: Deque[Tuple[float, float]] = deque(maxlen=10)
        self._right_wrist_history: Deque[Tuple[float, float]] = deque(maxlen=10)
        self._left_stable_count = 0
        self._right_stable_count = 0
        
        logger.info("Hybrid Estimator ready")
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float
    ) -> HybridPose:
        """
        Process frame with both MoveNet and MediaPipe Hands.
        
        Args:
            frame: BGR image from OpenCV
            frame_number: Current frame index
            timestamp: Timestamp in seconds
            
        Returns:
            HybridPose with body and hand data
        """
        # 1. Run MoveNet for body pose
        body_keypoints = self._run_movenet(frame)
        
        # 2. Check if wrists are in overhead zone
        left_height = self._get_wrist_height(body_keypoints, "left")
        right_height = self._get_wrist_height(body_keypoints, "right")
        
        run_hands = (
            (left_height is not None and left_height > self.OVERHEAD_THRESHOLD) or
            (right_height is not None and right_height > self.OVERHEAD_THRESHOLD)
        )
        
        # 3. Run MediaPipe Hands if in overhead zone
        left_hand_data = None
        right_hand_data = None
        
        if run_hands:
            left_hand_data, right_hand_data = self._run_mediapipe_hands(frame)
        
        # 4. Track wrist stability
        self._update_wrist_stability(body_keypoints, left_hand_data, right_hand_data)
        
        # 5. Build result
        keypoints = []
        total_conf = 0.0
        
        for i in range(17):
            y, x, conf = body_keypoints[i]
            keypoints.append(Keypoint(x=x, y=y, confidence=conf))
            total_conf += conf
        
        return HybridPose(
            frame_number=frame_number,
            timestamp=timestamp,
            body_keypoints=keypoints,
            overall_confidence=total_conf / 17,
            left_hand=left_hand_data,
            right_hand=right_hand_data,
            left_wrist_stable_frames=self._left_stable_count,
            right_wrist_stable_frames=self._right_stable_count
        )
    
    def _run_movenet(self, frame: np.ndarray) -> np.ndarray:
        """Run MoveNet Thunder inference."""
        # Preprocess
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.MOVENET_INPUT_SIZE, self.MOVENET_INPUT_SIZE))
        input_image = tf.cast(resized, dtype=tf.int32)
        input_image = tf.expand_dims(input_image, axis=0)
        
        # Inference
        outputs = self.movenet(input_image)
        keypoints = outputs['output_0'].numpy()[0, 0]
        
        # Temporal smoothing
        self._body_history.append(keypoints.copy())
        if len(self._body_history) >= 3:
            weights = np.array([0.5 ** (len(self._body_history) - 1 - i) 
                              for i in range(len(self._body_history))])
            weights /= weights.sum()
            
            smoothed = np.zeros_like(keypoints)
            for i, kp in enumerate(self._body_history):
                smoothed += weights[i] * kp
            smoothed[:, 2] = keypoints[:, 2]  # Keep original confidence
            return smoothed
        
        return keypoints
    
    def _run_mediapipe_hands(
        self, 
        frame: np.ndarray
    ) -> Tuple[Optional[HandData], Optional[HandData]]:
        """Run MediaPipe Hands for precise wrist tracking using Tasks API."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        results = self.hand_landmarker.detect(mp_image)
        
        left_hand = None
        right_hand = None
        
        if results.hand_landmarks and results.handedness:
            for hand_landmarks, handedness in zip(
                results.hand_landmarks, 
                results.handedness
            ):
                # Get hand label
                label = handedness[0].category_name
                score = handedness[0].score
                
                # Extract wrist (landmark 0)
                wrist_lm = hand_landmarks[0]
                wrist = Keypoint(
                    x=wrist_lm.x,
                    y=wrist_lm.y,
                    confidence=score
                )
                
                # Extract palm center (landmark 9 - middle finger MCP)
                palm_lm = hand_landmarks[9]
                palm = Keypoint(x=palm_lm.x, y=palm_lm.y, confidence=0.8)
                
                # Extract index tip (landmark 8)
                index_lm = hand_landmarks[8]
                index_tip = Keypoint(x=index_lm.x, y=index_lm.y, confidence=0.8)
                
                hand_data = HandData(
                    wrist=wrist,
                    palm_center=palm,
                    index_tip=index_tip
                )
                
                if label == "Left":
                    left_hand = hand_data
                else:
                    right_hand = hand_data
        
        return left_hand, right_hand
    
    def _get_wrist_height(self, keypoints: np.ndarray, side: str) -> Optional[float]:
        """Calculate wrist height from MoveNet keypoints."""
        wrist_idx = MoveNetKeypoint.LEFT_WRIST if side == "left" else MoveNetKeypoint.RIGHT_WRIST
        nose_idx = MoveNetKeypoint.NOSE
        hip_l_idx = MoveNetKeypoint.LEFT_HIP
        hip_r_idx = MoveNetKeypoint.RIGHT_HIP
        
        wrist_y = keypoints[wrist_idx][0]
        nose_y = keypoints[nose_idx][0]
        hip_y = (keypoints[hip_l_idx][0] + keypoints[hip_r_idx][0]) / 2
        
        body_height = hip_y - nose_y
        if abs(body_height) < 0.01:
            return None
        
        wrist_from_hip = hip_y - wrist_y
        return wrist_from_hip / body_height
    
    def _update_wrist_stability(
        self,
        body_keypoints: np.ndarray,
        left_hand: Optional[HandData],
        right_hand: Optional[HandData]
    ):
        """Track wrist stability for fixation detection."""
        # Get current wrist positions
        left_wrist = body_keypoints[MoveNetKeypoint.LEFT_WRIST][:2]
        right_wrist = body_keypoints[MoveNetKeypoint.RIGHT_WRIST][:2]
        
        # Use hand data if available (more precise)
        if left_hand and left_hand.wrist:
            left_wrist = (left_hand.wrist.y, left_hand.wrist.x)
        if right_hand and right_hand.wrist:
            right_wrist = (right_hand.wrist.y, right_hand.wrist.x)
        
        # Calculate velocity
        left_velocity = 0.0
        right_velocity = 0.0
        
        if self._left_wrist_history:
            prev = self._left_wrist_history[-1]
            left_velocity = np.sqrt((left_wrist[0] - prev[0])**2 + (left_wrist[1] - prev[1])**2)
        
        if self._right_wrist_history:
            prev = self._right_wrist_history[-1]
            right_velocity = np.sqrt((right_wrist[0] - prev[0])**2 + (right_wrist[1] - prev[1])**2)
        
        # Update history
        self._left_wrist_history.append(tuple(left_wrist))
        self._right_wrist_history.append(tuple(right_wrist))
        
        # Update stability counts
        if left_velocity < self.WRIST_STABILITY_THRESHOLD:
            self._left_stable_count += 1
            if left_hand:
                left_hand.is_stable = True
                left_hand.velocity = left_velocity
        else:
            self._left_stable_count = 0
            if left_hand:
                left_hand.is_stable = False
                left_hand.velocity = left_velocity
        
        if right_velocity < self.WRIST_STABILITY_THRESHOLD:
            self._right_stable_count += 1
            if right_hand:
                right_hand.is_stable = True
                right_hand.velocity = right_velocity
        else:
            self._right_stable_count = 0
            if right_hand:
                right_hand.is_stable = False
                right_hand.velocity = right_velocity
    
    def close(self):
        """Cleanup resources."""
        # HandLandmarker doesn't need explicit close
        self._body_history.clear()
        self._left_wrist_history.clear()
        self._right_wrist_history.clear()


# Aliases for compatibility
PoseKeypoints = HybridPose
MoveNetPose = HybridPose

