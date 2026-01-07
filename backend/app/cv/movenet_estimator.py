"""
MoveNet Thunder pose estimation for kettlebell movement analysis.

MoveNet Thunder provides 17 keypoints with excellent temporal stability
and accurate wrist/elbow tracking - ideal for kettlebell sport analysis.

Keypoints:
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
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
class MoveNetPose:
    """
    Pose result from MoveNet Thunder.
    
    Provides accessors for kettlebell-relevant measurements.
    """
    frame_number: int
    timestamp: float
    keypoints: List[Keypoint] = field(default_factory=list)
    overall_confidence: float = 0.0
    
    @property
    def is_valid(self) -> bool:
        """Check if pose has minimum required keypoints."""
        if len(self.keypoints) < 17:
            return False
        # Need at least wrists and hips visible
        required = [
            MoveNetKeypoint.LEFT_WRIST, MoveNetKeypoint.RIGHT_WRIST,
            MoveNetKeypoint.LEFT_HIP, MoveNetKeypoint.RIGHT_HIP
        ]
        return all(self.keypoints[i].is_visible for i in required)
    
    @property
    def left_wrist(self) -> Optional[Keypoint]:
        if len(self.keypoints) > MoveNetKeypoint.LEFT_WRIST:
            return self.keypoints[MoveNetKeypoint.LEFT_WRIST]
        return None
    
    @property
    def right_wrist(self) -> Optional[Keypoint]:
        if len(self.keypoints) > MoveNetKeypoint.RIGHT_WRIST:
            return self.keypoints[MoveNetKeypoint.RIGHT_WRIST]
        return None
    
    @property
    def nose(self) -> Optional[Keypoint]:
        if len(self.keypoints) > MoveNetKeypoint.NOSE:
            return self.keypoints[MoveNetKeypoint.NOSE]
        return None
    
    @property
    def left_hip(self) -> Optional[Keypoint]:
        if len(self.keypoints) > MoveNetKeypoint.LEFT_HIP:
            return self.keypoints[MoveNetKeypoint.LEFT_HIP]
        return None
    
    @property
    def right_hip(self) -> Optional[Keypoint]:
        if len(self.keypoints) > MoveNetKeypoint.RIGHT_HIP:
            return self.keypoints[MoveNetKeypoint.RIGHT_HIP]
        return None
    
    @property
    def left_shoulder(self) -> Optional[Keypoint]:
        if len(self.keypoints) > MoveNetKeypoint.LEFT_SHOULDER:
            return self.keypoints[MoveNetKeypoint.LEFT_SHOULDER]
        return None
    
    @property
    def right_shoulder(self) -> Optional[Keypoint]:
        if len(self.keypoints) > MoveNetKeypoint.RIGHT_SHOULDER:
            return self.keypoints[MoveNetKeypoint.RIGHT_SHOULDER]
        return None
    
    @property
    def left_elbow(self) -> Optional[Keypoint]:
        if len(self.keypoints) > MoveNetKeypoint.LEFT_ELBOW:
            return self.keypoints[MoveNetKeypoint.LEFT_ELBOW]
        return None
    
    @property
    def right_elbow(self) -> Optional[Keypoint]:
        if len(self.keypoints) > MoveNetKeypoint.RIGHT_ELBOW:
            return self.keypoints[MoveNetKeypoint.RIGHT_ELBOW]
        return None
    
    def get_wrist_height_ratio(self, side: str = "left") -> Optional[float]:
        """
        Get wrist height as ratio of body height.
        
        Returns value where 1.0 = wrist at nose level.
        Higher values = wrist above head (overhead).
        Lower values = wrist below hips (backswing).
        """
        wrist = self.left_wrist if side == "left" else self.right_wrist
        if not wrist or not self.nose or not self.left_hip or not self.right_hip:
            return None
        
        # Use nose as top reference, hip midpoint as bottom
        hip_y = (self.left_hip.y + self.right_hip.y) / 2
        nose_y = self.nose.y
        
        # In image coords, y increases downward (0 at top)
        body_height = hip_y - nose_y  # Positive value
        if abs(body_height) < 0.01:
            return None
        
        # Wrist position relative to hip
        wrist_from_hip = hip_y - wrist.y  # Positive = above hip
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
        
        # Calculate vectors
        v1 = np.array([shoulder.x - elbow.x, shoulder.y - elbow.y])
        v2 = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg


class MoveNetEstimator:
    """
    MoveNet Thunder pose estimator with temporal smoothing.
    
    Features:
    - 17 keypoints with excellent temporal consistency
    - Temporal smoothing for stable tracking
    - Optimized for CPU inference
    """
    
    INPUT_SIZE = 256
    CONFIDENCE_THRESHOLD = 0.35
    SMOOTHING_WINDOW = 7  # Frames for temporal smoothing
    
    def __init__(self):
        """Initialize MoveNet Thunder model."""
        logger.info("Loading MoveNet Thunder model...")
        
        # Load from TensorFlow Hub
        self.model = hub.load('https://tfhub.dev/google/movenet/singlepose/thunder/4')
        self.movenet = self.model.signatures['serving_default']
        
        # Temporal smoothing buffers
        self._keypoint_history: Deque[np.ndarray] = deque(maxlen=self.SMOOTHING_WINDOW)
        
        logger.info("MoveNet Thunder loaded successfully")
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float
    ) -> MoveNetPose:
        """
        Process a single frame and extract pose.
        
        Args:
            frame: BGR image from OpenCV
            frame_number: Current frame index
            timestamp: Timestamp in seconds
            
        Returns:
            MoveNetPose with 17 keypoints
        """
        # Preprocess frame
        input_image = self._preprocess(frame)
        
        # Run inference
        outputs = self.movenet(input_image)
        keypoints_with_scores = outputs['output_0'].numpy()[0, 0]  # Shape: (17, 3)
        
        # Apply temporal smoothing
        smoothed_keypoints = self._smooth_keypoints(keypoints_with_scores)
        
        # Convert to Keypoint objects
        keypoints = []
        total_conf = 0.0
        
        for i in range(17):
            y, x, conf = smoothed_keypoints[i]
            keypoints.append(Keypoint(x=x, y=y, confidence=conf))
            total_conf += conf
        
        overall_conf = total_conf / 17
        
        return MoveNetPose(
            frame_number=frame_number,
            timestamp=timestamp,
            keypoints=keypoints,
            overall_confidence=overall_conf
        )
    
    def _preprocess(self, frame: np.ndarray) -> tf.Tensor:
        """Preprocess frame for MoveNet input."""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to 256x256
        resized = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
        
        # Convert to tensor
        input_image = tf.cast(resized, dtype=tf.int32)
        input_image = tf.expand_dims(input_image, axis=0)
        
        return input_image
    
    def _smooth_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to keypoints."""
        self._keypoint_history.append(keypoints.copy())
        
        if len(self._keypoint_history) < 3:
            return keypoints
        
        # Weighted average favoring recent frames
        weights = np.array([0.5 ** (len(self._keypoint_history) - 1 - i) 
                          for i in range(len(self._keypoint_history))])
        weights /= weights.sum()
        
        smoothed = np.zeros_like(keypoints)
        for i, kp in enumerate(self._keypoint_history):
            smoothed += weights[i] * kp
        
        # Keep original confidence scores (don't smooth those)
        smoothed[:, 2] = keypoints[:, 2]
        
        return smoothed
    
    def close(self):
        """Cleanup resources."""
        self._keypoint_history.clear()


# Alias for compatibility with existing code
PoseKeypoints = MoveNetPose

