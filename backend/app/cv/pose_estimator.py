"""
Pose estimation using MediaPipe for kettlebell movement analysis.

This module extracts 33 body keypoints from video frames and provides
structured access to joint positions for biomechanical analysis.

Updated for MediaPipe 0.10.30+ Tasks API.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import IntEnum

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipeLandmark(IntEnum):
    """MediaPipe Pose landmark indices for quick reference."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


@dataclass
class Keypoint:
    """Single keypoint with 3D position and visibility."""
    x: float  # Normalized x coordinate (0-1)
    y: float  # Normalized y coordinate (0-1)
    z: float  # Depth relative to hips
    visibility: float  # Confidence score (0-1)
    
    @property
    def is_visible(self) -> bool:
        """Check if keypoint has sufficient visibility."""
        return self.visibility > 0.5
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z])
    
    def distance_to(self, other: "Keypoint") -> float:
        """Calculate Euclidean distance to another keypoint."""
        return np.linalg.norm(self.to_array() - other.to_array())


@dataclass
class PoseKeypoints:
    """
    Complete pose estimation result for a single frame.
    
    Provides convenient accessors for kettlebell-relevant joints
    and methods for biomechanical calculations.
    """
    frame_number: int
    timestamp: float
    landmarks: List[Keypoint] = field(default_factory=list)
    overall_confidence: float = 0.0
    
    @classmethod
    def from_mediapipe_tasks(
        cls,
        result,
        frame_number: int,
        timestamp: float,
    ) -> "PoseKeypoints":
        """Create PoseKeypoints from MediaPipe Tasks result."""
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return cls(
                frame_number=frame_number,
                timestamp=timestamp,
                landmarks=[],
                overall_confidence=0.0
            )
        
        # Use first detected pose
        pose_landmarks = result.pose_landmarks[0]
        
        landmarks = []
        visibility_sum = 0.0
        
        for landmark in pose_landmarks:
            # Tasks API uses visibility and presence
            vis = landmark.visibility if hasattr(landmark, 'visibility') else 0.5
            kp = Keypoint(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=vis
            )
            landmarks.append(kp)
            visibility_sum += vis
        
        overall_confidence = visibility_sum / len(landmarks) if landmarks else 0.0
        
        return cls(
            frame_number=frame_number,
            timestamp=timestamp,
            landmarks=landmarks,
            overall_confidence=overall_confidence
        )
    
    @property
    def is_valid(self) -> bool:
        """Check if pose was detected."""
        return len(self.landmarks) > 0
    
    # Convenient accessors for kettlebell-relevant keypoints
    @property
    def left_shoulder(self) -> Optional[Keypoint]:
        return self._get_landmark(MediaPipeLandmark.LEFT_SHOULDER)
    
    @property
    def right_shoulder(self) -> Optional[Keypoint]:
        return self._get_landmark(MediaPipeLandmark.RIGHT_SHOULDER)
    
    @property
    def left_elbow(self) -> Optional[Keypoint]:
        return self._get_landmark(MediaPipeLandmark.LEFT_ELBOW)
    
    @property
    def right_elbow(self) -> Optional[Keypoint]:
        return self._get_landmark(MediaPipeLandmark.RIGHT_ELBOW)
    
    @property
    def left_wrist(self) -> Optional[Keypoint]:
        return self._get_landmark(MediaPipeLandmark.LEFT_WRIST)
    
    @property
    def right_wrist(self) -> Optional[Keypoint]:
        return self._get_landmark(MediaPipeLandmark.RIGHT_WRIST)
    
    @property
    def left_hip(self) -> Optional[Keypoint]:
        return self._get_landmark(MediaPipeLandmark.LEFT_HIP)
    
    @property
    def right_hip(self) -> Optional[Keypoint]:
        return self._get_landmark(MediaPipeLandmark.RIGHT_HIP)
    
    @property
    def nose(self) -> Optional[Keypoint]:
        return self._get_landmark(MediaPipeLandmark.NOSE)
    
    def _get_landmark(self, index: int) -> Optional[Keypoint]:
        """Safely get landmark by index."""
        if 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        return None
    
    def get_elbow_angle(self, side: str = "left") -> Optional[float]:
        """
        Calculate elbow angle (shoulder-elbow-wrist) in degrees.
        
        Returns angle where 180 = fully extended.
        """
        if side == "left":
            shoulder, elbow, wrist = self.left_shoulder, self.left_elbow, self.left_wrist
        else:
            shoulder, elbow, wrist = self.right_shoulder, self.right_elbow, self.right_wrist
        
        if not all([shoulder, elbow, wrist]):
            return None
        
        if not all([shoulder.is_visible, elbow.is_visible, wrist.is_visible]):
            return None
        
        return self._calculate_angle(shoulder, elbow, wrist)
    
    def get_shoulder_angle(self, side: str = "left") -> Optional[float]:
        """
        Calculate shoulder angle (hip-shoulder-elbow) in degrees.
        
        Returns angle where 180 = arm straight up.
        """
        if side == "left":
            hip, shoulder, elbow = self.left_hip, self.left_shoulder, self.left_elbow
        else:
            hip, shoulder, elbow = self.right_hip, self.right_shoulder, self.right_elbow
        
        if not all([hip, shoulder, elbow]):
            return None
        
        if not all([hip.is_visible, shoulder.is_visible, elbow.is_visible]):
            return None
        
        return self._calculate_angle(hip, shoulder, elbow)
    
    def get_torso_lean_angle(self) -> Optional[float]:
        """
        Calculate torso lean from vertical in degrees.
        
        Uses midpoint of hips and midpoint of shoulders.
        """
        if not all([self.left_hip, self.right_hip, self.left_shoulder, self.right_shoulder]):
            return None
        
        # Calculate midpoints
        hip_mid = np.array([
            (self.left_hip.x + self.right_hip.x) / 2,
            (self.left_hip.y + self.right_hip.y) / 2
        ])
        shoulder_mid = np.array([
            (self.left_shoulder.x + self.right_shoulder.x) / 2,
            (self.left_shoulder.y + self.right_shoulder.y) / 2
        ])
        
        # Vector from hip to shoulder
        torso_vec = shoulder_mid - hip_mid
        
        # Vertical vector (note: y increases downward in image coords)
        vertical = np.array([0, -1])
        
        # Calculate angle
        cos_angle = np.dot(torso_vec, vertical) / (
            np.linalg.norm(torso_vec) * np.linalg.norm(vertical)
        )
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def get_wrist_height_ratio(self, side: str = "left") -> Optional[float]:
        """
        Get wrist height as ratio of body height.
        
        Returns value where 1.0 = wrist at nose level (full overhead).
        Lower values indicate lower wrist position.
        """
        wrist = self.left_wrist if side == "left" else self.right_wrist
        if not wrist or not self.nose or not self.left_hip or not self.right_hip:
            return None
        
        # Use nose as top reference, hip midpoint as bottom
        hip_y = (self.left_hip.y + self.right_hip.y) / 2
        nose_y = self.nose.y
        
        # Note: in image coords, y increases downward
        body_height = hip_y - nose_y
        if abs(body_height) < 0.01:
            return None
        
        wrist_from_hip = hip_y - wrist.y
        ratio = wrist_from_hip / body_height
        
        return ratio
    
    def get_arm_symmetry_score(self) -> Optional[float]:
        """
        Calculate symmetry between left and right arm positions.
        
        Returns score 0-1 where 1.0 = perfectly symmetric.
        """
        left_elbow_angle = self.get_elbow_angle("left")
        right_elbow_angle = self.get_elbow_angle("right")
        
        if left_elbow_angle is None or right_elbow_angle is None:
            return None
        
        angle_diff = abs(left_elbow_angle - right_elbow_angle)
        # Max expected difference is ~30 degrees for asymmetry
        symmetry = max(0, 1 - (angle_diff / 30))
        
        return symmetry
    
    def _calculate_angle(
        self, 
        point_a: Keypoint, 
        point_b: Keypoint, 
        point_c: Keypoint
    ) -> float:
        """Calculate angle at point_b formed by points a, b, c."""
        a = np.array([point_a.x, point_a.y])
        b = np.array([point_b.x, point_b.y])
        c = np.array([point_c.x, point_c.y])
        
        ba = a - b
        bc = c - b
        
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary for storage."""
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "overall_confidence": self.overall_confidence,
            "landmarks": [
                {
                    "x": kp.x, 
                    "y": kp.y, 
                    "z": kp.z, 
                    "visibility": kp.visibility
                }
                for kp in self.landmarks
            ]
        }


# Get model path relative to this file
def get_model_path(complexity: int = 0) -> str:
    """
    Get the path to the pose landmarker model.
    
    Args:
        complexity: 0=lite (fastest), 1=full, 2=heavy (most accurate)
    """
    model_names = {
        0: "pose_landmarker_lite.task",
        1: "pose_landmarker_full.task", 
        2: "pose_landmarker_heavy.task",
    }
    model_name = model_names.get(complexity, "pose_landmarker_lite.task")
    
    # Try multiple locations
    base_dirs = [
        os.path.join(os.path.dirname(__file__), "..", "..", "models"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"),
        "/Users/channie/Desktop/kettlebell-counter/backend/models",
    ]
    
    for base_dir in base_dirs:
        path = os.path.abspath(os.path.join(base_dir, model_name))
        if os.path.exists(path):
            return path
    
    # Fallback to any available model
    for base_dir in base_dirs:
        for name in ["pose_landmarker_lite.task", "pose_landmarker_full.task", "pose_landmarker_heavy.task"]:
            path = os.path.abspath(os.path.join(base_dir, name))
            if os.path.exists(path):
                return path
    
    raise FileNotFoundError(
        f"Pose landmarker model not found. "
        "Download from: https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    )


class PoseEstimator:
    """
    Pose estimation engine using MediaPipe Tasks API (0.10.30+).
    
    Processes video frames and extracts body keypoints for
    kettlebell movement analysis.
    """
    
    def __init__(
        self,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_segmentation: bool = False
    ):
        """
        Initialize pose estimator.
        
        Args:
            model_complexity: 0=lite (fastest), 1=full, 2=heavy (most accurate)
            min_detection_confidence: Minimum confidence for initial detection
            min_tracking_confidence: Minimum confidence for tracking
            enable_segmentation: Not used in Tasks API
        """
        model_path = get_model_path(model_complexity)
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_poses=1,  # Single athlete per video
        )
        
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0
        
    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float
    ) -> PoseKeypoints:
        """
        Process a single frame and extract pose keypoints.
        
        Args:
            frame: BGR image from OpenCV
            frame_number: Frame index in video
            timestamp: Timestamp in seconds
            
        Returns:
            PoseKeypoints with all detected landmarks
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Timestamp must be monotonically increasing in VIDEO mode
        timestamp_ms = int(timestamp * 1000)
        if timestamp_ms <= self._frame_timestamp_ms:
            timestamp_ms = self._frame_timestamp_ms + 1
        self._frame_timestamp_ms = timestamp_ms
        
        # Process frame
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Extract keypoints
        keypoints = PoseKeypoints.from_mediapipe_tasks(
            result,
            frame_number=frame_number,
            timestamp=timestamp,
        )
        
        return keypoints
    
    def process_frames(
        self,
        frames: List[Tuple[np.ndarray, int, float]]
    ) -> List[PoseKeypoints]:
        """
        Process multiple frames.
        
        Args:
            frames: List of (frame, frame_number, timestamp) tuples
            
        Returns:
            List of PoseKeypoints for each frame
        """
        return [
            self.process_frame(frame, frame_num, timestamp)
            for frame, frame_num, timestamp in frames
        ]
    
    def close(self):
        """Release resources."""
        self.landmarker.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
