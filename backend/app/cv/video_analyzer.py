"""
Pre-analysis video analyzer with auto-crop and quality validation.

Performs comprehensive initial pass before pose inference:
1. Detect video properties (resolution, FPS, duration)
2. Validate video quality (blur, FPS, brightness)
3. Detect athlete bounding box and check for truncation
4. Estimate camera angle
5. Auto-crop to ensure athlete fills 70-85% of frame
6. Generate warnings/rejections for quality issues

VIDEO QUALITY REQUIREMENTS:
- FPS >= 24 (recommended), >= 15 (minimum)
- Athlete >= 50% of frame height
- No severe motion blur (Laplacian variance > 100)
- Head and feet visible (not cropped)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CameraAngle(Enum):
    """Detected camera angle classification."""
    FRONT = "front"
    SIDE = "side"
    DIAGONAL = "diagonal"
    UNKNOWN = "unknown"


class VideoQuality(Enum):
    """Overall video quality assessment."""
    GOOD = "good"           # All checks passed
    ACCEPTABLE = "acceptable"  # Minor issues, can process
    POOR = "poor"           # Significant issues, accuracy reduced
    UNUSABLE = "unusable"   # Cannot reliably process


@dataclass
class VideoMetadata:
    """Video properties from pre-analysis."""
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    
    # Quality metrics
    resolution_ok: bool = True
    blur_score: float = 0.0
    blur_ok: bool = True
    brightness_avg: float = 0.0
    brightness_ok: bool = True
    fps_ok: bool = True
    
    # Athlete detection
    athlete_height_ratio: float = 0.0  # Height of athlete / frame height
    athlete_detected: bool = False
    needs_crop: bool = False
    confidence_downgrade: bool = False
    
    # Truncation detection
    head_visible: bool = True
    feet_visible: bool = True
    is_truncated: bool = False
    
    # Camera angle
    camera_angle: CameraAngle = CameraAngle.UNKNOWN
    
    # Overall quality
    quality: VideoQuality = VideoQuality.GOOD
    quality_score: float = 1.0  # 0-1 overall quality score
    

@dataclass
class CropRegion:
    """Crop region for a frame."""
    x: int
    y: int
    width: int
    height: int
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply crop to frame."""
        return frame[self.y:self.y+self.height, self.x:self.x+self.width]


@dataclass
class AnalysisResult:
    """Result of pre-analysis pass."""
    metadata: VideoMetadata
    crop_regions: List[CropRegion] = field(default_factory=list)
    sample_frames: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    can_process: bool = True  # False if video is unusable


class VideoAnalyzer:
    """
    Pre-analysis video analyzer with comprehensive quality validation.
    
    Performs thorough initial pass to:
    1. Extract video metadata
    2. Validate video quality (FPS, blur, brightness)
    3. Detect athlete bounding box
    4. Check for head/feet truncation
    5. Determine if cropping is needed
    6. Estimate camera angle
    7. Calculate overall quality score
    """
    
    # Analysis settings
    SAMPLE_INTERVAL_SECONDS = 1.0  # Sample every 1 second for better accuracy
    MIN_ATHLETE_HEIGHT_RATIO = 0.50  # Athlete must be at least 50% of frame (strict)
    OPTIMAL_ATHLETE_HEIGHT_RATIO = 0.75  # Target 70-85% of frame
    CROP_MARGIN = 0.15  # 15% margin around athlete
    
    # Quality thresholds
    MIN_FPS_RECOMMENDED = 24.0  # Recommended minimum FPS
    MIN_FPS_ACCEPTABLE = 15.0   # Absolute minimum FPS
    MIN_BLUR_SCORE = 100.0      # Laplacian variance threshold
    MIN_BRIGHTNESS = 40.0       # Minimum average brightness (0-255)
    MAX_BRIGHTNESS = 220.0      # Maximum average brightness (0-255)
    
    # Truncation detection
    HEAD_MARGIN_PIXELS = 20     # Minimum pixels from top for head
    FEET_MARGIN_PIXELS = 20     # Minimum pixels from bottom for feet
    
    def __init__(self):
        # Person detector (HOG-based, fast)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
    def analyze(self, video_path: str) -> AnalysisResult:
        """
        Perform comprehensive pre-analysis on video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            AnalysisResult with metadata, crop regions, and quality assessment
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return AnalysisResult(
                metadata=VideoMetadata(0, 0, 0, 0, 0),
                warnings=[],
                errors=[f"Cannot open video: {video_path}"],
                can_process=False
            )
        
        try:
            # Get basic metadata
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            metadata = VideoMetadata(
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration_seconds=duration,
                resolution_ok=height >= 720
            )
            
            warnings = []
            errors = []
            quality_score = 1.0
            can_process = True
            
            # =====================================================
            # CHECK 1: Resolution
            # =====================================================
            if height < 480:
                errors.append(f"Resolution too low: {height}p (minimum: 480p)")
                quality_score -= 0.3
            elif height < 720:
                warnings.append(f"Low resolution: {height}p (recommended: 720p+)")
                quality_score -= 0.1
            
            # =====================================================
            # CHECK 2: Frame Rate
            # =====================================================
            if fps < self.MIN_FPS_ACCEPTABLE:
                errors.append(f"FPS too low: {fps:.1f} (minimum: {self.MIN_FPS_ACCEPTABLE})")
                metadata.fps_ok = False
                quality_score -= 0.4
                can_process = False  # Can't reliably detect reps at low FPS
            elif fps < self.MIN_FPS_RECOMMENDED:
                warnings.append(f"Low FPS: {fps:.1f} (recommended: {self.MIN_FPS_RECOMMENDED}+)")
                metadata.fps_ok = False
                quality_score -= 0.15
            else:
                metadata.fps_ok = True
            
            # Sample frames for analysis
            sample_interval = max(1, int(fps * self.SAMPLE_INTERVAL_SECONDS))
            sample_frames = list(range(0, total_frames, sample_interval))
            
            # Collect bounding boxes and quality metrics
            bboxes = []
            bbox_positions = []  # Store (bbox, y_top, y_bottom) for truncation check
            blur_scores = []
            brightness_values = []
            
            for frame_idx in sample_frames[:50]:  # Analyze first 50 samples
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Quality metrics
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
                brightness_values.append(np.mean(gray))
                
                # Detect person
                bbox = self._detect_person(frame)
                if bbox is not None:
                    bboxes.append(bbox)
                    x, y, w, h = bbox
                    bbox_positions.append((y, y + h))  # top, bottom
            
            # =====================================================
            # CHECK 3: Motion Blur
            # =====================================================
            if blur_scores:
                metadata.blur_score = float(np.mean(blur_scores))
                min_blur = float(np.min(blur_scores))
                
                if metadata.blur_score < self.MIN_BLUR_SCORE:
                    warnings.append(f"Video appears blurry (score: {metadata.blur_score:.0f}, "
                                  f"threshold: {self.MIN_BLUR_SCORE})")
                    metadata.blur_ok = False
                    quality_score -= 0.2
                elif min_blur < self.MIN_BLUR_SCORE * 0.5:
                    warnings.append(f"Some frames are significantly blurry")
                    quality_score -= 0.1
                else:
                    metadata.blur_ok = True
            
            # =====================================================
            # CHECK 4: Brightness/Lighting
            # =====================================================
            if brightness_values:
                metadata.brightness_avg = float(np.mean(brightness_values))
                
                if metadata.brightness_avg < self.MIN_BRIGHTNESS:
                    warnings.append(f"Video too dark (brightness: {metadata.brightness_avg:.0f})")
                    metadata.brightness_ok = False
                    quality_score -= 0.15
                elif metadata.brightness_avg > self.MAX_BRIGHTNESS:
                    warnings.append(f"Video overexposed (brightness: {metadata.brightness_avg:.0f})")
                    metadata.brightness_ok = False
                    quality_score -= 0.15
                else:
                    metadata.brightness_ok = True
            
            # =====================================================
            # CHECK 5: Athlete Detection & Size
            # =====================================================
            crop_regions = []
            if bboxes:
                metadata.athlete_detected = True
                
                # Compute median bounding box (robust to outliers)
                median_bbox = self._compute_median_bbox(bboxes, width, height)
                athlete_height = median_bbox[3]
                metadata.athlete_height_ratio = athlete_height / height
                
                logger.info(f"Athlete height ratio: {metadata.athlete_height_ratio:.2%}")
                
                # Check if athlete is too small
                if metadata.athlete_height_ratio < self.MIN_ATHLETE_HEIGHT_RATIO:
                    warnings.append(f"Athlete too small in frame: {metadata.athlete_height_ratio:.0%} "
                                  f"(minimum: {self.MIN_ATHLETE_HEIGHT_RATIO:.0%})")
                    metadata.needs_crop = True
                    quality_score -= 0.25
                    
                    crop_region = self._compute_crop_region(median_bbox, width, height)
                    crop_regions = [crop_region]
                    
                    # Check if still too small after crop
                    new_ratio = median_bbox[3] / crop_region.height
                    if new_ratio < 0.40:
                        metadata.confidence_downgrade = True
                        errors.append("Athlete too small even after cropping - accuracy severely limited")
                        quality_score -= 0.2
                
                # =====================================================
                # CHECK 6: Head/Feet Truncation
                # =====================================================
                if bbox_positions:
                    tops = [pos[0] for pos in bbox_positions]
                    bottoms = [pos[1] for pos in bbox_positions]
                    
                    median_top = np.median(tops)
                    median_bottom = np.median(bottoms)
                    
                    # Check head visibility
                    if median_top < self.HEAD_MARGIN_PIXELS:
                        metadata.head_visible = False
                        metadata.is_truncated = True
                        warnings.append("Head may be cropped from frame - affects lockout detection")
                        quality_score -= 0.15
                    
                    # Check feet visibility  
                    if median_bottom > height - self.FEET_MARGIN_PIXELS:
                        metadata.feet_visible = False
                        metadata.is_truncated = True
                        warnings.append("Feet may be cropped from frame - affects backswing detection")
                        quality_score -= 0.1
                
                # Detect camera angle
                metadata.camera_angle = self._detect_camera_angle(bboxes)
                
            else:
                errors.append("Could not detect athlete in video")
                metadata.athlete_detected = False
                metadata.confidence_downgrade = True
                quality_score -= 0.5
            
            # =====================================================
            # FINAL: Calculate Overall Quality
            # =====================================================
            metadata.quality_score = max(0.0, min(1.0, quality_score))
            
            if metadata.quality_score >= 0.8:
                metadata.quality = VideoQuality.GOOD
            elif metadata.quality_score >= 0.5:
                metadata.quality = VideoQuality.ACCEPTABLE
            elif metadata.quality_score >= 0.3:
                metadata.quality = VideoQuality.POOR
                metadata.confidence_downgrade = True
            else:
                metadata.quality = VideoQuality.UNUSABLE
                can_process = False
            
            logger.info(f"Pre-analysis complete: quality={metadata.quality.value} "
                       f"(score={metadata.quality_score:.2f}), "
                       f"{metadata.camera_angle.value} view, "
                       f"crop needed: {metadata.needs_crop}")
            
            return AnalysisResult(
                metadata=metadata,
                crop_regions=crop_regions,
                sample_frames=sample_frames,
                warnings=warnings,
                errors=errors,
                can_process=can_process
            )
            
        finally:
            cap.release()
    
    def _detect_person(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect person bounding box in frame.
        
        Returns:
            (x, y, width, height) or None
        """
        # Resize for faster detection
        scale = 400 / max(frame.shape[:2])
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Detect using HOG
        rects, weights = self.hog.detectMultiScale(
            small, 
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        
        if len(rects) == 0:
            return None
        
        # Take largest detection (assume single athlete)
        areas = [w * h for (x, y, w, h) in rects]
        best_idx = np.argmax(areas)
        x, y, w, h = rects[best_idx]
        
        # Scale back to original size
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)
        
        return (x, y, w, h)
    
    def _compute_median_bbox(
        self,
        bboxes: List[Tuple[int, int, int, int]],
        frame_width: int,
        frame_height: int
    ) -> Tuple[int, int, int, int]:
        """Compute median bounding box from samples."""
        xs = [b[0] for b in bboxes]
        ys = [b[1] for b in bboxes]
        ws = [b[2] for b in bboxes]
        hs = [b[3] for b in bboxes]
        
        return (
            int(np.median(xs)),
            int(np.median(ys)),
            int(np.median(ws)),
            int(np.median(hs))
        )
    
    def _compute_crop_region(
        self,
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int
    ) -> CropRegion:
        """
        Compute crop region to make athlete 70-85% of frame.
        
        Ensures head and feet are included with margin.
        """
        x, y, w, h = bbox
        
        # Calculate center
        cx = x + w // 2
        cy = y + h // 2
        
        # Target size: make athlete height ~75% of crop height
        target_height = int(h / self.OPTIMAL_ATHLETE_HEIGHT_RATIO)
        target_width = int(target_height * (frame_width / frame_height))
        
        # Add margin
        margin_h = int(target_height * self.CROP_MARGIN)
        margin_w = int(target_width * self.CROP_MARGIN)
        
        crop_height = target_height + 2 * margin_h
        crop_width = target_width + 2 * margin_w
        
        # Center crop on athlete
        crop_x = cx - crop_width // 2
        crop_y = cy - crop_height // 2
        
        # Clamp to frame bounds
        crop_x = max(0, min(crop_x, frame_width - crop_width))
        crop_y = max(0, min(crop_y, frame_height - crop_height))
        crop_width = min(crop_width, frame_width - crop_x)
        crop_height = min(crop_height, frame_height - crop_y)
        
        return CropRegion(
            x=crop_x,
            y=crop_y,
            width=crop_width,
            height=crop_height
        )
    
    def _detect_camera_angle(
        self,
        bboxes: List[Tuple[int, int, int, int]]
    ) -> CameraAngle:
        """
        Estimate camera angle from bounding box aspect ratios.
        
        Side view: narrower bounding box (w/h smaller)
        Front view: wider bounding box (w/h larger)
        """
        if not bboxes:
            return CameraAngle.UNKNOWN
        
        aspect_ratios = [w / h for (x, y, w, h) in bboxes if h > 0]
        if not aspect_ratios:
            return CameraAngle.UNKNOWN
        
        median_ratio = np.median(aspect_ratios)
        
        if median_ratio < 0.35:
            return CameraAngle.SIDE
        elif median_ratio > 0.50:
            return CameraAngle.FRONT
        else:
            return CameraAngle.DIAGONAL

