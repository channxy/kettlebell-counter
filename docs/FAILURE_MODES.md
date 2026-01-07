# Known Failure Modes and Mitigations

This document catalogs known failure modes in the kettlebell rep counting system and the mitigations implemented or recommended.

## Classification: Video Quality Issues

### 1. Low Resolution Video

**Severity**: High  
**Impact**: Pose keypoint detection becomes unreliable

**Symptoms**:
- High percentage of AMBIGUOUS classifications
- Inconsistent lockout angle measurements
- Keypoint "jumping" between frames

**Mitigation**:
```python
# In video_processor.py
MIN_RESOLUTION = (720, 480)  # Minimum supported

def validate_video_resolution(cap):
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    if width < MIN_RESOLUTION[0] or height < MIN_RESOLUTION[1]:
        logger.warning(f"Low resolution video: {width}x{height}")
        # Continue but flag for user
        return QualityWarning.LOW_RESOLUTION
```

**User Communication**:
- Display warning before processing
- Recommend 720p or higher

---

### 2. Poor Lighting

**Severity**: Medium  
**Impact**: Reduced pose detection confidence, especially for limb extremities

**Symptoms**:
- Low visibility scores on wrist/elbow keypoints
- Inconsistent detection across frames
- Shadows causing false occlusions

**Mitigation**:
- Use per-keypoint visibility thresholds
- Apply temporal smoothing for brief drops
- Mark extended low-visibility periods as AMBIGUOUS

---

### 3. Motion Blur

**Severity**: Medium  
**Impact**: Inaccurate keypoint positions during fast movements

**Symptoms**:
- Keypoints "lag" behind actual positions
- Peak detection may be off by 1-2 frames
- Lockout angles may appear lower than actual

**Mitigation**:
```python
# Use frame interpolation for peak detection
def find_true_peak(poses, window=3):
    """
    Find peak by fitting curve to reduce motion blur impact.
    """
    heights = [p.wrist_height for p in poses]
    
    # Fit parabola around detected peak
    peak_idx = np.argmax(heights)
    start = max(0, peak_idx - window)
    end = min(len(heights), peak_idx + window + 1)
    
    window_heights = heights[start:end]
    coeffs = np.polyfit(range(len(window_heights)), window_heights, 2)
    
    # True peak from parabola vertex
    true_peak_offset = -coeffs[1] / (2 * coeffs[0])
    return start + true_peak_offset
```

---

## Classification: Occlusion Issues

### 4. Self-Occlusion

**Severity**: High  
**Impact**: Critical keypoints hidden during rep phases

**Common Scenarios**:
- Wrist behind kettlebell at lockout
- Elbow hidden during rack position
- One arm occluding the other (side angle)

**Mitigation**:
```python
def handle_occlusion(keypoint_history, current_frame):
    """
    Apply temporal interpolation for brief occlusions.
    """
    OCCLUSION_THRESHOLD = 3  # frames
    
    if current_visibility < 0.3:
        # Find last good reading
        last_good = find_last_visible(keypoint_history, current_frame)
        
        if current_frame - last_good <= OCCLUSION_THRESHOLD:
            # Short occlusion - interpolate
            return interpolate_keypoint(keypoint_history, last_good, current_frame)
        else:
            # Extended occlusion - mark ambiguous
            return None, OcclusionWarning.EXTENDED
```

---

### 5. Kettlebell Occlusion

**Severity**: Medium  
**Impact**: Wrist detection affected by kettlebell blocking view

**Mitigation**:
- Infer wrist position from forearm angle when occluded
- Use elbow position + arm length estimate
- Accept slight inaccuracy for wrist-based measurements

---

## Classification: Camera Angle Issues

### 6. Side-On Camera Angle

**Severity**: Medium  
**Impact**: Depth perception issues, symmetry check impossible

**Symptoms**:
- One arm consistently occluded
- Lockout appears incomplete due to perspective
- No ability to check arm symmetry

**Mitigation**:
```python
def detect_camera_angle(poses):
    """
    Detect camera angle based on shoulder visibility.
    """
    left_shoulder_vis = mean([p.left_shoulder.visibility for p in poses])
    right_shoulder_vis = mean([p.right_shoulder.visibility for p in poses])
    
    visibility_ratio = min(left_shoulder_vis, right_shoulder_vis) / max(left_shoulder_vis, right_shoulder_vis)
    
    if visibility_ratio < 0.6:
        return CameraAngle.SIDE_ON
    elif visibility_ratio < 0.85:
        return CameraAngle.ANGLED
    else:
        return CameraAngle.FRONTAL
```

**Adaptation**:
- For side-on angles, skip symmetry check
- Adjust lockout thresholds for perspective

---

### 7. Low Camera Angle

**Severity**: Low  
**Impact**: Overhead position harder to verify

**Mitigation**:
- Use shoulder-to-wrist vector rather than absolute height
- Compare to athlete's own baseline

---

## Classification: Athlete Detection Issues

### 8. Multiple People in Frame

**Severity**: High  
**Impact**: Wrong athlete tracked, inconsistent results

**Symptoms**:
- Pose "jumps" between different people
- Inconsistent body proportions
- Random spikes in metrics

**Mitigation**:
```python
def select_primary_athlete(all_poses):
    """
    Select the largest/most consistent pose as primary athlete.
    """
    if len(all_poses) == 1:
        return all_poses[0]
    
    # Score by size (bounding box area) and center position
    scores = []
    for pose in all_poses:
        bbox = get_bounding_box(pose)
        area = bbox.width * bbox.height
        center_score = 1 - abs(bbox.center_x - 0.5)  # Prefer centered
        scores.append(area * center_score)
    
    return all_poses[np.argmax(scores)]
```

**Recommendation**:
- Warn user if multiple athletes detected
- Track single athlete across frames using pose similarity

---

### 9. Athlete Leaving Frame

**Severity**: Medium  
**Impact**: Lost tracking, missing reps

**Mitigation**:
- Pause rep detection when athlete exits
- Resume with re-detection
- Mark gap period as AMBIGUOUS

---

## Classification: Technical Judgment Issues

### 10. Borderline Lockout

**Severity**: High  
**Impact**: Subjective calls on elbow extension

**Challenge**:
- Competition standard is ~180° elbow extension
- 165° vs 170° can be debatable
- Pose estimation has ±5° error margin

**Mitigation**:
```python
# Use hysteresis for borderline cases
def classify_lockout(angle, history):
    HARD_PASS = 170
    HARD_FAIL = 160
    
    if angle >= HARD_PASS:
        return LockoutResult.PASS
    elif angle <= HARD_FAIL:
        return LockoutResult.FAIL
    else:
        # Borderline - check consistency
        recent_angles = history[-5:]
        if mean(recent_angles) >= 165:
            return LockoutResult.MARGINAL_PASS
        else:
            return LockoutResult.MARGINAL_FAIL
```

**Philosophy**:
- Conservative classification (prefer false negatives)
- Provide angle data for user review
- Allow appeal process for flagged reps

---

### 11. Fixation Time Judgment

**Severity**: Medium  
**Impact**: Brief holds may or may not count

**Challenge**:
- Competition requires "momentary fixation"
- Typical requirement: 0.2-0.5 seconds
- At 12 FPS, this is 2-6 frames

**Mitigation**:
```python
MIN_FIXATION_FRAMES = 3  # ~0.25 seconds at 12 FPS

def check_fixation(poses_at_lockout):
    """
    Check for sustained lockout position.
    """
    stable_frames = 0
    lockout_threshold = 165  # degrees
    
    for pose in poses_at_lockout:
        if pose.elbow_angle >= lockout_threshold and pose.is_stable:
            stable_frames += 1
        else:
            stable_frames = 0  # Reset on instability
    
    return stable_frames >= MIN_FIXATION_FRAMES
```

---

## Classification: Processing Issues

### 12. Long Video Memory Issues

**Severity**: Medium  
**Impact**: Processing fails or slows for 60-minute videos

**Mitigation**:
```python
def process_video_chunked(video_path, chunk_minutes=10):
    """
    Process long videos in chunks to manage memory.
    """
    video_duration = get_video_duration(video_path)
    chunk_results = []
    
    for start_time in range(0, video_duration, chunk_minutes * 60):
        end_time = min(start_time + chunk_minutes * 60, video_duration)
        
        chunk = extract_chunk(video_path, start_time, end_time)
        result = process_chunk(chunk)
        chunk_results.append(result)
        
        # Clear memory
        del chunk
        gc.collect()
    
    return merge_results(chunk_results)
```

---

### 13. Processing Timeout

**Severity**: Low  
**Impact**: Very long videos may hit timeout limits

**Mitigation**:
- Set generous timeout (1 hour for 60-min video)
- Implement checkpointing for resume
- Provide progress updates to user

---

## Mitigation Summary Table

| Failure Mode | Detection | Mitigation | User Action |
|--------------|-----------|------------|-------------|
| Low Resolution | Frame size check | Warning + continue | Re-record at 720p+ |
| Poor Lighting | Low visibility scores | AMBIGUOUS marking | Improve lighting |
| Motion Blur | Peak detection variance | Curve fitting | Reduce speed |
| Self-Occlusion | Keypoint visibility | Interpolation | Camera angle |
| Multiple Athletes | Pose count > 1 | Primary selection | Clear frame |
| Side Angle | Shoulder ratio | Skip symmetry | Film from front |
| Borderline Lockout | Angle in margin | Conservative + appeal | Review footage |
| Long Video | Duration check | Chunked processing | Split videos |

## Future Improvements

1. **Ensemble Models**: Use multiple pose estimation models and vote
2. **Athlete-Specific Calibration**: Learn individual's normal ranges
3. **Competition Mode**: Stricter thresholds for official events
4. **Training Mode**: Looser thresholds with more feedback
5. **AR Overlay**: Real-time feedback during recording

