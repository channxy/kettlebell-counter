# Kettlebell Counter - System Architecture

## Overview

This document describes the complete architecture for a production-ready kettlebell rep counting application with computer vision, designed for competitive athletes and coaches.

## Core Counting Principle

```
┌─────────────────────────────────────────────────────────────┐
│                    REP COUNTING INVARIANT                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│     TOTAL ATTEMPTS = VALID REPS + NO-REPS + AMBIGUOUS       │
│                                                              │
│  This invariant is enforced at:                             │
│    • Data Model (database constraints)                      │
│    • API Layer (validation on every response)               │
│    • UI Layer (visual separation of all counts)             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────┐    ┌──────────────────────┐                   │
│  │   Next.js Frontend   │    │    iOS App (Future)   │                  │
│  │  ├─ Video Upload     │    │  ├─ HealthKit Sync    │                  │
│  │  ├─ Timeline View    │    │  └─ Push Notifications│                  │
│  │  ├─ Rep Dashboard    │    │                       │                  │
│  │  └─ Analytics        │    └───────────────────────┘                  │
│  └──────────────────────┘                                               │
│                                                                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            API LAYER                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  FastAPI Application                                                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │  /auth      │ │  /videos    │ │  /workouts  │ │  /analytics │        │
│  │  - register │ │  - upload   │ │  - list     │ │  - trends   │        │
│  │  - login    │ │  - status   │ │  - detail   │ │  - compare  │        │
│  │  - me       │ │  - delete   │ │  - timeline │ │  - workout  │        │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘        │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  /health-sync                                                │        │
│  │  - consent (GET/POST)                                        │        │
│  │  - export (POST)                                             │        │
│  │  - export-history (GET)                                      │        │
│  └─────────────────────────────────────────────────────────────┘        │
│                                                                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        PROCESSING LAYER                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Celery Task Queue (Redis Broker)                                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  process_video_task                                              │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │    │
│  │  │ Decode   │→ │ Pose     │→ │ Rep      │→ │ Rep Validation   │ │    │
│  │  │ Video    │  │ Estimate │  │ Detect   │  │ & Classification │ │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │    │
│  │                                                                  │    │
│  │  Input: Video file path, lift type                              │    │
│  │  Output: RepAttempt[] with classifications + analytics          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────┐     ┌─────────────────────────┐            │
│  │      PostgreSQL         │     │    Object Storage       │            │
│  │  ┌─────────────────┐    │     │  ┌─────────────────┐    │            │
│  │  │     users       │    │     │  │   Raw Videos    │    │            │
│  │  ├─────────────────┤    │     │  ├─────────────────┤    │            │
│  │  │    workouts     │    │     │  │ Processed Data  │    │            │
│  │  ├─────────────────┤    │     │  └─────────────────┘    │            │
│  │  │  rep_attempts   │    │     │                         │            │
│  │  └─────────────────┘    │     │  Local FS / S3 / MinIO  │            │
│  └─────────────────────────┘     └─────────────────────────┘            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## CV/ML Pipeline

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VIDEO PROCESSING PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STAGE 1: VIDEO PREPROCESSING                                           │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐   │
│  │ Input      │ →  │ FFmpeg     │ →  │ Frame      │ →  │ Quality    │   │
│  │ Video      │    │ Decode     │    │ Extraction │    │ Filter     │   │
│  │ (MP4/MOV)  │    │            │    │ (12 FPS)   │    │            │   │
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘   │
│                                                                          │
│  STAGE 2: POSE ESTIMATION                                                │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐                     │
│  │ Frame      │ →  │ MediaPipe  │ →  │ 33         │                     │
│  │ Sequence   │    │ Pose       │    │ Keypoints  │                     │
│  │            │    │            │    │ per frame  │                     │
│  └────────────┘    └────────────┘    └────────────┘                     │
│                                                                          │
│  STAGE 3: REP DETECTION (Deterministic)                                  │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐                     │
│  │ Pose       │ →  │ Movement   │ →  │ Rep Cycle  │                     │
│  │ Sequence   │    │ Cycle      │    │ Boundaries │                     │
│  │            │    │ Detection  │    │            │                     │
│  └────────────┘    └────────────┘    └────────────┘                     │
│                         │                                                │
│                         ▼                                                │
│                    Every cycle = 1 TOTAL ATTEMPT                         │
│                                                                          │
│  STAGE 4: REP VALIDATION (Rule-Based)                                    │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐                     │
│  │ Each Rep   │ →  │ Biomech    │ →  │ Classify   │                     │
│  │ Attempt    │    │ Checks     │    │            │                     │
│  │            │    │            │    │ VALID      │                     │
│  └────────────┘    └────────────┘    │ NO_REP     │                     │
│                                      │ AMBIGUOUS  │                     │
│                                      └────────────┘                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Pose Keypoints Used

```
MediaPipe 33-Point Pose Model
                    
         [0] NOSE
            ●
           /|\
   [11]●───┼───●[12]     SHOULDERS
          /|\
  [13]●──┤ │ ├──●[14]    ELBOWS
         / │ \
 [15]●──┤  │  ├──●[16]   WRISTS
        /  │  \
       ●   │   ●         HANDS
          /|\
  [23]●───┼───●[24]      HIPS
         / \
 [25]●──┤   ├──●[26]     KNEES
       /     \
[27]●─┘       └─●[28]    ANKLES

Key joints for kettlebell analysis:
• Shoulders (11, 12): Overhead position check
• Elbows (13, 14): Lockout angle measurement
• Wrists (15, 16): KB position tracking
• Hips (23, 24): Torso lean calculation
```

## Rep Detection Algorithm

### Pseudocode

```python
def detect_rep_cycles(pose_sequence, lift_type):
    """
    DETERMINISTIC rep cycle detection.
    
    Every detected cycle = 1 Total Attempt.
    This does NOT determine validity - only presence of attempt.
    """
    
    cycles = []
    state = RACK_POSITION
    rep_start = None
    peak_height = 0
    peak_frame = None
    
    for frame, pose in enumerate(pose_sequence):
        wrist_height = calculate_wrist_height_ratio(pose)
        
        if state == RACK_POSITION:
            # Looking for rep to start
            if wrist_height > RACK_THRESHOLD:
                # Wrist moved up - rep starting
                state = IN_REP
                rep_start = frame
                peak_height = wrist_height
                peak_frame = frame
                
        elif state == IN_REP:
            # Track peak height
            if wrist_height > peak_height:
                peak_height = wrist_height
                peak_frame = frame
            
            # Check for return to rack
            if wrist_height < RACK_THRESHOLD:
                # Rep cycle complete!
                if is_valid_duration(rep_start, frame):
                    cycles.append(RepCycle(
                        start_frame=rep_start,
                        end_frame=frame,
                        peak_frame=peak_frame,
                        peak_height=peak_height
                    ))
                
                # Reset for next rep
                state = RACK_POSITION
                rep_start = None
                peak_height = 0
    
    return cycles  # Each cycle = 1 Total Attempt
```

## Rep Validation Algorithm

### Pseudocode

```python
def validate_rep(cycle, poses, settings):
    """
    Rule-based validation for a single rep attempt.
    
    CRITICAL PRINCIPLES:
    1. Never infer or guess missing data
    2. Low confidence = AMBIGUOUS, not VALID
    3. All decisions must be explainable
    4. Prefer false negatives over false positives
    """
    
    # Extract poses during this cycle
    cycle_poses = get_poses_in_range(poses, cycle.start, cycle.end)
    
    # Check 1: Pose detection confidence
    avg_confidence = mean([p.confidence for p in cycle_poses])
    
    if avg_confidence < AMBIGUOUS_THRESHOLD:
        return ValidationResult(
            classification=AMBIGUOUS,
            reasons=[LOW_POSE_CONFIDENCE],
            confidence=avg_confidence
        )
    
    # Check 2: Lockout angle (elbow extension)
    peak_poses = get_poses_at_peak(cycle_poses, cycle.peak_frame)
    lockout_angle = max(get_elbow_angles(peak_poses))
    
    lockout_passed = lockout_angle >= settings.min_lockout_angle
    
    # Check 3: Overhead position
    overhead_passed = cycle.peak_height >= settings.min_overhead_ratio
    
    # Check 4: Fixation time
    fixation_frames = count_frames_at_lockout(cycle_poses)
    fixation_passed = fixation_frames >= settings.min_fixation_frames
    
    # Check 5: Torso lean
    torso_lean = get_max_torso_lean_at_peak(peak_poses)
    lean_passed = torso_lean <= settings.max_torso_lean
    
    # Check 6: Symmetry (for double KB lifts)
    if lift_type in [JERK, LONG_CYCLE]:
        symmetry = get_arm_symmetry_at_peak(peak_poses)
        symmetry_passed = symmetry >= 0.7
    else:
        symmetry_passed = True  # N/A for snatch
    
    # Determine classification
    all_checks = [
        (lockout_passed, ELBOWS_NOT_EXTENDED),
        (overhead_passed, KETTLEBELL_NOT_OVERHEAD),
        (fixation_passed, INSUFFICIENT_FIXATION),
        (lean_passed, EXCESSIVE_TORSO_LEAN),
        (symmetry_passed, ASYMMETRIC_LOCKOUT),
    ]
    
    failure_reasons = [reason for passed, reason in all_checks if not passed]
    
    if not failure_reasons:
        classification = VALID
    elif avg_confidence < POSE_THRESHOLD:
        classification = AMBIGUOUS
        failure_reasons.append(LOW_POSE_CONFIDENCE)
    else:
        classification = NO_REP
    
    return ValidationResult(
        classification=classification,
        failure_reasons=failure_reasons,
        confidence=avg_confidence,
        metrics={
            'lockout_angle': lockout_angle,
            'overhead_height': cycle.peak_height,
            'fixation_frames': fixation_frames,
            'torso_lean': torso_lean
        }
    )
```

## Database Schema

```sql
-- Users table
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    full_name       VARCHAR(255),
    auth_provider   VARCHAR(50) DEFAULT 'email',
    is_active       BOOLEAN DEFAULT true,
    is_verified     BOOLEAN DEFAULT false,
    healthkit_enabled BOOLEAN DEFAULT false,
    healthkit_consent_date TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Workouts table
CREATE TABLE workouts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Video metadata
    video_filename  VARCHAR(500) NOT NULL,
    video_path      VARCHAR(1000) NOT NULL,
    video_duration_seconds FLOAT,
    video_fps       FLOAT,
    
    -- Processing status
    processing_status VARCHAR(50) DEFAULT 'pending',
    processing_progress FLOAT DEFAULT 0,
    processing_error TEXT,
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Workout metadata
    workout_date    TIMESTAMP WITH TIME ZONE NOT NULL,
    lift_type       VARCHAR(50) NOT NULL,
    duration_seconds FLOAT,
    
    -- CRITICAL: Separated rep counts
    -- Invariant: total_attempts = valid_reps + no_reps + ambiguous_reps
    total_attempts  INTEGER DEFAULT 0 NOT NULL,
    valid_reps      INTEGER DEFAULT 0 NOT NULL,
    no_reps         INTEGER DEFAULT 0 NOT NULL,
    ambiguous_reps  INTEGER DEFAULT 0 NOT NULL,
    
    -- Analytics
    analytics_summary JSONB,
    
    -- Health export
    exported_to_health BOOLEAN DEFAULT false,
    health_export_date TIMESTAMP WITH TIME ZONE,
    
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Rep attempts table
CREATE TABLE rep_attempts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workout_id      UUID NOT NULL REFERENCES workouts(id) ON DELETE CASCADE,
    
    -- Temporal boundaries
    timestamp_start FLOAT NOT NULL,
    timestamp_end   FLOAT NOT NULL,
    frame_start     INTEGER NOT NULL,
    frame_end       INTEGER NOT NULL,
    
    -- Classification
    classification  VARCHAR(20) NOT NULL,  -- 'valid', 'no_rep', 'ambiguous'
    failure_reasons TEXT[],
    confidence_score FLOAT NOT NULL,
    pose_confidence_avg FLOAT NOT NULL,
    
    -- Detailed metrics
    metrics         JSONB,
    
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Indexes
CREATE INDEX idx_workouts_user_id ON workouts(user_id);
CREATE INDEX idx_workouts_status ON workouts(processing_status);
CREATE INDEX idx_rep_attempts_workout ON rep_attempts(workout_id);
CREATE INDEX idx_rep_attempts_classification ON rep_attempts(classification);
```

## API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/auth/register | Register new user |
| POST | /api/auth/login | Login and get JWT |
| GET | /api/auth/me | Get current user |

### Videos

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/videos/upload | Upload workout video |
| GET | /api/videos/{id}/status | Get processing status |
| DELETE | /api/videos/{id} | Delete video |

### Workouts

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/workouts | List workouts (paginated) |
| GET | /api/workouts/{id} | Get workout details |
| GET | /api/workouts/{id}/timeline | Get timeline overlay data |
| GET | /api/workouts/{id}/no-reps | Get no-rep breakdown |

### Analytics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/analytics/trends | Get trends over time |
| GET | /api/analytics/workout/{id} | Get workout analytics |
| GET | /api/analytics/compare | Compare workouts |

### Health Sync

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/health-sync/consent | Get consent status |
| POST | /api/health-sync/consent | Update consent |
| POST | /api/health-sync/export | Export to HealthKit |
| GET | /api/health-sync/export-history | Get export history |

## Apple HealthKit Integration

### Export Payload Structure

```swift
// HealthKit Workout Export
HKWorkout(
    activityType: .functionalStrengthTraining,
    start: workoutDate,
    end: workoutDate + duration,
    duration: durationSeconds,
    totalEnergyBurned: caloriesEstimate,
    metadata: [
        "lift_type": "jerk",
        
        // CRITICAL: Both counts included
        "valid_reps": 142,           // Official count
        "total_attempts": 150,       // For reference
        "no_reps": 6,                // Non-scoring
        "ambiguous_reps": 2,
        
        "source": "kettlebell_counter"
    ]
)
```

### Consent Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   User       │     │   App        │     │  HealthKit   │
│              │     │              │     │              │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │ Enable Sync        │                    │
       │───────────────────▶│                    │
       │                    │                    │
       │ Confirm Consent    │                    │
       │◀───────────────────│                    │
       │                    │                    │
       │ Grant Permission   │                    │
       │───────────────────▶│                    │
       │                    │                    │
       │                    │ Request Auth       │
       │                    │───────────────────▶│
       │                    │                    │
       │                    │ Auth Granted       │
       │                    │◀───────────────────│
       │                    │                    │
       │                    │ Store consent date │
       │                    │───────────────────▶│
       │                    │                    │
       │ Sync Enabled       │                    │
       │◀───────────────────│                    │
       │                    │                    │
```

## Known Failure Modes & Mitigations

### 1. Poor Video Quality

**Problem**: Low resolution, poor lighting, or motion blur affects pose detection.

**Mitigation**:
- Require minimum video quality (720p recommended)
- Mark low-confidence frames as AMBIGUOUS, never VALID
- Provide user feedback on video quality issues

### 2. Occlusion

**Problem**: Kettlebell or limbs may be occluded at key moments.

**Mitigation**:
- Track confidence per keypoint
- Use temporal smoothing for brief occlusions
- Mark extended occlusions as AMBIGUOUS

### 3. Camera Angle

**Problem**: Suboptimal angles make joint measurements unreliable.

**Mitigation**:
- Recommend front or 45° angle filming
- Apply angle-based confidence adjustments
- Flag workouts with consistent angle issues

### 4. Multiple Athletes

**Problem**: Multiple people in frame confuse pose detection.

**Mitigation**:
- Track single athlete (largest pose)
- Alert user if multiple poses detected
- Reject videos with inconsistent athlete count

### 5. False Positive No-Reps

**Problem**: Valid reps incorrectly marked as no-reps.

**Mitigation**:
- Conservative thresholds (prefer false negatives)
- Allow user to review and flag disputed calls
- Provide detailed metrics for each decision

### 6. Fatigue Detection Noise

**Problem**: Natural variation confused with fatigue.

**Mitigation**:
- Use rolling averages over multiple reps
- Require significant degradation threshold (>10%)
- Compare quartiles, not individual reps

## Performance Considerations

### Video Processing

- Target: 12 FPS sampling (sufficient for 60+ RPM)
- Long videos (60 min): ~43,200 frames → ~36,000 after sampling
- Processing time: ~2-5 minutes per 10 minutes of video

### Database Queries

- Index on `user_id` for workout queries
- Index on `classification` for no-rep analysis
- JSONB indexes for analytics queries if needed

### Scaling

- Celery workers can scale horizontally
- Video processing is CPU-bound (pose estimation)
- Consider GPU workers for faster processing

## Security

### Authentication

- JWT tokens with 7-day expiry
- Bcrypt password hashing
- Rate limiting on auth endpoints

### Data Privacy

- User data isolated by user_id
- Videos stored with UUID filenames
- HealthKit sync requires explicit consent
- Old videos can be auto-deleted (configurable)

### API Security

- CORS restricted to known origins
- Input validation on all endpoints
- File type validation for uploads

