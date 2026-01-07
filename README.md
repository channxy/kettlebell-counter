# Kettlebell Rep Counter

A production-ready computer vision application for accurate kettlebell rep counting, no-rep detection, and form analysis for competitive athletes and coaches.

## Features

- **Accurate Rep Counting**: Distinguishes between Total Attempts, Valid Reps, and No-Reps
- **No-Rep Detection**: Explicit classification with failure reasons (incomplete lockout, elbow issues, etc.)
- **Form Analytics**: Joint angles, ROM, symmetry, tempo, and fatigue trends
- **Long Video Support**: Process 10-60 minute workout videos
- **Historical Tracking**: Per-user workout history with detailed analytics
- **Apple Health Export**: Sync workouts to HealthKit with proper rep classification

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Next.js Frontend                                                            │
│  ├── Video Upload Component                                                  │
│  ├── Timeline Visualization (Green/Red/Yellow overlays)                     │
│  ├── Rep Counter Dashboard                                                   │
│  └── Analytics & History Views                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  FastAPI Backend                                                             │
│  ├── /api/auth         - User authentication                                │
│  ├── /api/workouts     - Workout CRUD operations                            │
│  ├── /api/videos       - Video upload & processing status                   │
│  ├── /api/analytics    - Form analysis & trends                             │
│  └── /api/health-sync  - Apple HealthKit integration                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROCESSING LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Celery Task Queue (Redis)                                                   │
│  ├── Video Preprocessing (FFmpeg)                                            │
│  ├── Frame Extraction Pipeline                                               │
│  ├── Pose Estimation (MediaPipe)                                             │
│  ├── Rep Detection Engine                                                    │
│  └── Validation & Analytics Engine                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  PostgreSQL                          │  Object Storage (S3/MinIO)            │
│  ├── Users                           │  ├── Raw Videos                       │
│  ├── Workouts                        │  ├── Processed Frames                 │
│  ├── RepAttempts                     │  └── Pose Data (Parquet)              │
│  └── AnalyticsSummaries              │                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## CV/ML Pipeline

### Stage 1: Video Preprocessing
```
Input Video → FFmpeg Decode → Frame Extraction (10-15 FPS) → Quality Filter
```

### Stage 2: Pose Estimation
```
Frames → MediaPipe Pose → 33 Keypoints per Frame → Confidence Filtering
```

### Stage 3: Rep Detection (Deterministic)
```
Pose Sequence → Movement Cycle Detection → Rep Attempt Boundaries
                                          ├── Start: KB leaves rack/backswing
                                          └── End: KB returns to rack/fixation
```

### Stage 4: Rep Validation (Rule-Based)
```
Each Rep Attempt → Biomechanical Checks → Classification
                   ├── Lockout angle check      ├── VALID
                   ├── Elbow extension check    ├── NO_REP (with reasons)
                   ├── Overhead position check  └── AMBIGUOUS (low confidence)
                   ├── Fixation time check
                   └── Symmetry check
```

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- PostgreSQL 15+
- Redis 7+
- FFmpeg

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Run Workers
```bash
cd backend
celery -A app.worker worker --loglevel=info
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/auth/register | Register new user |
| POST | /api/auth/login | User login |
| POST | /api/videos/upload | Upload workout video |
| GET | /api/videos/{id}/status | Check processing status |
| GET | /api/workouts | List user workouts |
| GET | /api/workouts/{id} | Get workout details with rep attempts |
| GET | /api/workouts/{id}/timeline | Get timeline overlay data |
| GET | /api/analytics/trends | Get form analytics over time |
| POST | /api/health-sync/export | Export to Apple Health |

## License

MIT License - See LICENSE file for details.

