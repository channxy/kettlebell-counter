"""API routes."""

from fastapi import APIRouter

from app.api import auth, videos, workouts, analytics, health_sync

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(videos.router, prefix="/videos", tags=["Videos"])
api_router.include_router(workouts.router, prefix="/workouts", tags=["Workouts"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
api_router.include_router(health_sync.router, prefix="/health-sync", tags=["Apple Health"])

