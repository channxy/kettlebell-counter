"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api import api_router

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(f"Starting {settings.app_name}")
    yield
    logger.info(f"Shutting down {settings.app_name}")


app = FastAPI(
    title=settings.app_name,
    description="""
    Kettlebell Rep Counter API
    
    A production-ready computer vision application for accurate kettlebell rep counting,
    no-rep detection, and form analysis for competitive athletes and coaches.
    
    ## Key Features
    
    - **Accurate Rep Counting**: Distinguishes between Total Attempts, Valid Reps, and No-Reps
    - **No-Rep Detection**: Explicit classification with failure reasons
    - **Form Analytics**: Joint angles, ROM, symmetry, tempo, and fatigue trends
    - **Apple Health Export**: Sync workouts to HealthKit
    
    ## Rep Counting Principle
    
    **Total Attempts = Valid Reps + No-Reps + Ambiguous**
    
    This invariant is enforced at all layers.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "app": settings.app_name,
        "docs": "/docs",
        "health": "/health"
    }

