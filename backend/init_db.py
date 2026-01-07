"""Initialize SQLite database for local development."""

import asyncio
from sqlalchemy import create_engine
from app.models.base import Base
from app.models.user import User
from app.models.workout import Workout
from app.models.rep_attempt import RepAttempt
from app.config import get_settings

settings = get_settings()

def init_db():
    """Create all tables in the database."""
    # Use sync engine for table creation
    engine = create_engine(settings.database_url_sync, echo=True)
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

if __name__ == "__main__":
    init_db()

