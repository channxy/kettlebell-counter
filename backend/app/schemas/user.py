"""User schemas."""

import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """Schema for user registration."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Schema for user response."""
    id: uuid.UUID
    email: str
    full_name: Optional[str]
    auth_provider: str
    is_active: bool
    is_verified: bool
    healthkit_enabled: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """Schema for user update."""
    full_name: Optional[str] = None
    healthkit_enabled: Optional[bool] = None

