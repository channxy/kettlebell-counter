"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(255), nullable=True),
        sa.Column('auth_provider', sa.String(50), nullable=False, server_default='email'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_verified', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('healthkit_enabled', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('healthkit_consent_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)

    # Workouts table
    op.create_table(
        'workouts',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('video_filename', sa.String(500), nullable=False),
        sa.Column('video_path', sa.String(1000), nullable=False),
        sa.Column('video_duration_seconds', sa.Float(), nullable=True),
        sa.Column('video_fps', sa.Float(), nullable=True),
        sa.Column('processing_status', sa.String(50), nullable=False, server_default='pending'),
        sa.Column('processing_progress', sa.Float(), server_default='0'),
        sa.Column('processing_error', sa.Text(), nullable=True),
        sa.Column('processing_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processing_completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('workout_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('lift_type', sa.String(50), nullable=False),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        # CRITICAL: Separated rep counts
        sa.Column('total_attempts', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('valid_reps', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('no_reps', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('ambiguous_reps', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('analytics_summary', postgresql.JSONB(), nullable=True),
        sa.Column('exported_to_health', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('health_export_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_workouts_user_id', 'workouts', ['user_id'])
    op.create_index('ix_workouts_processing_status', 'workouts', ['processing_status'])
    op.create_index('ix_workouts_lift_type', 'workouts', ['lift_type'])

    # Rep attempts table
    op.create_table(
        'rep_attempts',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('workout_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('timestamp_start', sa.Float(), nullable=False),
        sa.Column('timestamp_end', sa.Float(), nullable=False),
        sa.Column('frame_start', sa.Integer(), nullable=False),
        sa.Column('frame_end', sa.Integer(), nullable=False),
        sa.Column('classification', sa.String(20), nullable=False),
        sa.Column('failure_reasons', postgresql.ARRAY(sa.String(100)), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('pose_confidence_avg', sa.Float(), nullable=False),
        sa.Column('metrics', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['workout_id'], ['workouts.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_rep_attempts_workout_id', 'rep_attempts', ['workout_id'])
    op.create_index('ix_rep_attempts_classification', 'rep_attempts', ['classification'])


def downgrade() -> None:
    op.drop_table('rep_attempts')
    op.drop_table('workouts')
    op.drop_table('users')

