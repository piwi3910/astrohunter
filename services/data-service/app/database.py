from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from .models import Base, Job, File, Detection
import os
import logging
from typing import Optional, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database URL
DATABASE_URL = "sqlite+aiosqlite:///data/db/astrohunter.db"

class DatabaseManager:
    def __init__(self):
        self.engine = create_async_engine(
            DATABASE_URL,
            echo=True  # Set to False in production
        )
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def init_db(self):
        """Initialize the database"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        async with self.async_session() as session:
            result = await session.execute(
                select(Job).where(Job.id == job_id)
            )
            return result.scalars().first()

    async def create_job(self, job_id: str, ra: float, dec: float) -> Job:
        """Create a new job"""
        async with self.async_session() as session:
            job = Job(
                id=job_id,
                status="created",
                ra=ra,
                dec=dec
            )
            session.add(job)
            await session.commit()
            return job

    async def update_job_status(self, job_id: str, status: str, metadata: dict = None):
        """Update job status"""
        async with self.async_session() as session:
            job = await session.execute(
                select(Job).where(Job.id == job_id)
            )
            job = job.scalars().first()
            if job:
                job.status = status
                if metadata:
                    job.metadata = metadata
                await session.commit()

    async def add_files(self, job_id: str, file_paths: List[str], metadata: dict = None):
        """Add files to a job"""
        async with self.async_session() as session:
            for file_path in file_paths:
                file = File(
                    job_id=job_id,
                    file_path=file_path,
                    file_type='FITS',
                    metadata=metadata
                )
                session.add(file)
            await session.commit()

    async def add_detection(self, job_id: str, file_id: int, detection_data: dict):
        """Add a detection result"""
        async with self.async_session() as session:
            detection = Detection(
                job_id=job_id,
                file_id=file_id,
                ra=detection_data.get('ra'),
                dec=detection_data.get('dec'),
                magnitude=detection_data.get('magnitude'),
                detection_type=detection_data.get('type'),
                confidence=detection_data.get('confidence'),
                metadata=detection_data.get('metadata')
            )
            session.add(detection)
            await session.commit()

    async def get_job_files(self, job_id: str) -> List[File]:
        """Get all files for a job"""
        async with self.async_session() as session:
            result = await session.execute(
                select(File).where(File.job_id == job_id)
            )
            return result.scalars().all()

    async def get_job_detections(self, job_id: str) -> List[Detection]:
        """Get all detections for a job"""
        async with self.async_session() as session:
            result = await session.execute(
                select(Detection).where(Detection.job_id == job_id)
            )
            return result.scalars().all()

    async def get_file_metadata(self, file_id: int) -> Optional[dict]:
        """Get metadata for a specific file"""
        async with self.async_session() as session:
            result = await session.execute(
                select(File).where(File.id == file_id)
            )
            file = result.scalars().first()
            return file.metadata if file else None