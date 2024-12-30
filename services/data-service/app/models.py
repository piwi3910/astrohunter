from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Job(Base):
    __tablename__ = 'jobs'

    id = Column(String, primary_key=True)  # UUID
    status = Column(String, nullable=False)
    ra = Column(Float, nullable=False)
    dec = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    files = relationship("File", back_populates="job")
    metadata = Column(JSON)

class File(Base):
    __tablename__ = 'files'

    id = Column(Integer, primary_key=True)
    job_id = Column(String, ForeignKey('jobs.id'))
    file_path = Column(String, nullable=False)
    file_type = Column(String)  # e.g., 'FITS'
    size_bytes = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)  # Store FITS header info and other metadata
    job = relationship("Job", back_populates="files")

class Detection(Base):
    __tablename__ = 'detections'

    id = Column(Integer, primary_key=True)
    job_id = Column(String, ForeignKey('jobs.id'))
    file_id = Column(Integer, ForeignKey('files.id'))
    ra = Column(Float)
    dec = Column(Float)
    magnitude = Column(Float)
    detection_type = Column(String)  # 'fast' or 'detailed'
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)  # Additional detection metadata