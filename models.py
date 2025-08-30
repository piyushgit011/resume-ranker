from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel
import json

Base = declarative_base()

class Candidate(Base):
    __tablename__ = "candidates"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    phone = Column(String)
    resume_text = Column(Text)
    resume_file_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Analysis Results
    skills = Column(JSON)
    experience = Column(JSON)
    education = Column(JSON)
    overall_score = Column(Float)
    
    # Status
    status = Column(String, default="new")  # new, screened, interviewed, hired, rejected
    notes = Column(Text)

class JobPosting(Base):
    __tablename__ = "job_postings"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    requirements = Column(JSON)
    skills_required = Column(JSON)
    experience_required = Column(Integer)
    education_required = Column(String)
    location = Column(String)
    salary_range = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, index=True)
    job_posting_id = Column(Integer, index=True)
    
    # Detailed Scores
    skills_match_score = Column(Float)
    experience_score = Column(Float)
    education_score = Column(Float)
    cultural_fit_score = Column(Float)
    technical_score = Column(Float)
    overall_score = Column(Float)
    
    # Analysis Details
    analysis_details = Column(JSON)
    reasoning = Column(Text)
    recommendations = Column(Text)
    
    # AI Agent Results
    screener_notes = Column(Text)
    technical_evaluator_notes = Column(Text)
    cultural_assessor_notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class Interview(Base):
    __tablename__ = "interviews"
    
    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, index=True)
    job_posting_id = Column(Integer, index=True)
    scheduled_at = Column(DateTime)
    interview_type = Column(String)  # phone, video, in-person
    interviewer = Column(String)
    status = Column(String, default="scheduled")  # scheduled, completed, cancelled
    notes = Column(Text)
    rating = Column(Integer)  # 1-5 scale
    feedback = Column(Text)

# Pydantic Models for API
class CandidateCreate(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    resume_text: str
    resume_file_path: Optional[str] = None

class CandidateResponse(BaseModel):
    id: int
    name: str
    email: str
    phone: Optional[str]
    overall_score: Optional[float]
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class JobPostingCreate(BaseModel):
    title: str
    description: str
    requirements: Dict[str, Any]
    skills_required: List[str]
    experience_required: int
    education_required: str
    location: str
    salary_range: str

class AnalysisRequest(BaseModel):
    candidate_id: int
    job_posting_id: int
    criteria: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    candidate_id: int
    job_posting_id: int
    overall_score: float
    detailed_scores: Dict[str, float]
    reasoning: str
    recommendations: str
    created_at: datetime
    
    class Config:
        from_attributes = True

# Database setup
def create_database():
    from config import settings
    engine = create_engine(settings.DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    return engine

def get_session():
    engine = create_database()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()