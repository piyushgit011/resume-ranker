from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import tempfile
import shutil
from datetime import datetime

from ..agents.ranking_orchestrator import RankingOrchestrator
from ..services.pdf_service import PDFService
from ..services.database_service import DatabaseService
from ..services.auth_service import AuthService
from ..models.resume import ScoringCriteria, JobRequirements, ResumeScore
from ..config.settings import settings

# FastAPI app for API endpoints
api_app = FastAPI(
    title="Resume Ranker API",
    description="Advanced Resume Ranking API with AI Agents",
    version="2.0.0"
)

# CORS middleware
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Services
pdf_service = PDFService()
db_service = DatabaseService()
auth_service = AuthService()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API token and get current user"""
    user = auth_service.validate_session(credentials.credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return user

@api_app.post("/api/v1/analyze-resume")
async def analyze_single_resume(
    file: UploadFile = File(...),
    job_requirements: Dict[str, Any] = None,
    current_user: dict = Depends(get_current_user)
):
    """Analyze a single resume file"""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded file
        file_path = f"{temp_dir}/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract text
        resume_text = pdf_service.extract_text_from_pdf(file_path)
        
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Create job requirements and criteria
        if job_requirements:
            job_req = JobRequirements(**job_requirements)
            criteria = ScoringCriteria(
                required_skills=job_req.required_skills,
                preferred_skills=job_req.preferred_skills,
                min_experience_years=job_req.min_experience
            )
        else:
            # Use default criteria
            job_req = JobRequirements(
                title="General Position",
                description="General position analysis",
                required_skills=[],
                preferred_skills=[],
                min_experience=0,
                education_requirements="",
                industry="General",
                location=""
            )
            criteria = ScoringCriteria()
        
        # Initialize orchestrator and analyze
        orchestrator = RankingOrchestrator(settings.openai_api_key, settings.openai_model)
        
        resume_texts = [{'filename': file.filename, 'text': resume_text}]
        scores = orchestrator.process_resume_batch(resume_texts, criteria, job_req)
        
        if scores:
            # Save to database
            job_id = db_service.save_job_posting(job_req)
            analysis_id = db_service.save_resume_analysis(scores[0], job_id)
            
            return {
                'success': True,
                'analysis_id': analysis_id,
                'result': scores[0].to_dict()
            }
        else:
            raise HTTPException(status_code=500, detail="Analysis failed")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@api_app.post("/api/v1/analyze-batch")
async def analyze_resume_batch(
    files: List[UploadFile] = File(...),
    job_requirements: Dict[str, Any] = None,
    current_user: dict = Depends(get_current_user)
):
    """Analyze multiple resume files"""
    
    if len(files) > settings.max_resumes_per_batch:
        raise HTTPException(
            status_code=400, 
            detail=f"Too many files. Maximum {settings.max_resumes_per_batch} allowed"
        )
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Process all files
        resume_texts = []
        
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                continue
            
            file_path = f"{temp_dir}/{file.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            resume_text = pdf_service.extract_text_from_pdf(file_path)
            
            if resume_text.strip():
                resume_texts.append({
                    'filename': file.filename,
                    'text': resume_text
                })
        
        if not resume_texts:
            raise HTTPException(status_code=400, detail="No valid PDF files with extractable text")
        
        # Create job requirements
        if job_requirements:
            job_req = JobRequirements(**job_requirements)
            criteria = ScoringCriteria(
                required_skills=job_req.required_skills,
                preferred_skills=job_req.preferred_skills,
                min_experience_years=job_req.min_experience
            )
        else:
            job_req = JobRequirements(
                title="Batch Analysis",
                description="Batch resume analysis",
                required_skills=[],
                preferred_skills=[],
                min_experience=0,
                education_requirements="",
                industry="General",
                location=""
            )
            criteria = ScoringCriteria()
        
        # Analyze batch
        orchestrator = RankingOrchestrator(settings.openai_api_key, settings.openai_model)
        scores = orchestrator.process_resume_batch(resume_texts, criteria, job_req)
        
        # Save to database
        job_id = db_service.save_job_posting(job_req)
        analysis_ids = []
        
        for score in scores:
            analysis_id = db_service.save_resume_analysis(score, job_id)
            analysis_ids.append(analysis_id)
        
        return {
            'success': True,
            'job_id': job_id,
            'analysis_ids': analysis_ids,
            'total_processed': len(scores),
            'results': [score.to_dict() for score in scores]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@api_app.get("/api/v1/analyses")
async def get_analyses(
    job_id: Optional[int] = None,
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    """Get resume analyses"""
    
    analyses = db_service.get_resume_analyses(job_id, limit)
    
    return {
        'success': True,
        'total': len(analyses),
        'analyses': analyses
    }

@api_app.get("/api/v1/analytics")
async def get_analytics(
    days: int = 30,
    current_user: dict = Depends(get_current_user)
):
    """Get analytics data"""
    
    from ..services.analytics_service import AnalyticsService
    analytics_service = AnalyticsService()
    
    funnel_metrics = analytics_service.get_hiring_funnel_metrics(days)
    skills_analysis = analytics_service.get_skills_demand_analysis(days)
    
    return {
        'success': True,
        'period_days': days,
        'funnel_metrics': funnel_metrics,
        'skills_analysis': skills_analysis
    }

@api_app.post("/api/v1/feedback")
async def submit_candidate_feedback(
    analysis_id: int,
    feedback_type: str,
    notes: str = "",
    current_user: dict = Depends(get_current_user)
):
    """Submit feedback about a candidate"""
    
    valid_feedback_types = ['hired', 'rejected', 'interviewed', 'on_hold']
    
    if feedback_type not in valid_feedback_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid feedback type. Must be one of: {valid_feedback_types}"
        )
    
    try:
        feedback_id = db_service.save_candidate_feedback(analysis_id, feedback_type, notes)
        
        return {
            'success': True,
            'feedback_id': feedback_id,
            'message': 'Feedback submitted successfully'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_app.get("/api/v1/job-templates")
async def get_job_templates(current_user: dict = Depends(get_current_user)):
    """Get available job templates"""
    
    from ..config.settings import JOB_TEMPLATES
    
    return {
        'success': True,
        'templates': JOB_TEMPLATES
    }

@api_app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0'
    }

# Error handlers
@api_app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        'success': False,
        'error': 'Endpoint not found',
        'detail': str(exc)
    }

@api_app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        'success': False,
        'error': 'Internal server error',
        'detail': str(exc)
    }