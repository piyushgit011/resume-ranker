from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import tempfile
import os
from datetime import datetime
import json

# Import our modules
from config import settings
from database import db_manager
from resume_parser import AdvancedResumeParser
from job_analyzer import JobDescriptionAnalyzer
from ai_agents import AIAgentOrchestrator
from scoring_engine import AdvancedScoringEngine
from models import CandidateCreate, JobPostingCreate, AnalysisRequest

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Resume Ranker API",
    description="AI-Powered Talent Acquisition Platform API",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
parser = AdvancedResumeParser()
job_analyzer = JobDescriptionAnalyzer()
ai_orchestrator = AIAgentOrchestrator()
scoring_engine = AdvancedScoringEngine()

# Pydantic models for API
class CandidateResponse(BaseModel):
    id: int
    name: str
    email: str
    phone: Optional[str]
    overall_score: Optional[float]
    status: str
    created_at: datetime

class JobPostingResponse(BaseModel):
    id: int
    title: str
    description: str
    location: str
    salary_range: str
    is_active: bool
    created_at: datetime

class AnalysisResponse(BaseModel):
    candidate_id: int
    job_posting_id: int
    overall_score: float
    component_scores: Dict[str, float]
    reasoning: str
    recommendations: str
    created_at: datetime

class BulkAnalysisRequest(BaseModel):
    job_description: str
    job_title: Optional[str] = None
    criteria: Optional[Dict[str, Any]] = None

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0", "timestamp": datetime.utcnow()}

# Candidate endpoints
@app.get("/api/candidates", response_model=List[CandidateResponse])
async def get_candidates(limit: int = 100, offset: int = 0):
    """Get all candidates with pagination"""
    try:
        candidates = db_manager.get_candidates(limit=limit, offset=offset)
        return [CandidateResponse(
            id=c.id,
            name=c.name,
            email=c.email,
            phone=c.phone,
            overall_score=c.overall_score,
            status=c.status,
            created_at=c.created_at
        ) for c in candidates]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/candidates/{candidate_id}", response_model=CandidateResponse)
async def get_candidate(candidate_id: int):
    """Get candidate by ID"""
    try:
        candidate = db_manager.get_candidate(candidate_id)
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        return CandidateResponse(
            id=candidate.id,
            name=candidate.name,
            email=candidate.email,
            phone=candidate.phone,
            overall_score=candidate.overall_score,
            status=candidate.status,
            created_at=candidate.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/candidates", response_model=CandidateResponse)
async def create_candidate(candidate: CandidateCreate):
    """Create a new candidate"""
    try:
        candidate_data = candidate.dict()
        new_candidate = db_manager.create_candidate(candidate_data)
        
        if not new_candidate:
            raise HTTPException(status_code=400, detail="Failed to create candidate")
        
        return CandidateResponse(
            id=new_candidate.id,
            name=new_candidate.name,
            email=new_candidate.email,
            phone=new_candidate.phone,
            overall_score=new_candidate.overall_score,
            status=new_candidate.status,
            created_at=new_candidate.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/candidates/{candidate_id}")
async def update_candidate(candidate_id: int, update_data: Dict[str, Any]):
    """Update candidate"""
    try:
        success = db_manager.update_candidate(candidate_id, update_data)
        if not success:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        return {"message": "Candidate updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/candidates/{candidate_id}")
async def delete_candidate(candidate_id: int):
    """Delete candidate"""
    try:
        success = db_manager.delete_candidate(candidate_id)
        if not success:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        return {"message": "Candidate deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Job posting endpoints
@app.get("/api/jobs", response_model=List[JobPostingResponse])
async def get_job_postings(active_only: bool = True):
    """Get all job postings"""
    try:
        jobs = db_manager.get_job_postings(active_only=active_only)
        return [JobPostingResponse(
            id=job.id,
            title=job.title,
            description=job.description,
            location=job.location,
            salary_range=job.salary_range,
            is_active=job.is_active,
            created_at=job.created_at
        ) for job in jobs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs/{job_id}", response_model=JobPostingResponse)
async def get_job_posting(job_id: int):
    """Get job posting by ID"""
    try:
        job = db_manager.get_job_posting(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job posting not found")
        
        return JobPostingResponse(
            id=job.id,
            title=job.title,
            description=job.description,
            location=job.location,
            salary_range=job.salary_range,
            is_active=job.is_active,
            created_at=job.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jobs", response_model=JobPostingResponse)
async def create_job_posting(job: JobPostingCreate):
    """Create a new job posting"""
    try:
        job_data = job.dict()
        new_job = db_manager.create_job_posting(job_data)
        
        if not new_job:
            raise HTTPException(status_code=400, detail="Failed to create job posting")
        
        return JobPostingResponse(
            id=new_job.id,
            title=new_job.title,
            description=new_job.description,
            location=new_job.location,
            salary_range=new_job.salary_range,
            is_active=new_job.is_active,
            created_at=new_job.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Analysis endpoints
@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_candidate(request: AnalysisRequest):
    """Analyze candidate against job posting"""
    try:
        # Get candidate and job posting
        candidate = db_manager.get_candidate(request.candidate_id)
        job_posting = db_manager.get_job_posting(request.job_posting_id)
        
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        if not job_posting:
            raise HTTPException(status_code=404, detail="Job posting not found")
        
        # Parse resume
        parsed_resume = parser.parse_file(candidate.resume_file_path) if candidate.resume_file_path else {
            'raw_text': candidate.resume_text,
            'skills': candidate.skills or {},
            'experience': candidate.experience or [],
            'education': candidate.education or []
        }
        
        # Analyze job description
        job_analysis = job_analyzer.analyze_job_description(job_posting.description)
        
        # AI agent analysis
        ai_results = ai_orchestrator.analyze_candidate(
            parsed_resume['raw_text'], 
            job_posting.description
        )
        
        # Calculate comprehensive score
        scoring_results = scoring_engine.calculate_comprehensive_score(
            parsed_resume, 
            job_analysis, 
            ai_results
        )
        
        # Save analysis result
        analysis_data = {
            "candidate_id": candidate.id,
            "job_posting_id": job_posting.id,
            "skills_match_score": scoring_results.get('component_scores', {}).get('skills_match', 0),
            "experience_score": scoring_results.get('component_scores', {}).get('experience_relevance', 0),
            "education_score": scoring_results.get('component_scores', {}).get('education', 0),
            "cultural_fit_score": scoring_results.get('component_scores', {}).get('cultural_fit', 0),
            "technical_score": scoring_results.get('component_scores', {}).get('technical_skills', 0),
            "overall_score": scoring_results.get('overall_score', 0),
            "analysis_details": scoring_results,
            "reasoning": ai_results.get('screener', {}).get('reasoning', ''),
            "recommendations": '; '.join(scoring_results.get('recommendations', [])),
            "screener_notes": ai_results.get('screener', {}).get('reasoning', ''),
            "technical_evaluator_notes": ai_results.get('technical_evaluator', {}).get('recommendations', ''),
            "cultural_assessor_notes": ai_results.get('cultural_fit_assessor', {}).get('overall_assessment', '')
        }
        
        analysis_result = db_manager.create_analysis_result(analysis_data)
        
        if not analysis_result:
            raise HTTPException(status_code=500, detail="Failed to save analysis result")
        
        return AnalysisResponse(
            candidate_id=candidate.id,
            job_posting_id=job_posting.id,
            overall_score=scoring_results.get('overall_score', 0),
            component_scores=scoring_results.get('component_scores', {}),
            reasoning=ai_results.get('screener', {}).get('reasoning', ''),
            recommendations='; '.join(scoring_results.get('recommendations', [])),
            created_at=analysis_result.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/bulk")
async def bulk_analyze_resumes(
    files: List[UploadFile] = File(...),
    job_description: str = Form(...),
    job_title: Optional[str] = Form(None)
):
    """Bulk analyze multiple resume files"""
    try:
        results = []
        
        # Create job posting
        job_data = {
            "title": job_title or "Bulk Analysis Job",
            "description": job_description,
            "requirements": {},
            "skills_required": [],
            "experience_required": 0,
            "education_required": "",
            "location": "",
            "salary_range": ""
        }
        
        job_posting = db_manager.create_job_posting(job_data)
        if not job_posting:
            raise HTTPException(status_code=400, detail="Failed to create job posting")
        
        for file in files:
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                # Parse resume
                parsed_resume = parser.parse_file(tmp_file_path)
                
                if 'error' not in parsed_resume:
                    # Analyze job description
                    job_analysis = job_analyzer.analyze_job_description(job_description)
                    
                    # AI agent analysis
                    ai_results = ai_orchestrator.analyze_candidate(
                        parsed_resume['raw_text'], 
                        job_description
                    )
                    
                    # Calculate comprehensive score
                    scoring_results = scoring_engine.calculate_comprehensive_score(
                        parsed_resume, 
                        job_analysis, 
                        ai_results
                    )
                    
                    # Save to database
                    candidate_data = {
                        "name": parsed_resume.get('personal_info', {}).get('name', 'Unknown'),
                        "email": parsed_resume.get('contact_info', {}).get('email', ''),
                        "phone": parsed_resume.get('contact_info', {}).get('phone', ''),
                        "resume_text": parsed_resume['raw_text'],
                        "resume_file_path": tmp_file_path,
                        "skills": parsed_resume.get('skills', {}),
                        "experience": parsed_resume.get('experience', []),
                        "education": parsed_resume.get('education', []),
                        "overall_score": scoring_results.get('overall_score', 0)
                    }
                    
                    candidate = db_manager.create_candidate(candidate_data)
                    
                    if candidate:
                        # Save analysis result
                        analysis_data = {
                            "candidate_id": candidate.id,
                            "job_posting_id": job_posting.id,
                            "skills_match_score": scoring_results.get('component_scores', {}).get('skills_match', 0),
                            "experience_score": scoring_results.get('component_scores', {}).get('experience_relevance', 0),
                            "education_score": scoring_results.get('component_scores', {}).get('education', 0),
                            "cultural_fit_score": scoring_results.get('component_scores', {}).get('cultural_fit', 0),
                            "technical_score": scoring_results.get('component_scores', {}).get('technical_skills', 0),
                            "overall_score": scoring_results.get('overall_score', 0),
                            "analysis_details": scoring_results,
                            "reasoning": ai_results.get('screener', {}).get('reasoning', ''),
                            "recommendations": '; '.join(scoring_results.get('recommendations', [])),
                            "screener_notes": ai_results.get('screener', {}).get('reasoning', ''),
                            "technical_evaluator_notes": ai_results.get('technical_evaluator', {}).get('recommendations', ''),
                            "cultural_assessor_notes": ai_results.get('cultural_fit_assessor', {}).get('overall_assessment', '')
                        }
                        
                        db_manager.create_analysis_result(analysis_data)
                    
                    results.append({
                        'filename': file.filename,
                        'candidate_name': candidate_data['name'],
                        'overall_score': scoring_results.get('overall_score', 0),
                        'component_scores': scoring_results.get('component_scores', {}),
                        'recommendations': scoring_results.get('recommendations', [])
                    })
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
            
            finally:
                # Clean up temp file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        
        return {
            "job_posting_id": job_posting.id,
            "total_files": len(files),
            "successful_analyses": len([r for r in results if 'error' not in r]),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints
@app.get("/api/analytics")
async def get_analytics(days: int = 30):
    """Get analytics data"""
    try:
        analytics_data = db_manager.get_analytics_data(days)
        return analytics_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/pipeline")
async def get_pipeline_analytics():
    """Get candidate pipeline analytics"""
    try:
        pipeline_data = db_manager.get_candidate_pipeline_data()
        return pipeline_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs/{job_id}/candidates")
async def get_top_candidates_for_job(job_id: int, limit: int = 10):
    """Get top candidates for a specific job"""
    try:
        candidates = db_manager.get_top_candidates(job_id, limit)
        return candidates
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search/candidates")
async def search_candidates(query: str, limit: int = 20):
    """Search candidates"""
    try:
        candidates = db_manager.search_candidates(query, limit)
        return candidates
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Export endpoints
@app.get("/api/export/candidates")
async def export_candidates(format: str = "csv"):
    """Export candidates data"""
    try:
        candidates = db_manager.get_candidates(limit=1000)
        
        if format == "csv":
            import pandas as pd
            
            df_data = []
            for candidate in candidates:
                df_data.append({
                    'ID': candidate.id,
                    'Name': candidate.name,
                    'Email': candidate.email,
                    'Phone': candidate.phone,
                    'Status': candidate.status,
                    'Overall Score': candidate.overall_score,
                    'Created At': candidate.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'Updated At': candidate.updated_at.strftime('%Y-%m-%d %H:%M:%S')
                })
            
            df = pd.DataFrame(df_data)
            csv_content = df.to_csv(index=False)
            
            return FileResponse(
                path=None,
                filename=f"candidates_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                media_type="text/csv",
                content=csv_content.encode()
            )
        
        elif format == "json":
            return JSONResponse(
                content=[{
                    'id': c.id,
                    'name': c.name,
                    'email': c.email,
                    'phone': c.phone,
                    'status': c.status,
                    'overall_score': c.overall_score,
                    'created_at': c.created_at.isoformat(),
                    'updated_at': c.updated_at.isoformat()
                } for c in candidates]
            )
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Database stats endpoint
@app.get("/api/stats")
async def get_database_stats():
    """Get database statistics"""
    try:
        stats = db_manager.get_database_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)