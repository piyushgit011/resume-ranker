from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from models import Base, Candidate, JobPosting, AnalysisResult, Interview
from config import settings
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(settings.DATABASE_URL)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def close_session(self, session: Session):
        """Close database session"""
        session.close()
    
    # Candidate Management
    def create_candidate(self, candidate_data: Dict[str, Any]) -> Optional[Candidate]:
        """Create a new candidate"""
        try:
            session = self.get_session()
            candidate = Candidate(**candidate_data)
            session.add(candidate)
            session.commit()
            session.refresh(candidate)
            session.close()
            return candidate
        except SQLAlchemyError as e:
            logger.error(f"Error creating candidate: {str(e)}")
            session.rollback()
            session.close()
            return None
    
    def get_candidate(self, candidate_id: int) -> Optional[Candidate]:
        """Get candidate by ID"""
        try:
            session = self.get_session()
            candidate = session.query(Candidate).filter(Candidate.id == candidate_id).first()
            session.close()
            return candidate
        except SQLAlchemyError as e:
            logger.error(f"Error getting candidate: {str(e)}")
            session.close()
            return None
    
    def get_candidates(self, limit: int = 100, offset: int = 0) -> List[Candidate]:
        """Get all candidates with pagination"""
        try:
            session = self.get_session()
            candidates = session.query(Candidate).offset(offset).limit(limit).all()
            session.close()
            return candidates
        except SQLAlchemyError as e:
            logger.error(f"Error getting candidates: {str(e)}")
            session.close()
            return []
    
    def update_candidate(self, candidate_id: int, update_data: Dict[str, Any]) -> bool:
        """Update candidate"""
        try:
            session = self.get_session()
            candidate = session.query(Candidate).filter(Candidate.id == candidate_id).first()
            if candidate:
                for key, value in update_data.items():
                    setattr(candidate, key, value)
                candidate.updated_at = datetime.utcnow()
                session.commit()
                session.close()
                return True
            session.close()
            return False
        except SQLAlchemyError as e:
            logger.error(f"Error updating candidate: {str(e)}")
            session.rollback()
            session.close()
            return False
    
    def delete_candidate(self, candidate_id: int) -> bool:
        """Delete candidate"""
        try:
            session = self.get_session()
            candidate = session.query(Candidate).filter(Candidate.id == candidate_id).first()
            if candidate:
                session.delete(candidate)
                session.commit()
                session.close()
                return True
            session.close()
            return False
        except SQLAlchemyError as e:
            logger.error(f"Error deleting candidate: {str(e)}")
            session.rollback()
            session.close()
            return False
    
    # Job Posting Management
    def create_job_posting(self, job_data: Dict[str, Any]) -> Optional[JobPosting]:
        """Create a new job posting"""
        try:
            session = self.get_session()
            job_posting = JobPosting(**job_data)
            session.add(job_posting)
            session.commit()
            session.refresh(job_posting)
            session.close()
            return job_posting
        except SQLAlchemyError as e:
            logger.error(f"Error creating job posting: {str(e)}")
            session.rollback()
            session.close()
            return None
    
    def get_job_posting(self, job_id: int) -> Optional[JobPosting]:
        """Get job posting by ID"""
        try:
            session = self.get_session()
            job_posting = session.query(JobPosting).filter(JobPosting.id == job_id).first()
            session.close()
            return job_posting
        except SQLAlchemyError as e:
            logger.error(f"Error getting job posting: {str(e)}")
            session.close()
            return None
    
    def get_job_postings(self, active_only: bool = True) -> List[JobPosting]:
        """Get all job postings"""
        try:
            session = self.get_session()
            query = session.query(JobPosting)
            if active_only:
                query = query.filter(JobPosting.is_active == True)
            job_postings = query.all()
            session.close()
            return job_postings
        except SQLAlchemyError as e:
            logger.error(f"Error getting job postings: {str(e)}")
            session.close()
            return []
    
    # Analysis Results Management
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> Optional[AnalysisResult]:
        """Create analysis result"""
        try:
            session = self.get_session()
            analysis_result = AnalysisResult(**analysis_data)
            session.add(analysis_result)
            session.commit()
            session.refresh(analysis_result)
            session.close()
            return analysis_result
        except SQLAlchemyError as e:
            logger.error(f"Error creating analysis result: {str(e)}")
            session.rollback()
            session.close()
            return None
    
    def get_analysis_results(self, candidate_id: Optional[int] = None, 
                           job_posting_id: Optional[int] = None) -> List[AnalysisResult]:
        """Get analysis results with optional filters"""
        try:
            session = self.get_session()
            query = session.query(AnalysisResult)
            
            if candidate_id:
                query = query.filter(AnalysisResult.candidate_id == candidate_id)
            if job_posting_id:
                query = query.filter(AnalysisResult.job_posting_id == job_posting_id)
            
            results = query.order_by(AnalysisResult.overall_score.desc()).all()
            session.close()
            return results
        except SQLAlchemyError as e:
            logger.error(f"Error getting analysis results: {str(e)}")
            session.close()
            return []
    
    def get_top_candidates(self, job_posting_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top candidates for a job posting"""
        try:
            session = self.get_session()
            
            # Query to get top candidates with their details
            query = text("""
                SELECT 
                    c.id,
                    c.name,
                    c.email,
                    c.phone,
                    ar.overall_score,
                    ar.skills_match_score,
                    ar.experience_score,
                    ar.education_score,
                    ar.cultural_fit_score,
                    ar.technical_score,
                    ar.reasoning,
                    ar.recommendations,
                    ar.created_at
                FROM candidates c
                JOIN analysis_results ar ON c.id = ar.candidate_id
                WHERE ar.job_posting_id = :job_id
                ORDER BY ar.overall_score DESC
                LIMIT :limit
            """)
            
            result = session.execute(query, {"job_id": job_posting_id, "limit": limit})
            candidates = []
            
            for row in result:
                candidates.append({
                    "id": row.id,
                    "name": row.name,
                    "email": row.email,
                    "phone": row.phone,
                    "overall_score": row.overall_score,
                    "skills_match_score": row.skills_match_score,
                    "experience_score": row.experience_score,
                    "education_score": row.education_score,
                    "cultural_fit_score": row.cultural_fit_score,
                    "technical_score": row.technical_score,
                    "reasoning": row.reasoning,
                    "recommendations": row.recommendations,
                    "created_at": row.created_at
                })
            
            session.close()
            return candidates
        except SQLAlchemyError as e:
            logger.error(f"Error getting top candidates: {str(e)}")
            session.close()
            return []
    
    # Interview Management
    def create_interview(self, interview_data: Dict[str, Any]) -> Optional[Interview]:
        """Create interview record"""
        try:
            session = self.get_session()
            interview = Interview(**interview_data)
            session.add(interview)
            session.commit()
            session.refresh(interview)
            session.close()
            return interview
        except SQLAlchemyError as e:
            logger.error(f"Error creating interview: {str(e)}")
            session.rollback()
            session.close()
            return None
    
    def get_interviews(self, candidate_id: Optional[int] = None, 
                      job_posting_id: Optional[int] = None) -> List[Interview]:
        """Get interviews with optional filters"""
        try:
            session = self.get_session()
            query = session.query(Interview)
            
            if candidate_id:
                query = query.filter(Interview.candidate_id == candidate_id)
            if job_posting_id:
                query = query.filter(Interview.job_posting_id == job_posting_id)
            
            interviews = query.order_by(Interview.scheduled_at.desc()).all()
            session.close()
            return interviews
        except SQLAlchemyError as e:
            logger.error(f"Error getting interviews: {str(e)}")
            session.close()
            return []
    
    # Analytics and Reporting
    def get_analytics_data(self, days: int = 30) -> Dict[str, Any]:
        """Get analytics data for dashboard"""
        try:
            session = self.get_session()
            
            # Date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Total candidates
            total_candidates = session.query(Candidate).count()
            
            # New candidates in period
            new_candidates = session.query(Candidate).filter(
                Candidate.created_at >= start_date
            ).count()
            
            # Total job postings
            total_jobs = session.query(JobPosting).filter(
                JobPosting.is_active == True
            ).count()
            
            # Analysis results in period
            analysis_results = session.query(AnalysisResult).filter(
                AnalysisResult.created_at >= start_date
            ).all()
            
            # Score distribution
            score_ranges = {
                "90-100": 0,
                "80-89": 0,
                "70-79": 0,
                "60-69": 0,
                "0-59": 0
            }
            
            for result in analysis_results:
                score = result.overall_score or 0
                if score >= 90:
                    score_ranges["90-100"] += 1
                elif score >= 80:
                    score_ranges["80-89"] += 1
                elif score >= 70:
                    score_ranges["70-79"] += 1
                elif score >= 60:
                    score_ranges["60-69"] += 1
                else:
                    score_ranges["0-59"] += 1
            
            # Top skills
            skills_query = text("""
                SELECT 
                    JSON_EXTRACT(skills, '$.technical') as technical_skills,
                    JSON_EXTRACT(skills, '$.programming_languages') as programming_skills
                FROM candidates 
                WHERE skills IS NOT NULL
                LIMIT 100
            """)
            
            skills_result = session.execute(skills_query)
            skill_frequency = {}
            
            for row in skills_result:
                if row.technical_skills:
                    try:
                        skills = json.loads(row.technical_skills)
                        if isinstance(skills, list):
                            for skill in skills:
                                skill_frequency[skill] = skill_frequency.get(skill, 0) + 1
                    except:
                        pass
                
                if row.programming_skills:
                    try:
                        skills = json.loads(row.programming_skills)
                        if isinstance(skills, list):
                            for skill in skills:
                                skill_frequency[skill] = skill_frequency.get(skill, 0) + 1
                    except:
                        pass
            
            top_skills = sorted(skill_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Candidate status distribution
            status_query = text("""
                SELECT status, COUNT(*) as count
                FROM candidates
                GROUP BY status
            """)
            
            status_result = session.execute(status_query)
            status_distribution = {row.status: row.count for row in status_result}
            
            # Average scores by component
            avg_scores_query = text("""
                SELECT 
                    AVG(skills_match_score) as avg_skills,
                    AVG(experience_score) as avg_experience,
                    AVG(education_score) as avg_education,
                    AVG(cultural_fit_score) as avg_cultural,
                    AVG(technical_score) as avg_technical,
                    AVG(overall_score) as avg_overall
                FROM analysis_results
                WHERE created_at >= :start_date
            """)
            
            avg_scores_result = session.execute(avg_scores_query, {"start_date": start_date})
            avg_scores = avg_scores_result.fetchone()
            
            session.close()
            
            return {
                "total_candidates": total_candidates,
                "new_candidates": new_candidates,
                "total_jobs": total_jobs,
                "score_distribution": score_ranges,
                "top_skills": top_skills,
                "status_distribution": status_distribution,
                "average_scores": {
                    "skills": float(avg_scores.avg_skills or 0),
                    "experience": float(avg_scores.avg_experience or 0),
                    "education": float(avg_scores.avg_education or 0),
                    "cultural": float(avg_scores.avg_cultural or 0),
                    "technical": float(avg_scores.avg_technical or 0),
                    "overall": float(avg_scores.avg_overall or 0)
                },
                "period_days": days
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting analytics data: {str(e)}")
            session.close()
            return {}
    
    def get_candidate_pipeline_data(self) -> Dict[str, Any]:
        """Get candidate pipeline data"""
        try:
            session = self.get_session()
            
            # Pipeline stages
            pipeline_query = text("""
                SELECT 
                    status,
                    COUNT(*) as count,
                    AVG(overall_score) as avg_score
                FROM candidates c
                LEFT JOIN analysis_results ar ON c.id = ar.candidate_id
                GROUP BY status
                ORDER BY 
                    CASE status
                        WHEN 'new' THEN 1
                        WHEN 'screened' THEN 2
                        WHEN 'interviewed' THEN 3
                        WHEN 'hired' THEN 4
                        WHEN 'rejected' THEN 5
                        ELSE 6
                    END
            """)
            
            pipeline_result = session.execute(pipeline_query)
            pipeline_data = []
            
            for row in pipeline_result:
                pipeline_data.append({
                    "stage": row.status,
                    "count": row.count,
                    "avg_score": float(row.avg_score or 0)
                })
            
            # Conversion rates
            conversion_query = text("""
                SELECT 
                    COUNT(CASE WHEN status = 'screened' THEN 1 END) as screened,
                    COUNT(CASE WHEN status = 'interviewed' THEN 1 END) as interviewed,
                    COUNT(CASE WHEN status = 'hired' THEN 1 END) as hired,
                    COUNT(*) as total
                FROM candidates
            """)
            
            conversion_result = session.execute(conversion_query)
            conversion_row = conversion_result.fetchone()
            
            total = conversion_row.total or 1
            conversion_rates = {
                "screening_rate": (conversion_row.screened or 0) / total * 100,
                "interview_rate": (conversion_row.interviewed or 0) / total * 100,
                "hire_rate": (conversion_row.hired or 0) / total * 100
            }
            
            session.close()
            
            return {
                "pipeline_stages": pipeline_data,
                "conversion_rates": conversion_rates
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting pipeline data: {str(e)}")
            session.close()
            return {}
    
    def get_job_performance_data(self, job_posting_id: int) -> Dict[str, Any]:
        """Get performance data for a specific job posting"""
        try:
            session = self.get_session()
            
            # Job details
            job = session.query(JobPosting).filter(JobPosting.id == job_posting_id).first()
            if not job:
                session.close()
                return {}
            
            # Analysis results for this job
            results = session.query(AnalysisResult).filter(
                AnalysisResult.job_posting_id == job_posting_id
            ).all()
            
            if not results:
                session.close()
                return {"job": job, "no_data": True}
            
            # Score statistics
            scores = [r.overall_score for r in results if r.overall_score]
            avg_score = sum(scores) / len(scores) if scores else 0
            max_score = max(scores) if scores else 0
            min_score = min(scores) if scores else 0
            
            # Score distribution
            score_ranges = {
                "90-100": 0,
                "80-89": 0,
                "70-79": 0,
                "60-69": 0,
                "0-59": 0
            }
            
            for score in scores:
                if score >= 90:
                    score_ranges["90-100"] += 1
                elif score >= 80:
                    score_ranges["80-89"] += 1
                elif score >= 70:
                    score_ranges["70-79"] += 1
                elif score >= 60:
                    score_ranges["60-69"] += 1
                else:
                    score_ranges["0-59"] += 1
            
            # Top candidates
            top_candidates = self.get_top_candidates(job_posting_id, 5)
            
            session.close()
            
            return {
                "job": job,
                "total_candidates": len(results),
                "score_statistics": {
                    "average": avg_score,
                    "maximum": max_score,
                    "minimum": min_score
                },
                "score_distribution": score_ranges,
                "top_candidates": top_candidates
            }
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting job performance data: {str(e)}")
            session.close()
            return {}
    
    def search_candidates(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search candidates by name, email, or skills"""
        try:
            session = self.get_session()
            
            search_query = text("""
                SELECT 
                    c.id,
                    c.name,
                    c.email,
                    c.phone,
                    c.status,
                    c.overall_score,
                    c.created_at
                FROM candidates c
                WHERE 
                    c.name LIKE :query OR
                    c.email LIKE :query OR
                    c.resume_text LIKE :query
                ORDER BY c.overall_score DESC
                LIMIT :limit
            """)
            
            result = session.execute(search_query, {
                "query": f"%{query}%",
                "limit": limit
            })
            
            candidates = []
            for row in result:
                candidates.append({
                    "id": row.id,
                    "name": row.name,
                    "email": row.email,
                    "phone": row.phone,
                    "status": row.status,
                    "overall_score": row.overall_score,
                    "created_at": row.created_at
                })
            
            session.close()
            return candidates
            
        except SQLAlchemyError as e:
            logger.error(f"Error searching candidates: {str(e)}")
            session.close()
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            session = self.get_session()
            
            stats = {
                "candidates": session.query(Candidate).count(),
                "job_postings": session.query(JobPosting).count(),
                "analysis_results": session.query(AnalysisResult).count(),
                "interviews": session.query(Interview).count()
            }
            
            session.close()
            return stats
            
        except SQLAlchemyError as e:
            logger.error(f"Error getting database stats: {str(e)}")
            session.close()
            return {}

# Global database manager instance
db_manager = DatabaseManager()