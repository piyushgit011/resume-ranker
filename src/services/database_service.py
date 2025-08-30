import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..models.resume import ResumeScore, JobRequirements, ScoringCriteria

class DatabaseService:
    """Database service for persisting resume analysis results"""
    
    def __init__(self, db_path: str = "/workspace/data/resume_ranker.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS job_postings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT,
                    requirements TEXT,  -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS resume_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_posting_id INTEGER,
                    filename TEXT NOT NULL,
                    candidate_name TEXT,
                    overall_score REAL,
                    skill_score REAL,
                    experience_score REAL,
                    education_score REAL,
                    culture_fit_score REAL,
                    career_progression_score REAL,
                    detailed_analysis TEXT,  -- JSON
                    recommendations TEXT,    -- JSON
                    red_flags TEXT,         -- JSON
                    resume_data TEXT,       -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_posting_id) REFERENCES job_postings (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candidate_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_analysis_id INTEGER,
                    feedback_type TEXT,  -- 'hired', 'rejected', 'interviewed'
                    feedback_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (resume_analysis_id) REFERENCES resume_analyses (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ranking_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_name TEXT,
                    criteria TEXT,  -- JSON
                    total_resumes INTEGER,
                    processed_resumes INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
    
    def save_job_posting(self, job_requirements: JobRequirements) -> int:
        """Save job posting and return ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO job_postings (title, description, requirements)
                VALUES (?, ?, ?)
            """, (
                job_requirements.title,
                job_requirements.description,
                json.dumps(job_requirements.__dict__, default=str)
            ))
            return cursor.lastrowid
    
    def save_resume_analysis(self, resume_score: ResumeScore, job_posting_id: Optional[int] = None) -> int:
        """Save resume analysis results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO resume_analyses (
                    job_posting_id, filename, candidate_name, overall_score,
                    skill_score, experience_score, education_score, 
                    culture_fit_score, career_progression_score,
                    detailed_analysis, recommendations, red_flags, resume_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_posting_id,
                resume_score.resume_data.filename,
                resume_score.resume_data.candidate_name,
                resume_score.overall_score,
                resume_score.skill_score,
                resume_score.experience_score,
                resume_score.education_score,
                resume_score.culture_fit_score,
                resume_score.career_progression_score,
                json.dumps(resume_score.detailed_analysis),
                json.dumps(resume_score.recommendations),
                json.dumps(resume_score.red_flags),
                json.dumps(resume_score.resume_data.to_dict())
            ))
            return cursor.lastrowid
    
    def get_resume_analyses(self, job_posting_id: Optional[int] = None, 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve resume analyses"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if job_posting_id:
                cursor = conn.execute("""
                    SELECT * FROM resume_analyses 
                    WHERE job_posting_id = ? 
                    ORDER BY overall_score DESC 
                    LIMIT ?
                """, (job_posting_id, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM resume_analyses 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_analytics_data(self, job_posting_id: Optional[int] = None) -> Dict[str, Any]:
        """Get analytics data for dashboard"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Basic statistics
            if job_posting_id:
                stats_query = """
                    SELECT 
                        COUNT(*) as total_candidates,
                        AVG(overall_score) as avg_score,
                        MAX(overall_score) as max_score,
                        MIN(overall_score) as min_score
                    FROM resume_analyses 
                    WHERE job_posting_id = ?
                """
                stats = conn.execute(stats_query, (job_posting_id,)).fetchone()
            else:
                stats_query = """
                    SELECT 
                        COUNT(*) as total_candidates,
                        AVG(overall_score) as avg_score,
                        MAX(overall_score) as max_score,
                        MIN(overall_score) as min_score
                    FROM resume_analyses
                """
                stats = conn.execute(stats_query).fetchone()
            
            # Score distribution
            score_dist_query = """
                SELECT 
                    CASE 
                        WHEN overall_score >= 90 THEN 'Excellent (90-100)'
                        WHEN overall_score >= 80 THEN 'Very Good (80-89)'
                        WHEN overall_score >= 70 THEN 'Good (70-79)'
                        WHEN overall_score >= 60 THEN 'Average (60-69)'
                        ELSE 'Below Average (<60)'
                    END as score_range,
                    COUNT(*) as count
                FROM resume_analyses
                GROUP BY score_range
            """
            score_distribution = conn.execute(score_dist_query).fetchall()
            
            return {
                'statistics': dict(stats) if stats else {},
                'score_distribution': [dict(row) for row in score_distribution]
            }
    
    def save_candidate_feedback(self, resume_analysis_id: int, 
                              feedback_type: str, notes: str) -> int:
        """Save feedback about a candidate (hired, rejected, etc.)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO candidate_feedback (resume_analysis_id, feedback_type, feedback_notes)
                VALUES (?, ?, ?)
            """, (resume_analysis_id, feedback_type, notes))
            return cursor.lastrowid
    
    def create_ranking_session(self, session_name: str, criteria: ScoringCriteria, 
                             total_resumes: int) -> int:
        """Create a new ranking session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO ranking_sessions (session_name, criteria, total_resumes, processed_resumes)
                VALUES (?, ?, ?, ?)
            """, (
                session_name,
                json.dumps(criteria.__dict__, default=str),
                total_resumes,
                0
            ))
            return cursor.lastrowid
    
    def update_ranking_session_progress(self, session_id: int, processed_count: int):
        """Update ranking session progress"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE ranking_sessions 
                SET processed_resumes = ?
                WHERE id = ?
            """, (processed_count, session_id))
    
    def complete_ranking_session(self, session_id: int):
        """Mark ranking session as completed"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE ranking_sessions 
                SET completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (session_id,))
    
    def get_ranking_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent ranking sessions"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM ranking_sessions 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]