from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import json

@dataclass
class Contact:
    """Contact information for a candidate"""
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None

@dataclass
class Education:
    """Education information"""
    degree: str
    institution: str
    graduation_year: Optional[int] = None
    gpa: Optional[float] = None
    relevant_coursework: List[str] = field(default_factory=list)

@dataclass
class Experience:
    """Work experience information"""
    title: str
    company: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: Optional[int] = None
    description: str = ""
    technologies: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)

@dataclass
class Skills:
    """Skills categorization"""
    technical: List[str] = field(default_factory=list)
    soft: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)

@dataclass
class ResumeData:
    """Structured resume data"""
    filename: str
    candidate_name: str
    contact: Contact = field(default_factory=Contact)
    summary: str = ""
    skills: Skills = field(default_factory=Skills)
    experience: List[Experience] = field(default_factory=list)
    education: List[Education] = field(default_factory=list)
    projects: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'filename': self.filename,
            'candidate_name': self.candidate_name,
            'contact': self.contact.__dict__,
            'summary': self.summary,
            'skills': self.skills.__dict__,
            'experience': [exp.__dict__ for exp in self.experience],
            'education': [edu.__dict__ for edu in self.education],
            'projects': self.projects,
            'raw_text': self.raw_text
        }

@dataclass
class ScoringCriteria:
    """Scoring criteria for resume evaluation"""
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    min_experience_years: int = 0
    required_education: str = ""
    industry_experience: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=lambda: {
        'skills_match': 0.3,
        'experience_relevance': 0.25,
        'education_match': 0.15,
        'culture_fit': 0.15,
        'career_progression': 0.15
    })

@dataclass
class ResumeScore:
    """Resume scoring result"""
    resume_data: ResumeData
    overall_score: float
    skill_score: float
    experience_score: float
    education_score: float
    culture_fit_score: float
    career_progression_score: float
    detailed_analysis: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'resume_data': self.resume_data.to_dict(),
            'overall_score': self.overall_score,
            'skill_score': self.skill_score,
            'experience_score': self.experience_score,
            'education_score': self.education_score,
            'culture_fit_score': self.culture_fit_score,
            'career_progression_score': self.career_progression_score,
            'detailed_analysis': self.detailed_analysis,
            'recommendations': self.recommendations,
            'red_flags': self.red_flags,
            'created_at': self.created_at.isoformat()
        }

@dataclass
class JobRequirements:
    """Job requirements for matching"""
    title: str
    description: str
    required_skills: List[str]
    preferred_skills: List[str]
    min_experience: int
    education_requirements: str
    industry: str
    location: str
    salary_range: Optional[Dict[str, int]] = None
    company_culture: List[str] = field(default_factory=list)