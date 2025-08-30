import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class AppSettings:
    """Application configuration settings"""
    
    # API Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    # Database Configuration
    database_path: str = os.getenv("DATABASE_PATH", "/workspace/data/resume_ranker.db")
    
    # File Processing Configuration
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    max_resumes_per_batch: int = int(os.getenv("MAX_RESUMES_PER_BATCH", "100"))
    
    # AI Processing Configuration
    ai_temperature: float = float(os.getenv("AI_TEMPERATURE", "0.3"))
    max_tokens_per_resume: int = int(os.getenv("MAX_TOKENS_PER_RESUME", "50000"))
    
    # UI Configuration
    page_title: str = "Advanced Resume Ranker"
    page_icon: str = "🎯"
    layout: str = "wide"
    
    # Business Configuration
    default_scoring_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.default_scoring_weights is None:
            self.default_scoring_weights = {
                'skills_match': 0.30,
                'experience_relevance': 0.25,
                'education_match': 0.15,
                'culture_fit': 0.15,
                'career_progression': 0.15
            }
    
    def validate(self) -> Dict[str, str]:
        """Validate configuration and return any errors"""
        errors = {}
        
        if not self.openai_api_key or self.openai_api_key == "sk-your-api-key-here":
            errors['openai_api_key'] = "OpenAI API key is required"
        
        if self.max_file_size_mb <= 0:
            errors['max_file_size_mb'] = "Max file size must be positive"
        
        if self.max_resumes_per_batch <= 0:
            errors['max_resumes_per_batch'] = "Max resumes per batch must be positive"
        
        if not (0 <= self.ai_temperature <= 1):
            errors['ai_temperature'] = "AI temperature must be between 0 and 1"
        
        # Validate scoring weights sum to 1.0
        weights_sum = sum(self.default_scoring_weights.values())
        if abs(weights_sum - 1.0) > 0.01:
            errors['scoring_weights'] = f"Scoring weights must sum to 1.0, got {weights_sum}"
        
        return errors

# Global settings instance
settings = AppSettings()

# Predefined job templates for common roles
JOB_TEMPLATES = {
    "Software Engineer": {
        "required_skills": ["Python", "JavaScript", "Git", "SQL"],
        "preferred_skills": ["React", "Node.js", "AWS", "Docker"],
        "min_experience": 2,
        "education_requirements": "Bachelor's degree in Computer Science or related field",
        "industry": "Technology",
        "company_culture": ["collaborative", "innovative", "growth-oriented"]
    },
    "Data Scientist": {
        "required_skills": ["Python", "SQL", "Machine Learning", "Statistics"],
        "preferred_skills": ["TensorFlow", "PyTorch", "R", "Tableau", "AWS"],
        "min_experience": 3,
        "education_requirements": "Bachelor's degree in Data Science, Statistics, or related field",
        "industry": "Technology",
        "company_culture": ["analytical", "research-oriented", "data-driven"]
    },
    "DevOps Engineer": {
        "required_skills": ["Docker", "Kubernetes", "CI/CD", "Linux", "AWS"],
        "preferred_skills": ["Terraform", "Ansible", "Jenkins", "Monitoring"],
        "min_experience": 3,
        "education_requirements": "Bachelor's degree in Computer Science or related field",
        "industry": "Technology",
        "company_culture": ["automation-focused", "reliability-oriented", "collaborative"]
    },
    "Product Manager": {
        "required_skills": ["Product Strategy", "Agile", "Analytics", "User Research"],
        "preferred_skills": ["SQL", "A/B Testing", "Roadmapping", "Stakeholder Management"],
        "min_experience": 4,
        "education_requirements": "Bachelor's degree in Business, Engineering, or related field",
        "industry": "Technology",
        "company_culture": ["customer-focused", "strategic", "cross-functional"]
    },
    "Frontend Developer": {
        "required_skills": ["JavaScript", "HTML", "CSS", "React"],
        "preferred_skills": ["TypeScript", "Vue.js", "Angular", "SASS", "Webpack"],
        "min_experience": 2,
        "education_requirements": "Bachelor's degree in Computer Science or related field",
        "industry": "Technology",
        "company_culture": ["design-oriented", "user-focused", "innovative"]
    }
}