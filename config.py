import os
from typing import Dict, Any
from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Database
    DATABASE_URL: str = "sqlite:///./resume_ranker.db"
    
    # Application Settings
    APP_NAME: str = "Advanced Resume Ranker"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # AI Model Settings
    DEFAULT_MODEL: str = "gpt-4o"
    TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 4000
    
    # File Processing
    MAX_FILE_SIZE_MB: int = 50
    SUPPORTED_FORMATS: list = [".pdf", ".docx", ".txt", ".doc"]
    
    # Scoring Weights
    SCORING_WEIGHTS: Dict[str, float] = {
        "skills_match": 0.3,
        "experience_relevance": 0.25,
        "education": 0.15,
        "cultural_fit": 0.15,
        "technical_skills": 0.15
    }
    
    # Business Intelligence
    ENABLE_ANALYTICS: bool = True
    ENABLE_BIAS_DETECTION: bool = True
    
    class Config:
        env_file = ".env"

settings = Settings()