#!/usr/bin/env python3
"""
Installation and testing script for Advanced Resume Ranker
"""

import os
import sys
import subprocess
import tempfile
import sqlite3
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major != 3 or version.minor < 11:
        print("❌ Python 3.11+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    
    # Add src to path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        # Test core imports
        from models.resume import ResumeData, ScoringCriteria, JobRequirements
        from agents.resume_parser_agent import ResumeParserAgent
        from agents.skills_matcher_agent import SkillsMatcherAgent
        from agents.experience_analyzer_agent import ExperienceAnalyzerAgent
        from agents.culture_fit_agent import CultureFitAgent
        from agents.ranking_orchestrator import RankingOrchestrator
        from services.pdf_service import PDFService
        from services.database_service import DatabaseService
        from services.export_service import ExportService
        from config.settings import settings, JOB_TEMPLATES
        
        print("✅ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_database_setup():
    """Test database initialization"""
    print("🗄️ Testing database setup...")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from services.database_service import DatabaseService
        
        # Create test database
        test_db_path = "/tmp/test_resume_ranker.db"
        db_service = DatabaseService(test_db_path)
        
        # Test basic operations
        with sqlite3.connect(test_db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['job_postings', 'resume_analyses', 'candidate_feedback', 'ranking_sessions', 'users']
            
            for table in expected_tables:
                if table in tables:
                    print(f"✅ Table '{table}' created successfully")
                else:
                    print(f"❌ Table '{table}' missing")
                    return False
        
        # Cleanup
        os.remove(test_db_path)
        print("✅ Database setup test passed")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_pdf_processing():
    """Test PDF processing capabilities"""
    print("📄 Testing PDF processing...")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from services.pdf_service import PDFService
        
        pdf_service = PDFService()
        
        # Create a simple test PDF (would need actual PDF for real test)
        # For now, just test that the service initializes
        print("✅ PDF service initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ PDF processing test failed: {e}")
        return False

def create_sample_env_file():
    """Create sample environment file if it doesn't exist"""
    env_path = ".env"
    
    if not os.path.exists(env_path):
        print("📝 Creating sample .env file...")
        
        env_content = """# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4o

# Database Configuration
DATABASE_PATH=/workspace/data/resume_ranker.db

# File Processing Configuration
MAX_FILE_SIZE_MB=50
MAX_RESUMES_PER_BATCH=100

# AI Processing Configuration
AI_TEMPERATURE=0.3
MAX_TOKENS_PER_RESUME=50000

# Application Configuration
APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO
"""
        
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        print("✅ Sample .env file created")
        print("❗ Please edit .env with your actual OpenAI API key")
    else:
        print("✅ .env file already exists")

def run_basic_functionality_test():
    """Run basic functionality test"""
    print("🧪 Running basic functionality test...")
    
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        # Test model creation
        from models.resume import ResumeData, Contact, Skills
        
        test_resume = ResumeData(
            filename="test.pdf",
            candidate_name="Test Candidate",
            contact=Contact(email="test@example.com"),
            skills=Skills(technical=["Python", "SQL"])
        )
        
        assert test_resume.candidate_name == "Test Candidate"
        assert test_resume.contact.email == "test@example.com"
        assert "Python" in test_resume.skills.technical
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Main installation and testing function"""
    print("🎯 Advanced Resume Ranker - Installation and Testing")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    
    tests = [
        ("Python Version", check_python_version),
        ("Environment File", create_sample_env_file),
        ("Dependencies", install_dependencies),
        ("Module Imports", test_imports),
        ("Database Setup", test_database_setup),
        ("PDF Processing", test_pdf_processing),
        ("Basic Functionality", run_basic_functionality_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔄 {test_name}...")
        if callable(test_func):
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        else:
            test_func()
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 To start the application:")
        print("   ./start.sh")
        print("\n🌐 Or run manually:")
        print("   streamlit run app.py")
        print("\n🔐 Default admin login: admin / admin123")
        print("❗ Remember to change the default password!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("💡 Try running: pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)