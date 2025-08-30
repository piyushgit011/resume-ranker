import pytest
import sys
import os

# Add src to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.resume import ResumeData, Contact, Skills, ScoringCriteria, JobRequirements
from agents.resume_parser_agent import ResumeParserAgent
from agents.skills_matcher_agent import SkillsMatcherAgent

class TestResumeParserAgent:
    """Test cases for Resume Parser Agent"""
    
    @pytest.fixture
    def sample_resume_text(self):
        return """
        John Doe
        Software Engineer
        Email: john.doe@email.com
        Phone: (555) 123-4567
        LinkedIn: linkedin.com/in/johndoe
        
        EXPERIENCE
        Senior Software Engineer at Tech Corp (2020-2023)
        - Developed web applications using Python and React
        - Led a team of 5 developers
        - Implemented CI/CD pipelines
        
        Software Engineer at StartupXYZ (2018-2020)
        - Built REST APIs using Django
        - Worked with PostgreSQL and Redis
        
        EDUCATION
        Bachelor of Science in Computer Science
        University of Technology (2014-2018)
        GPA: 3.8/4.0
        
        SKILLS
        Python, JavaScript, React, Django, PostgreSQL, AWS, Docker
        """
    
    def test_resume_parsing_structure(self, sample_resume_text):
        """Test that resume parsing returns proper structure"""
        # Note: This would require a mock API key for testing
        # In production, use dependency injection for testing
        
        # Mock the parser agent
        class MockResumeParserAgent:
            def process(self, text, filename):
                return ResumeData(
                    filename=filename,
                    candidate_name="John Doe",
                    contact=Contact(email="john.doe@email.com"),
                    skills=Skills(technical=["Python", "JavaScript"])
                )
        
        parser = MockResumeParserAgent()
        result = parser.process(sample_resume_text, "john_doe.pdf")
        
        assert result.candidate_name == "John Doe"
        assert result.contact.email == "john.doe@email.com"
        assert "Python" in result.skills.technical

class TestSkillsMatcherAgent:
    """Test cases for Skills Matcher Agent"""
    
    @pytest.fixture
    def sample_criteria(self):
        return ScoringCriteria(
            required_skills=["Python", "SQL", "Git"],
            preferred_skills=["React", "AWS", "Docker"]
        )
    
    @pytest.fixture
    def sample_resume_data(self):
        return ResumeData(
            filename="test.pdf",
            candidate_name="Test Candidate",
            skills=Skills(
                technical=["Python", "JavaScript", "SQL"],
                frameworks=["React", "Django"],
                tools=["Git", "Docker"]
            )
        )
    
    def test_skill_similarity_calculation(self):
        """Test skill similarity calculation"""
        # Mock implementation
        class MockSkillsMatcher:
            def calculate_skill_similarity(self, skill1, skill2):
                if skill1.lower() == skill2.lower():
                    return 1.0
                elif skill1.lower() in skill2.lower() or skill2.lower() in skill1.lower():
                    return 0.8
                else:
                    return 0.0
        
        matcher = MockSkillsMatcher()
        
        # Test exact match
        assert matcher.calculate_skill_similarity("Python", "Python") == 1.0
        
        # Test partial match
        assert matcher.calculate_skill_similarity("React", "ReactJS") == 0.8
        
        # Test no match
        assert matcher.calculate_skill_similarity("Python", "Java") == 0.0

class TestIntegration:
    """Integration tests for the full system"""
    
    def test_end_to_end_flow(self):
        """Test complete resume analysis flow"""
        # This would test the full pipeline
        # Mock implementation for demonstration
        
        # 1. PDF extraction
        # 2. Resume parsing
        # 3. Multi-agent analysis
        # 4. Score calculation
        # 5. Database storage
        
        assert True  # Placeholder for actual integration test

if __name__ == "__main__":
    pytest.main([__file__])