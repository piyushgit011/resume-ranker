from typing import Dict, Any, List
import asyncio
import concurrent.futures
from .base_agent import BaseAgent
from .resume_parser_agent import ResumeParserAgent
from .skills_matcher_agent import SkillsMatcherAgent
from .experience_analyzer_agent import ExperienceAnalyzerAgent
from .culture_fit_agent import CultureFitAgent
from ..models.resume import ResumeData, ScoringCriteria, ResumeScore, JobRequirements

class RankingOrchestrator:
    """Orchestrates multiple AI agents to provide comprehensive resume ranking"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        
        # Initialize agents
        self.parser_agent = ResumeParserAgent(api_key, model)
        self.skills_agent = SkillsMatcherAgent(api_key, model)
        self.experience_agent = ExperienceAnalyzerAgent(api_key, model)
        self.culture_agent = CultureFitAgent(api_key, model)
    
    def process_resume_batch(self, resume_texts: List[Dict[str, str]], 
                           criteria: ScoringCriteria, 
                           job_requirements: JobRequirements) -> List[ResumeScore]:
        """Process multiple resumes in parallel for efficiency"""
        
        # Parse all resumes first
        parsed_resumes = []
        for resume_info in resume_texts:
            resume_data = self.parser_agent.process(
                resume_info['text'], 
                resume_info['filename']
            )
            parsed_resumes.append(resume_data)
        
        # Analyze each resume with all agents
        scores = []
        for resume_data in parsed_resumes:
            score = self.analyze_single_resume(resume_data, criteria, job_requirements)
            scores.append(score)
        
        # Sort by overall score
        return sorted(scores, key=lambda x: x.overall_score, reverse=True)
    
    def analyze_single_resume(self, resume_data: ResumeData, 
                            criteria: ScoringCriteria, 
                            job_requirements: JobRequirements) -> ResumeScore:
        """Analyze a single resume with all agents"""
        
        # Run all agents in parallel for efficiency
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all agent tasks
            skills_future = executor.submit(
                self.skills_agent.process, resume_data, criteria
            )
            experience_future = executor.submit(
                self.experience_agent.process, resume_data, criteria
            )
            culture_future = executor.submit(
                self.culture_agent.process, resume_data, job_requirements
            )
            education_future = executor.submit(
                self._analyze_education, resume_data, criteria
            )
            
            # Collect results
            skills_analysis = skills_future.result()
            experience_analysis = experience_future.result()
            culture_analysis = culture_future.result()
            education_analysis = education_future.result()
        
        # Calculate weighted overall score
        skill_score = skills_analysis.get('skills_score', 0)
        experience_score = experience_analysis.get('experience_score', 0)
        education_score = education_analysis.get('education_score', 0)
        culture_score = culture_analysis.get('culture_fit_score', 0)
        career_score = experience_analysis.get('career_progression', {}).get('progression_score', 0)
        
        overall_score = (
            skill_score * criteria.weights['skills_match'] +
            experience_score * criteria.weights['experience_relevance'] +
            education_score * criteria.weights['education_match'] +
            culture_score * criteria.weights['culture_fit'] +
            career_score * criteria.weights['career_progression']
        )
        
        # Compile detailed analysis
        detailed_analysis = {
            'skills_analysis': skills_analysis,
            'experience_analysis': experience_analysis,
            'culture_analysis': culture_analysis,
            'education_analysis': education_analysis
        }
        
        # Generate recommendations and red flags
        recommendations = self._generate_recommendations(
            skills_analysis, experience_analysis, culture_analysis, education_analysis
        )
        red_flags = self._identify_red_flags(
            skills_analysis, experience_analysis, culture_analysis, education_analysis
        )
        
        return ResumeScore(
            resume_data=resume_data,
            overall_score=overall_score,
            skill_score=skill_score,
            experience_score=experience_score,
            education_score=education_score,
            culture_fit_score=culture_score,
            career_progression_score=career_score,
            detailed_analysis=detailed_analysis,
            recommendations=recommendations,
            red_flags=red_flags
        )
    
    def _analyze_education(self, resume_data: ResumeData, criteria: ScoringCriteria) -> Dict[str, Any]:
        """Analyze education background"""
        
        education_summary = ""
        for edu in resume_data.education:
            education_summary += f"- {edu.degree} from {edu.institution}"
            if edu.graduation_year:
                education_summary += f" ({edu.graduation_year})"
            if edu.gpa:
                education_summary += f" - GPA: {edu.gpa}"
            education_summary += "\n"
        
        prompt = f"""
        Analyze education background for job fit:
        
        Required Education: {criteria.required_education}
        Candidate Education: {education_summary}
        
        Return JSON:
        {{
            "education_score": score_0_to_100,
            "degree_relevance": "highly_relevant/relevant/somewhat_relevant/not_relevant",
            "institution_quality": "top_tier/good/average/unknown",
            "education_level": "phd/masters/bachelors/associates/other",
            "meets_requirements": boolean,
            "additional_credentials": ["additional certifications or training"],
            "education_concerns": ["any concerns about education"],
            "overall_assessment": "education fit assessment"
        }}
        """
        
        response = self.parser_agent._invoke_llm(prompt)
        return self.parser_agent._parse_json_response(response)
    
    def _generate_recommendations(self, skills_analysis: Dict, experience_analysis: Dict, 
                                culture_analysis: Dict, education_analysis: Dict) -> List[str]:
        """Generate hiring recommendations based on all analyses"""
        recommendations = []
        
        # Add recommendations from each analysis
        recommendations.extend(skills_analysis.get('recommendations', []))
        recommendations.extend(experience_analysis.get('recommendations', []))
        recommendations.extend(culture_analysis.get('recommendations', []))
        recommendations.extend(education_analysis.get('recommendations', []))
        
        return list(set(recommendations))  # Remove duplicates
    
    def _identify_red_flags(self, skills_analysis: Dict, experience_analysis: Dict, 
                          culture_analysis: Dict, education_analysis: Dict) -> List[str]:
        """Identify potential red flags from all analyses"""
        red_flags = []
        
        # Skills red flags
        if skills_analysis.get('required_skills_match', {}).get('match_percentage', 0) < 50:
            red_flags.append("Missing majority of required skills")
        
        # Experience red flags
        red_flags.extend(experience_analysis.get('experience_concerns', []))
        red_flags.extend(experience_analysis.get('gap_analysis', []))
        
        # Culture red flags
        red_flags.extend(culture_analysis.get('red_flags', []))
        
        # Education red flags
        red_flags.extend(education_analysis.get('education_concerns', []))
        
        return list(set(red_flags))  # Remove duplicates