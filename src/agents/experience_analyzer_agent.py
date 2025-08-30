from typing import Dict, Any, List
from .base_agent import BaseAgent
from ..models.resume import ResumeData, ScoringCriteria

class ExperienceAnalyzerAgent(BaseAgent):
    """AI agent specialized in analyzing work experience and career progression"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        super().__init__(api_key, model, temperature=0.2)
    
    def process(self, resume_data: ResumeData, criteria: ScoringCriteria) -> Dict[str, Any]:
        """Analyze experience relevance and career progression"""
        
        # Calculate total experience
        total_months = sum(exp.duration_months or 0 for exp in resume_data.experience)
        total_years = total_months / 12 if total_months > 0 else 0
        
        prompt_template = """
        You are an experience analysis expert. Analyze the candidate's work experience for relevance and career progression.
        
        Job Requirements:
        - Minimum Experience: {min_experience} years
        - Industry Experience: {industry_experience}
        
        Candidate Experience:
        Total Years: {total_years}
        
        Work History:
        {work_history}
        
        Provide detailed analysis in JSON format:
        {{
            "experience_score": score_0_to_100,
            "total_experience_years": {total_years},
            "relevant_experience_years": years_of_directly_relevant_experience,
            "experience_analysis": {{
                "meets_minimum": boolean_meets_minimum_requirement,
                "experience_quality": "junior/mid/senior/expert",
                "industry_relevance": "high/medium/low",
                "role_progression": "excellent/good/average/poor",
                "leadership_experience": "extensive/some/minimal/none"
            }},
            "career_progression": {{
                "progression_score": score_0_to_100,
                "advancement_pattern": "rapid/steady/slow/unclear",
                "responsibility_growth": ["examples of increased responsibility"],
                "title_progression": ["progression of job titles"],
                "company_quality": "top-tier/established/startup/mixed",
                "tenure_analysis": "stable/job-hopper/mixed"
            }},
            "experience_highlights": ["most impressive experience points"],
            "experience_concerns": ["potential red flags or concerns"],
            "gap_analysis": ["employment gaps or concerns"],
            "industry_fit": {{
                "relevant_industries": ["industries with experience"],
                "transferable_experience": ["experience that transfers to target role"],
                "industry_knowledge": "deep/moderate/limited"
            }},
            "technical_depth": {{
                "architecture_experience": "extensive/some/minimal/none",
                "project_leadership": "led_major_projects/led_small_projects/contributor/unclear",
                "technical_complexity": "high/medium/low",
                "innovation_examples": ["examples of innovation or technical leadership"]
            }},
            "recommendations": ["recommendations based on experience analysis"],
            "overall_assessment": "comprehensive assessment of experience fit"
        }}
        
        Consider:
        1. Relevance of each role to the target position
        2. Career progression and advancement
        3. Quality and reputation of previous employers
        4. Consistency of career path
        5. Leadership and management experience
        6. Project complexity and impact
        7. Employment gaps and job tenure
        8. Technical depth and breadth
        """
        
        # Format work history
        work_history = ""
        for i, exp in enumerate(resume_data.experience, 1):
            work_history += f"{i}. {exp.title} at {exp.company}\n"
            work_history += f"   Duration: {exp.start_date} - {exp.end_date}"
            if exp.duration_months:
                work_history += f" ({exp.duration_months} months)"
            work_history += "\n"
            work_history += f"   Description: {exp.description}\n"
            work_history += f"   Technologies: {', '.join(exp.technologies)}\n"
            work_history += f"   Achievements: {'; '.join(exp.achievements)}\n\n"
        
        formatted_prompt = prompt_template.format(
            min_experience=criteria.min_experience_years,
            industry_experience=', '.join(criteria.industry_experience),
            total_years=total_years,
            work_history=work_history
        )
        
        response = self._invoke_llm(formatted_prompt)
        return self._parse_json_response(response)
    
    def analyze_career_gaps(self, resume_data: ResumeData) -> Dict[str, Any]:
        """Analyze employment gaps and provide insights"""
        
        prompt = f"""
        Analyze the employment history for gaps and provide insights:
        
        Work History Timeline:
        {self._format_experience_timeline(resume_data.experience)}
        
        Return JSON:
        {{
            "gaps_found": boolean,
            "gap_details": [
                {{
                    "start_date": "gap start",
                    "end_date": "gap end",
                    "duration_months": estimated_duration,
                    "potential_reasons": ["possible explanations"],
                    "concern_level": "low/medium/high"
                }}
            ],
            "overall_stability": "very_stable/stable/somewhat_unstable/unstable",
            "tenure_pattern": "long_tenure/average_tenure/short_tenure/mixed",
            "gap_recommendations": ["recommendations regarding gaps"],
            "stability_score": score_0_to_100
        }}
        """
        
        response = self._invoke_llm(prompt)
        return self._parse_json_response(response)
    
    def assess_technical_leadership(self, resume_data: ResumeData) -> Dict[str, Any]:
        """Assess technical leadership capabilities"""
        
        # Extract technical leadership indicators
        leadership_indicators = []
        for exp in resume_data.experience:
            if any(keyword in exp.title.lower() for keyword in ['lead', 'senior', 'principal', 'architect', 'manager']):
                leadership_indicators.append(f"{exp.title}: {exp.description}")
            
            for achievement in exp.achievements:
                if any(keyword in achievement.lower() for keyword in ['led', 'managed', 'architected', 'designed', 'mentored']):
                    leadership_indicators.append(achievement)
        
        prompt = f"""
        Assess technical leadership capabilities:
        
        Leadership Indicators:
        {chr(10).join(leadership_indicators)}
        
        Projects:
        {self._format_projects(resume_data.projects)}
        
        Return JSON:
        {{
            "technical_leadership_score": score_0_to_100,
            "leadership_evidence": ["specific evidence of technical leadership"],
            "team_size_managed": "estimated team sizes managed",
            "technical_mentoring": "extensive/some/minimal/none",
            "architecture_decisions": ["evidence of architecture/design decisions"],
            "project_ownership": "full_ownership/shared_ownership/contributor",
            "cross_functional_collaboration": "extensive/some/minimal",
            "leadership_potential": "high/medium/low",
            "leadership_style": "assessment of likely leadership style"
        }}
        """
        
        response = self._invoke_llm(prompt)
        return self._parse_json_response(response)
    
    def _format_experience_timeline(self, experiences: List) -> str:
        """Format experience for timeline analysis"""
        timeline = ""
        for exp in experiences:
            timeline += f"{exp.start_date} - {exp.end_date}: {exp.title} at {exp.company}\n"
        return timeline
    
    def _format_projects(self, projects: List[Dict[str, Any]]) -> str:
        """Format projects for analysis"""
        project_text = ""
        for project in projects:
            project_text += f"- {project.get('name', 'Project')}: {project.get('description', '')}\n"
            project_text += f"  Technologies: {', '.join(project.get('technologies', []))}\n"
        return project_text