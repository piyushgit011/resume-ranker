from typing import Dict, Any, List
from .base_agent import BaseAgent
from ..models.resume import ResumeData, JobRequirements

class CultureFitAgent(BaseAgent):
    """AI agent specialized in assessing cultural fit and soft skills"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        super().__init__(api_key, model, temperature=0.4)  # Higher temperature for nuanced analysis
    
    def process(self, resume_data: ResumeData, job_requirements: JobRequirements) -> Dict[str, Any]:
        """Analyze cultural fit and soft skills alignment"""
        
        prompt_template = """
        You are a culture fit and soft skills assessment expert. Analyze how well this candidate would fit the company culture and role requirements.
        
        Job Requirements:
        - Role: {job_title}
        - Company Culture: {company_culture}
        - Industry: {industry}
        
        Candidate Information:
        - Summary: {summary}
        - Experience: {experience_summary}
        - Skills: {soft_skills}
        - Projects: {projects}
        
        Provide detailed analysis in JSON format:
        {{
            "culture_fit_score": score_0_to_100,
            "soft_skills_analysis": {{
                "communication": {{
                    "score": score_0_to_100,
                    "evidence": ["evidence from resume"],
                    "assessment": "detailed assessment"
                }},
                "leadership": {{
                    "score": score_0_to_100,
                    "evidence": ["evidence from resume"],
                    "assessment": "detailed assessment"
                }},
                "teamwork": {{
                    "score": score_0_to_100,
                    "evidence": ["evidence from resume"],
                    "assessment": "detailed assessment"
                }},
                "problem_solving": {{
                    "score": score_0_to_100,
                    "evidence": ["evidence from resume"],
                    "assessment": "detailed assessment"
                }},
                "adaptability": {{
                    "score": score_0_to_100,
                    "evidence": ["evidence from resume"],
                    "assessment": "detailed assessment"
                }}
            }},
            "culture_alignment": {{
                "values_match": ["company values that align with candidate"],
                "work_style_fit": "excellent/good/average/poor",
                "team_dynamics": "strong_contributor/good_fit/potential_issues",
                "growth_mindset": "strong/moderate/limited",
                "innovation_potential": "high/medium/low"
            }},
            "personality_insights": {{
                "work_preferences": ["inferred work preferences"],
                "motivation_drivers": ["what likely motivates this candidate"],
                "potential_challenges": ["potential cultural challenges"],
                "collaboration_style": "assessment of collaboration approach"
            }},
            "red_flags": ["cultural or behavioral red flags"],
            "culture_strengths": ["cultural strengths and positive indicators"],
            "interview_recommendations": ["specific areas to explore in interviews"],
            "onboarding_suggestions": ["suggestions for successful onboarding"],
            "overall_assessment": "comprehensive culture fit assessment"
        }}
        
        Consider:
        1. Communication style evident in resume
        2. Leadership and initiative examples
        3. Collaboration and teamwork indicators
        4. Problem-solving approach
        5. Adaptability and learning agility
        6. Cultural values alignment
        7. Work environment preferences
        """
        
        # Format experience summary
        experience_summary = ""
        for exp in resume_data.experience[:3]:  # Top 3 most recent
            experience_summary += f"- {exp.title} at {exp.company}: {exp.description[:200]}...\n"
        
        # Format projects
        projects_summary = ""
        for project in resume_data.projects[:3]:  # Top 3 projects
            projects_summary += f"- {project.get('name', 'Project')}: {project.get('description', '')[:150]}...\n"
        
        formatted_prompt = prompt_template.format(
            job_title=job_requirements.title,
            company_culture=', '.join(job_requirements.company_culture),
            industry=job_requirements.industry,
            summary=resume_data.summary,
            experience_summary=experience_summary,
            soft_skills=', '.join(resume_data.skills.soft),
            projects=projects_summary
        )
        
        response = self._invoke_llm(formatted_prompt)
        return self._parse_json_response(response)
    
    def assess_leadership_potential(self, resume_data: ResumeData) -> Dict[str, Any]:
        """Assess leadership potential based on resume content"""
        
        prompt = f"""
        Assess the leadership potential of this candidate based on their resume:
        
        Experience: {self._format_leadership_indicators(resume_data)}
        
        Return JSON:
        {{
            "leadership_score": score_0_to_100,
            "leadership_evidence": ["specific evidence of leadership"],
            "leadership_style": "assessment of likely leadership style",
            "management_readiness": "ready/developing/not_ready",
            "leadership_potential": "high/medium/low",
            "development_areas": ["areas for leadership development"]
        }}
        """
        
        response = self._invoke_llm(prompt)
        return self._parse_json_response(response)
    
    def _format_leadership_indicators(self, resume_data: ResumeData) -> str:
        """Format experience data focusing on leadership indicators"""
        indicators = ""
        for exp in resume_data.experience:
            indicators += f"Role: {exp.title} at {exp.company}\n"
            indicators += f"Achievements: {'; '.join(exp.achievements)}\n"
            indicators += f"Description: {exp.description}\n\n"
        return indicators