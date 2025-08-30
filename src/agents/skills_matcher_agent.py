from typing import Dict, Any, List
from .base_agent import BaseAgent
from ..models.resume import ResumeData, ScoringCriteria

class SkillsMatcherAgent(BaseAgent):
    """AI agent specialized in matching candidate skills with job requirements"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        super().__init__(api_key, model, temperature=0.2)
    
    def process(self, resume_data: ResumeData, criteria: ScoringCriteria) -> Dict[str, Any]:
        """Analyze skills match between resume and job requirements"""
        
        prompt_template = """
        You are a skills matching expert. Analyze how well a candidate's skills match the job requirements.
        
        Job Requirements:
        - Required Skills: {required_skills}
        - Preferred Skills: {preferred_skills}
        
        Candidate Skills:
        - Technical: {technical_skills}
        - Languages: {languages}
        - Frameworks: {frameworks}
        - Tools: {tools}
        - Certifications: {certifications}
        
        Provide a detailed analysis in JSON format:
        {{
            "skills_score": score_0_to_100,
            "required_skills_match": {{
                "matched": ["list of matched required skills"],
                "missing": ["list of missing required skills"],
                "match_percentage": percentage_of_required_skills_matched
            }},
            "preferred_skills_match": {{
                "matched": ["list of matched preferred skills"],
                "bonus_skills": ["additional valuable skills not in requirements"],
                "match_percentage": percentage_of_preferred_skills_matched
            }},
            "skill_gaps": ["critical skills missing that would impact performance"],
            "skill_strengths": ["standout skills that exceed requirements"],
            "transferable_skills": ["skills that could transfer to required areas"],
            "skill_level_assessment": {{
                "beginner": ["skills at beginner level"],
                "intermediate": ["skills at intermediate level"],
                "advanced": ["skills at advanced level"],
                "expert": ["skills at expert level"]
            }},
            "recommendations": ["specific recommendations for skill development"],
            "overall_assessment": "comprehensive assessment of skills fit"
        }}
        
        Consider:
        1. Exact matches vs. related/transferable skills
        2. Depth of experience with each skill
        3. Relevance of skill combinations
        4. Industry-specific vs. general skills
        5. Emerging vs. established technologies
        """
        
        formatted_prompt = prompt_template.format(
            required_skills=', '.join(criteria.required_skills),
            preferred_skills=', '.join(criteria.preferred_skills),
            technical_skills=', '.join(resume_data.skills.technical),
            languages=', '.join(resume_data.skills.languages),
            frameworks=', '.join(resume_data.skills.frameworks),
            tools=', '.join(resume_data.skills.tools),
            certifications=', '.join(resume_data.skills.certifications)
        )
        
        response = self._invoke_llm(formatted_prompt)
        return self._parse_json_response(response)
    
    def calculate_skill_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate similarity between two skills using AI"""
        
        prompt = f"""
        Rate the similarity between these two skills on a scale of 0-1:
        Skill 1: {skill1}
        Skill 2: {skill2}
        
        Consider:
        - Exact matches = 1.0
        - Very similar (e.g., React vs ReactJS) = 0.9
        - Related technologies (e.g., Python vs Django) = 0.7
        - Same category (e.g., Java vs C++) = 0.5
        - Different but transferable = 0.3
        - Unrelated = 0.0
        
        Return only the numeric score (0.0 to 1.0):
        """
        
        try:
            response = self._invoke_llm(prompt)
            return float(response.strip())
        except:
            return 0.0