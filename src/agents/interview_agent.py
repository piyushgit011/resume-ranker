from typing import Dict, Any, List
from .base_agent import BaseAgent
from ..models.resume import ResumeData, JobRequirements

class InterviewAgent(BaseAgent):
    """AI agent for generating personalized interview questions"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        super().__init__(api_key, model, temperature=0.4)
    
    def generate_interview_questions(self, resume_data: ResumeData, 
                                   job_requirements: JobRequirements,
                                   interview_type: str = "technical") -> Dict[str, Any]:
        """Generate personalized interview questions based on resume and job requirements"""
        
        prompt_template = """
        You are an expert interview designer. Generate personalized interview questions for this candidate based on their background and the job requirements.
        
        Job Role: {job_title}
        Job Requirements: {job_description}
        Required Skills: {required_skills}
        
        Candidate Background:
        - Name: {candidate_name}
        - Summary: {summary}
        - Key Skills: {key_skills}
        - Recent Experience: {recent_experience}
        - Education: {education}
        
        Interview Type: {interview_type}
        
        Generate a comprehensive set of interview questions in JSON format:
        {{
            "technical_questions": [
                {{
                    "question": "detailed technical question",
                    "category": "programming/system_design/algorithms/etc",
                    "difficulty": "junior/mid/senior",
                    "reasoning": "why this question is relevant for this candidate",
                    "follow_up_questions": ["related follow-up questions"]
                }}
            ],
            "behavioral_questions": [
                {{
                    "question": "behavioral question",
                    "category": "leadership/teamwork/problem_solving/etc",
                    "reasoning": "why this question targets this candidate's background",
                    "key_indicators": ["what to look for in the answer"]
                }}
            ],
            "situational_questions": [
                {{
                    "question": "situational scenario question",
                    "scenario": "detailed scenario description",
                    "evaluation_criteria": ["what makes a good answer"],
                    "red_flags": ["warning signs in answers"]
                }}
            ],
            "experience_deep_dive": [
                {{
                    "question": "question about specific experience",
                    "focus_area": "specific project/role to explore",
                    "probing_questions": ["follow-up questions to verify claims"]
                }}
            ],
            "culture_fit_questions": [
                {{
                    "question": "culture fit question",
                    "culture_aspect": "specific culture element being assessed",
                    "ideal_response_indicators": ["signs of good culture fit"]
                }}
            ],
            "skill_verification": [
                {{
                    "skill": "specific skill to verify",
                    "verification_method": "how to test this skill",
                    "sample_questions": ["specific questions to ask"],
                    "practical_exercise": "hands-on exercise if applicable"
                }}
            ],
            "interview_strategy": {{
                "recommended_duration": "suggested interview length",
                "focus_areas": ["main areas to focus on"],
                "potential_concerns": ["areas that need clarification"],
                "strengths_to_explore": ["candidate strengths to dive deeper into"],
                "overall_approach": "recommended interview approach"
            }}
        }}
        
        Tailor questions to:
        1. Candidate's specific experience and background
        2. Job requirements and responsibilities
        3. Identified strengths and potential gaps
        4. Career level and progression
        5. Technical depth and breadth
        """
        
        # Format candidate information
        recent_exp = resume_data.experience[0] if resume_data.experience else None
        recent_experience = f"{recent_exp.title} at {recent_exp.company}: {recent_exp.description}" if recent_exp else "No recent experience"
        
        education_summary = ""
        if resume_data.education:
            edu = resume_data.education[0]
            education_summary = f"{edu.degree} from {edu.institution}"
        
        key_skills = ", ".join(
            resume_data.skills.technical[:5] + 
            resume_data.skills.frameworks[:3] + 
            resume_data.skills.tools[:3]
        )
        
        formatted_prompt = prompt_template.format(
            job_title=job_requirements.title,
            job_description=job_requirements.description,
            required_skills=", ".join(job_requirements.required_skills),
            candidate_name=resume_data.candidate_name,
            summary=resume_data.summary,
            key_skills=key_skills,
            recent_experience=recent_experience,
            education=education_summary,
            interview_type=interview_type
        )
        
        response = self._invoke_llm(formatted_prompt)
        return self._parse_json_response(response)
    
    def generate_coding_challenge(self, resume_data: ResumeData, 
                                job_requirements: JobRequirements) -> Dict[str, Any]:
        """Generate a personalized coding challenge"""
        
        prompt = f"""
        Create a coding challenge tailored to this candidate and role:
        
        Candidate Skills: {", ".join(resume_data.skills.technical + resume_data.skills.languages)}
        Job Requirements: {job_requirements.description}
        Required Skills: {", ".join(job_requirements.required_skills)}
        
        Generate a coding challenge in JSON format:
        {{
            "challenge_title": "descriptive title",
            "difficulty": "junior/mid/senior",
            "estimated_time": "time in minutes",
            "problem_statement": "clear problem description",
            "requirements": ["specific requirements"],
            "input_format": "description of input format",
            "output_format": "description of expected output",
            "example_input": "sample input",
            "example_output": "expected output for sample",
            "evaluation_criteria": ["what to look for in solutions"],
            "bonus_points": ["additional challenges for extra credit"],
            "technologies_allowed": ["languages/frameworks candidate can use"],
            "hints": ["optional hints if candidate gets stuck"],
            "follow_up_questions": ["questions to ask after reviewing solution"]
        }}
        
        Make the challenge:
        1. Relevant to the job requirements
        2. Appropriate for candidate's skill level
        3. Solvable in the given time frame
        4. Test both technical skills and problem-solving approach
        """
        
        response = self._invoke_llm(prompt)
        return self._parse_json_response(response)
    
    def analyze_interview_responses(self, questions: List[Dict[str, Any]], 
                                  responses: List[str]) -> Dict[str, Any]:
        """Analyze candidate responses to interview questions"""
        
        if len(questions) != len(responses):
            return {"error": "Mismatch between questions and responses count"}
        
        qa_pairs = ""
        for i, (question, response) in enumerate(zip(questions, responses), 1):
            qa_pairs += f"Q{i}: {question.get('question', '')}\n"
            qa_pairs += f"A{i}: {response}\n\n"
        
        prompt = f"""
        Analyze these interview responses and provide detailed feedback:
        
        Interview Q&A:
        {qa_pairs}
        
        Provide analysis in JSON format:
        {{
            "overall_performance": "excellent/very_good/good/average/poor",
            "technical_competency": {{
                "score": score_0_to_100,
                "strengths": ["technical strengths demonstrated"],
                "weaknesses": ["areas needing improvement"],
                "depth_of_knowledge": "deep/moderate/surface/unclear"
            }},
            "communication_skills": {{
                "score": score_0_to_100,
                "clarity": "excellent/good/average/poor",
                "technical_explanation": "excellent/good/average/poor",
                "confidence_level": "high/medium/low"
            }},
            "problem_solving": {{
                "score": score_0_to_100,
                "approach": "systematic/intuitive/scattered/unclear",
                "creativity": "high/medium/low",
                "analytical_thinking": "strong/moderate/weak"
            }},
            "cultural_indicators": {{
                "teamwork_orientation": "strong/moderate/weak",
                "learning_mindset": "growth/fixed/unclear",
                "adaptability": "high/medium/low",
                "initiative": "proactive/reactive/passive"
            }},
            "red_flags": ["concerning responses or behaviors"],
            "positive_indicators": ["strong positive signals"],
            "follow_up_recommendations": ["areas to explore further"],
            "hiring_recommendation": "strong_hire/hire/no_hire/more_info_needed",
            "overall_assessment": "comprehensive assessment of interview performance"
        }}
        """
        
        response = self._invoke_llm(prompt)
        return self._parse_json_response(response)
    
    def generate_reference_check_questions(self, resume_data: ResumeData) -> List[Dict[str, str]]:
        """Generate reference check questions based on candidate background"""
        
        prompt = f"""
        Generate reference check questions for this candidate:
        
        Candidate: {resume_data.candidate_name}
        Recent Role: {resume_data.experience[0].title if resume_data.experience else "N/A"}
        Key Skills: {", ".join(resume_data.skills.technical[:5])}
        
        Return JSON array of questions:
        [
            {{
                "question": "specific reference question",
                "category": "performance/teamwork/leadership/technical/etc",
                "purpose": "what this question aims to uncover"
            }}
        ]
        
        Include questions about:
        1. Work performance and quality
        2. Technical abilities
        3. Teamwork and collaboration
        4. Leadership potential
        5. Areas for improvement
        6. Rehire decision
        """
        
        response = self._invoke_llm(prompt)
        result = self._parse_json_response(response)
        
        if isinstance(result, list):
            return result
        else:
            return result.get('questions', [])