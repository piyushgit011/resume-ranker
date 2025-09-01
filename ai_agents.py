import json
import re
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIAgent:
    def __init__(self, role: str, model: str = None):
        self.role = role
        self.model = model or settings.DEFAULT_MODEL
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=settings.TEMPERATURE,
            openai_api_key=settings.OPENAI_API_KEY
        )
    
    def analyze(self, resume_text: str, job_description: str, criteria: Dict[str, Any] = None) -> Dict[str, Any]:
        raise NotImplementedError

class ResumeScreenerAgent(AIAgent):
    def __init__(self):
        super().__init__("Resume Screener")
    
    def analyze(self, resume_text: str, job_description: str, criteria: Dict[str, Any] = None) -> Dict[str, Any]:
        """Initial screening to filter out obviously unqualified candidates"""
        
        system_prompt = """
        You are an expert resume screener with 10+ years of experience in talent acquisition.
        Your role is to perform initial screening of resumes to identify qualified candidates.
        
        Focus on:
        1. Basic qualifications match
        2. Required skills presence
        3. Experience level appropriateness
        4. Education requirements
        5. Red flags or concerns
        
        Provide a structured analysis with clear pass/fail recommendation.
        """
        
        human_prompt = f"""
        Job Description:
        {job_description}
        
        Resume Text:
        {resume_text[:8000]}
        
        Please analyze this resume and provide:
        1. Pass/Fail recommendation with reasoning
        2. Key qualifications found
        3. Missing requirements
        4. Any red flags
        5. Overall screening score (0-100)
        
        Respond in JSON format:
        {{
            "recommendation": "pass" or "fail",
            "screening_score": 0-100,
            "qualifications_found": ["list of found qualifications"],
            "missing_requirements": ["list of missing requirements"],
            "red_flags": ["list of any concerns"],
            "reasoning": "detailed explanation"
        }}
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            result = self._parse_json_response(response.content)
            result["agent_type"] = "screener"
            return result
            
        except Exception as e:
            logger.error(f"Screener agent error: {str(e)}")
            return {"error": str(e), "agent_type": "screener"}

class TechnicalEvaluatorAgent(AIAgent):
    def __init__(self):
        super().__init__("Technical Evaluator")
    
    def analyze(self, resume_text: str, job_description: str, criteria: Dict[str, Any] = None) -> Dict[str, Any]:
        """Deep technical evaluation of candidate's skills and experience"""
        
        system_prompt = """
        You are a senior technical recruiter and engineer with expertise in evaluating technical candidates.
        Your role is to assess technical skills, experience depth, and project relevance.
        
        Evaluate:
        1. Technical skills proficiency levels
        2. Project complexity and impact
        3. Technology stack relevance
        4. Problem-solving capabilities
        5. Innovation and leadership in technical roles
        
        Provide detailed technical assessment with specific examples.
        """
        
        human_prompt = f"""
        Job Description:
        {job_description}
        
        Resume Text:
        {resume_text[:8000]}
        
        Please provide a detailed technical evaluation:
        1. Technical skills assessment with proficiency levels
        2. Project analysis and impact evaluation
        3. Technology stack alignment
        4. Problem-solving evidence
        5. Technical leadership indicators
        
        Respond in JSON format:
        {{
            "technical_score": 0-100,
            "skills_assessment": {{
                "proficient_skills": ["skills with high proficiency"],
                "intermediate_skills": ["skills with medium proficiency"],
                "beginner_skills": ["skills with basic proficiency"]
            }},
            "project_analysis": [
                {{
                    "project": "project name",
                    "complexity": "high/medium/low",
                    "impact": "description of impact",
                    "technologies": ["technologies used"]
                }}
            ],
            "technology_alignment": 0-100,
            "problem_solving_evidence": ["examples of problem-solving"],
            "leadership_indicators": ["examples of technical leadership"],
            "recommendations": "specific technical recommendations"
        }}
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            result = self._parse_json_response(response.content)
            result["agent_type"] = "technical_evaluator"
            return result
            
        except Exception as e:
            logger.error(f"Technical evaluator error: {str(e)}")
            return {"error": str(e), "agent_type": "technical_evaluator"}

class CulturalFitAssessorAgent(AIAgent):
    def __init__(self):
        super().__init__("Cultural Fit Assessor")
    
    def analyze(self, resume_text: str, job_description: str, criteria: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess cultural fit and soft skills"""
        
        system_prompt = """
        You are an expert in organizational psychology and cultural fit assessment.
        Your role is to evaluate candidates for cultural alignment and soft skills.
        
        Assess:
        1. Communication skills and style
        2. Leadership and teamwork indicators
        3. Adaptability and learning agility
        4. Values alignment
        5. Growth mindset and motivation
        
        Focus on evidence from resume content, not assumptions.
        """
        
        human_prompt = f"""
        Job Description:
        {job_description}
        
        Resume Text:
        {resume_text[:8000]}
        
        Please assess cultural fit and soft skills:
        1. Communication evidence
        2. Leadership indicators
        3. Teamwork examples
        4. Adaptability signs
        5. Values alignment
        6. Growth mindset indicators
        
        Respond in JSON format:
        {{
            "cultural_fit_score": 0-100,
            "communication_skills": {{
                "evidence": ["examples of communication skills"],
                "score": 0-100
            }},
            "leadership_indicators": {{
                "evidence": ["examples of leadership"],
                "score": 0-100
            }},
            "teamwork_evidence": {{
                "evidence": ["examples of teamwork"],
                "score": 0-100
            }},
            "adaptability": {{
                "evidence": ["examples of adaptability"],
                "score": 0-100
            }},
            "values_alignment": {{
                "evidence": ["evidence of values alignment"],
                "score": 0-100
            }},
            "growth_mindset": {{
                "evidence": ["examples of growth mindset"],
                "score": 0-100
            }},
            "overall_assessment": "comprehensive cultural fit assessment"
        }}
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            result = self._parse_json_response(response.content)
            result["agent_type"] = "cultural_fit_assessor"
            return result
            
        except Exception as e:
            logger.error(f"Cultural fit assessor error: {str(e)}")
            return {"error": str(e), "agent_type": "cultural_fit_assessor"}

class BiasDetectionAgent(AIAgent):
    def __init__(self):
        super().__init__("Bias Detection Agent")
    
    def analyze(self, resume_text: str, job_description: str, criteria: Dict[str, Any] = None) -> Dict[str, Any]:
        """Detect potential biases in the evaluation process"""
        
        system_prompt = """
        You are an expert in diversity, equity, and inclusion with expertise in bias detection.
        Your role is to identify potential biases in resume evaluation and job matching.
        
        Look for:
        1. Unconscious bias indicators
        2. Demographic bias risks
        3. Educational bias
        4. Experience bias
        5. Language bias
        6. Recommendation for fair evaluation
        """
        
        human_prompt = f"""
        Job Description:
        {job_description}
        
        Resume Text:
        {resume_text[:8000]}
        
        Please analyze for potential biases:
        1. Demographic bias risks
        2. Educational bias indicators
        3. Experience bias patterns
        4. Language bias concerns
        5. Unconscious bias factors
        
        Respond in JSON format:
        {{
            "bias_risk_score": 0-100,
            "demographic_bias_risks": ["identified risks"],
            "educational_bias": ["educational bias indicators"],
            "experience_bias": ["experience bias patterns"],
            "language_bias": ["language bias concerns"],
            "unconscious_bias_factors": ["unconscious bias indicators"],
            "fair_evaluation_recommendations": ["recommendations for fair evaluation"],
            "overall_assessment": "comprehensive bias assessment"
        }}
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            result = self._parse_json_response(response.content)
            result["agent_type"] = "bias_detection"
            return result
            
        except Exception as e:
            logger.error(f"Bias detection agent error: {str(e)}")
            return {"error": str(e), "agent_type": "bias_detection"}

class AIAgentOrchestrator:
    def __init__(self):
        self.agents = {
            "screener": ResumeScreenerAgent(),
            "technical_evaluator": TechnicalEvaluatorAgent(),
            "cultural_fit_assessor": CulturalFitAssessorAgent(),
            "bias_detection": BiasDetectionAgent()
        }
    
    def analyze_candidate(self, resume_text: str, job_description: str, criteria: Dict[str, Any] = None) -> Dict[str, Any]:
        """Orchestrate all agents to provide comprehensive analysis"""
        
        results = {}
        
        for agent_name, agent in self.agents.items():
            try:
                logger.info(f"Running {agent_name} analysis...")
                result = agent.analyze(resume_text, job_description, criteria)
                results[agent_name] = result
            except Exception as e:
                logger.error(f"Error in {agent_name}: {str(e)}")
                results[agent_name] = {"error": str(e), "agent_type": agent_name}
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(results)
        results["overall_score"] = overall_score
        
        return results
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate weighted overall score from all agent results"""
        
        weights = settings.SCORING_WEIGHTS
        total_score = 0
        total_weight = 0
        
        # Map agent results to scoring categories
        score_mapping = {
            "screener": results.get("screener", {}).get("screening_score", 0),
            "technical_evaluator": results.get("technical_evaluator", {}).get("technical_score", 0),
            "cultural_fit_assessor": results.get("cultural_fit_assessor", {}).get("cultural_fit_score", 0)
        }
        
        # Calculate weighted average
        for category, score in score_mapping.items():
            if isinstance(score, (int, float)) and score > 0:
                weight = weights.get(category.replace("_", "_"), 0.2)  # Default weight
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from AI agent"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'({.*})', response.replace('\n', ''), re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except:
                    return {"error": "JSON parsing failed", "raw_response": response}
            else:
                return {"error": "JSON extraction failed", "raw_response": response}