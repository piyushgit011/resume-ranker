import re
import json
from typing import Dict, List, Any, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobDescriptionAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.DEFAULT_MODEL,
            temperature=settings.TEMPERATURE,
            openai_api_key=settings.OPENAI_API_KEY
        )
    
    def analyze_job_description(self, job_description: str) -> Dict[str, Any]:
        """Analyze job description and extract structured requirements"""
        
        try:
            # Extract basic information
            basic_info = self._extract_basic_info(job_description)
            
            # Extract requirements
            requirements = self._extract_requirements(job_description)
            
            # Extract skills
            skills = self._extract_skills(job_description)
            
            # Extract experience requirements
            experience = self._extract_experience_requirements(job_description)
            
            # Extract education requirements
            education = self._extract_education_requirements(job_description)
            
            # Extract location and salary
            location_salary = self._extract_location_salary(job_description)
            
            # Extract company culture indicators
            culture = self._extract_culture_indicators(job_description)
            
            # Use AI for advanced analysis
            ai_analysis = self._ai_analyze_job_description(job_description)
            
            return {
                "basic_info": basic_info,
                "requirements": requirements,
                "skills": skills,
                "experience": experience,
                "education": education,
                "location_salary": location_salary,
                "culture": culture,
                "ai_analysis": ai_analysis,
                "raw_text": job_description
            }
            
        except Exception as e:
            logger.error(f"Error analyzing job description: {str(e)}")
            return {"error": str(e)}
    
    def _extract_basic_info(self, text: str) -> Dict[str, str]:
        """Extract basic job information"""
        
        # Job title
        title_patterns = [
            r'(?:position|role|job)\s*:?\s*([A-Z][a-zA-Z\s&.,-]+?)(?:\n|$|,|\.)',
            r'^([A-Z][a-zA-Z\s&.,-]+?)(?:\n|$|,|\.)',
            r'(?:hiring|seeking|looking for)\s+([A-Z][a-zA-Z\s&.,-]+?)(?:\n|$|,|\.)'
        ]
        
        title = ""
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                title = match.group(1).strip()
                break
        
        # Company name
        company_patterns = [
            r'(?:at|company|organization)\s+([A-Z][a-zA-Z\s&.,-]+?)(?:\n|$|,|\.)',
            r'([A-Z][a-zA-Z\s&.,-]+?)\s+(?:is hiring|is seeking|is looking)',
            r'(?:about|company)\s+([A-Z][a-zA-Z\s&.,-]+?)(?:\n|$|,|\.)'
        ]
        
        company = ""
        for pattern in company_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                company = match.group(1).strip()
                break
        
        # Job type
        job_type_patterns = [
            r'(full-time|part-time|contract|freelance|remote|hybrid|onsite)',
            r'(permanent|temporary|internship|entry-level|senior|junior|mid-level)'
        ]
        
        job_type = []
        for pattern in job_type_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            job_type.extend(matches)
        
        return {
            "title": title,
            "company": company,
            "job_type": list(set(job_type)),
            "department": self._extract_department(text)
        }
    
    def _extract_department(self, text: str) -> str:
        """Extract department information"""
        
        department_patterns = [
            r'(?:department|team|division)\s*:?\s*([A-Z][a-zA-Z\s&.,-]+?)(?:\n|$|,|\.)',
            r'(?:reporting to|under)\s+([A-Z][a-zA-Z\s&.,-]+?)(?:\n|$|,|\.)',
            r'(?:engineering|marketing|sales|hr|finance|operations|product|design)\s+(?:team|department|division)'
        ]
        
        for pattern in department_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_requirements(self, text: str) -> Dict[str, List[str]]:
        """Extract job requirements"""
        
        # Required qualifications
        required_patterns = [
            r'(?:required|must have|essential|mandatory)\s*:?\s*(.*?)(?=\n\s*(?:preferred|nice to have|optional|bonus))',
            r'(?:qualifications|requirements)\s*:?\s*(.*?)(?=\n\s*(?:preferred|nice to have|optional|bonus))'
        ]
        
        required = []
        for pattern in required_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                required_text = match.group(1)
                required = self._parse_requirement_list(required_text)
                break
        
        # Preferred qualifications
        preferred_patterns = [
            r'(?:preferred|nice to have|optional|bonus)\s*:?\s*(.*?)(?=\n\s*(?:responsibilities|duties|what you\'ll do))',
            r'(?:plus|additional|bonus)\s*:?\s*(.*?)(?=\n\s*(?:responsibilities|duties|what you\'ll do))'
        ]
        
        preferred = []
        for pattern in preferred_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                preferred_text = match.group(1)
                preferred = self._parse_requirement_list(preferred_text)
                break
        
        return {
            "required": required,
            "preferred": preferred
        }
    
    def _parse_requirement_list(self, text: str) -> List[str]:
        """Parse requirement list from text"""
        
        # Split by common list separators
        requirements = re.split(r'\n\s*[-•*]\s*|\n\s*\d+\.\s*', text)
        
        # Clean up requirements
        cleaned_requirements = []
        for req in requirements:
            req = req.strip()
            if req and len(req) > 10:  # Filter out very short requirements
                cleaned_requirements.append(req)
        
        return cleaned_requirements
    
    def _extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract required skills"""
        
        # Technical skills
        technical_skills = self._extract_technical_skills(text)
        
        # Soft skills
        soft_skills = self._extract_soft_skills(text)
        
        # Tools and technologies
        tools = self._extract_tools_technologies(text)
        
        return {
            "technical": technical_skills,
            "soft": soft_skills,
            "tools": tools
        }
    
    def _extract_technical_skills(self, text: str) -> List[str]:
        """Extract technical skills"""
        
        technical_patterns = [
            r'\b(?:Python|Java|JavaScript|C\+\+|C#|Ruby|PHP|Swift|Kotlin|Go|Rust|Scala|R|MATLAB)\b',
            r'\b(?:React|Angular|Vue|Django|Flask|Spring|Express|Laravel|Rails|ASP\.NET|Node\.js)\b',
            r'\b(?:MySQL|PostgreSQL|MongoDB|Redis|Oracle|SQLite|Cassandra|Elasticsearch)\b',
            r'\b(?:AWS|Azure|GCP|Google Cloud|Amazon Web Services|Microsoft Azure)\b',
            r'\b(?:Git|Docker|Kubernetes|Jenkins|CI/CD|Terraform|Ansible)\b',
            r'\b(?:HTML|CSS|JavaScript|TypeScript|jQuery|Bootstrap|SASS|LESS)\b',
            r'\b(?:Machine Learning|AI|Data Science|Analytics|Statistics|Deep Learning)\b'
        ]
        
        skills = []
        for pattern in technical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend(matches)
        
        return list(set(skills))
    
    def _extract_soft_skills(self, text: str) -> List[str]:
        """Extract soft skills"""
        
        soft_skill_patterns = [
            r'\b(?:leadership|communication|teamwork|problem solving|critical thinking|creativity|adaptability|time management|project management|mentoring|collaboration|interpersonal|negotiation|presentation|analytical|detail-oriented|self-motivated|proactive|innovative|strategic thinking)\b'
        ]
        
        skills = []
        for pattern in soft_skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend(matches)
        
        return list(set(skills))
    
    def _extract_tools_technologies(self, text: str) -> List[str]:
        """Extract tools and technologies"""
        
        tool_patterns = [
            r'\b(?:Jira|Confluence|Slack|Trello|Asana|Figma|Sketch|Photoshop|Illustrator|Tableau|Power BI|Excel|Google Analytics|Mixpanel|Amplitude)\b',
            r'\b(?:Linux|Windows|macOS|Unix|Bash|Shell|PowerShell)\b',
            r'\b(?:REST|GraphQL|API|Microservices|Agile|Scrum|Kanban|DevOps|TDD|BDD)\b'
        ]
        
        tools = []
        for pattern in tool_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            tools.extend(matches)
        
        return list(set(tools))
    
    def _extract_experience_requirements(self, text: str) -> Dict[str, Any]:
        """Extract experience requirements"""
        
        # Years of experience
        experience_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'(?:minimum|at least|minimum of)\s*(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'(?:entry-level|junior|senior|mid-level|lead|principal|architect)'
        ]
        
        years_required = 0
        level = ""
        
        for pattern in experience_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if match.group(1):
                    years_required = int(match.group(1))
                else:
                    level = match.group(0).lower()
        
        # Experience type
        experience_type_patterns = [
            r'(?:software|web|mobile|data|machine learning|AI|cloud|devops|frontend|backend|full-stack|product|marketing|sales|finance|hr|operations)\s+(?:development|engineering|experience|background)'
        ]
        
        experience_type = []
        for pattern in experience_type_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            experience_type.extend(matches)
        
        return {
            "years_required": years_required,
            "level": level,
            "type": list(set(experience_type))
        }
    
    def _extract_education_requirements(self, text: str) -> Dict[str, Any]:
        """Extract education requirements"""
        
        education_patterns = [
            r'(?:bachelor|master|phd|doctorate|associate|certificate|diploma|degree)\s+(?:in|of)\s+([A-Za-z\s&.,-]+)',
            r'(?:B\.?S\.?|M\.?S\.?|B\.?A\.?|M\.?A\.?|B\.?E\.?|M\.?E\.?)\s+(?:in|of)\s+([A-Za-z\s&.,-]+)',
            r'(?:computer science|engineering|mathematics|statistics|business|marketing|finance|economics|psychology|design)'
        ]
        
        education_required = []
        for pattern in education_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            education_required.extend(matches)
        
        # Education level
        level_patterns = [
            r'(?:bachelor|master|phd|doctorate|associate|certificate|diploma)',
            r'(?:B\.?S\.?|M\.?S\.?|B\.?A\.?|M\.?A\.?|B\.?E\.?|M\.?E\.?|PhD)'
        ]
        
        education_level = ""
        for pattern in level_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                education_level = match.group(0).lower()
                break
        
        return {
            "level": education_level,
            "fields": list(set(education_required)),
            "required": "required" in text.lower() or "must have" in text.lower()
        }
    
    def _extract_location_salary(self, text: str) -> Dict[str, Any]:
        """Extract location and salary information"""
        
        # Location
        location_patterns = [
            r'(?:location|based in|office in|work from)\s*:?\s*([A-Za-z\s&.,-]+?)(?:\n|$|,|\.)',
            r'(?:remote|hybrid|onsite|work from home)',
            r'(?:San Francisco|New York|Los Angeles|Chicago|Boston|Seattle|Austin|Denver|Portland|Miami|Atlanta|Dallas|Houston|Phoenix|Las Vegas|San Diego|San Jose|Oakland|Berkeley|Palo Alto|Mountain View|Sunnyvale|Cupertino|Redwood City|Fremont|Hayward|Richmond|Concord|Vallejo|Fairfield|Vacaville|Napa|Santa Rosa|Petaluma|Healdsburg|Cloverdale|Geyserville|Calistoga|St. Helena|Yountville|Rutherford|Oakville|Stags Leap|Atlas Peak|Howell Mountain|Spring Mountain|Diamond Mountain|Mount Veeder|Chiles Valley|Wild Horse Valley|Los Carneros|Sonoma Valley|Russian River Valley|Dry Creek Valley|Alexander Valley|Knights Valley|Chalk Hill|Green Valley|Sonoma Coast|Fort Ross-Seaview|Petaluma Gap|Bennett Valley|Moon Mountain|Sonoma Mountain|Carneros|Sonoma Valley|Russian River Valley|Dry Creek Valley|Alexander Valley|Knights Valley|Chalk Hill|Green Valley|Sonoma Coast|Fort Ross-Seaview|Petaluma Gap|Bennett Valley|Moon Mountain|Sonoma Mountain)'
        ]
        
        location = ""
        work_type = ""
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if "remote" in match.group(0).lower() or "hybrid" in match.group(0).lower() or "onsite" in match.group(0).lower():
                    work_type = match.group(0).lower()
                else:
                    location = match.group(1).strip()
        
        # Salary
        salary_patterns = [
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:-\s*\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?))?\s*(?:per\s*(?:year|annum|hour|month))?',
            r'(?:salary|compensation|pay|wage)\s*:?\s*\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:-\s*\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?))?',
            r'(?:competitive|market rate|negotiable|based on experience)'
        ]
        
        salary_min = 0
        salary_max = 0
        salary_type = ""
        
        for pattern in salary_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if match.group(1):
                    salary_min = int(match.group(1).replace(',', ''))
                    if match.group(2):
                        salary_max = int(match.group(2).replace(',', ''))
                else:
                    salary_type = match.group(0).lower()
        
        return {
            "location": location,
            "work_type": work_type,
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_type": salary_type
        }
    
    def _extract_culture_indicators(self, text: str) -> Dict[str, Any]:
        """Extract company culture indicators"""
        
        culture_patterns = [
            r'(?:startup|fast-paced|dynamic|innovative|collaborative|inclusive|diverse|flexible|remote-first|work-life balance|growth|learning|mentorship|team-oriented|entrepreneurial|creative|agile|transparent|open|casual|professional|formal)'
        ]
        
        culture_keywords = []
        for pattern in culture_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            culture_keywords.extend(matches)
        
        # Benefits
        benefits_patterns = [
            r'(?:health insurance|dental|vision|401k|retirement|vacation|pto|paid time off|flexible hours|remote work|home office|laptop|equipment|training|conference|education|gym|wellness|snacks|meals|transportation|parking|bonus|equity|stock options|profit sharing)'
        ]
        
        benefits = []
        for pattern in benefits_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            benefits.extend(matches)
        
        return {
            "culture_keywords": list(set(culture_keywords)),
            "benefits": list(set(benefits))
        }
    
    def _ai_analyze_job_description(self, text: str) -> Dict[str, Any]:
        """Use AI to analyze job description for advanced insights"""
        
        system_prompt = """
        You are an expert job description analyst with deep understanding of talent acquisition and job market trends.
        Your role is to provide comprehensive analysis of job descriptions for better candidate matching.
        
        Analyze the job description and provide:
        1. Job complexity and seniority level
        2. Required vs nice-to-have skills
        3. Company culture and values
        4. Growth opportunities
        5. Potential challenges
        6. Ideal candidate profile
        7. Market competitiveness
        """
        
        human_prompt = f"""
        Job Description:
        {text}
        
        Please provide a comprehensive analysis in JSON format:
        {{
            "job_complexity": "entry/mid/senior/lead/principal",
            "seniority_level": "junior/mid/senior/lead/principal/architect",
            "required_skills": ["list of must-have skills"],
            "nice_to_have_skills": ["list of preferred skills"],
            "company_culture": "description of company culture",
            "growth_opportunities": ["list of growth opportunities"],
            "potential_challenges": ["list of potential challenges"],
            "ideal_candidate_profile": "description of ideal candidate",
            "market_competitiveness": "high/medium/low",
            "salary_benchmark": "estimated salary range",
            "key_requirements": ["list of key requirements"],
            "red_flags": ["list of any red flags"],
            "recommendations": "recommendations for job posting improvement"
        }}
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            result = self._parse_json_response(response.content)
            return result
            
        except Exception as e:
            logger.error(f"AI analysis error: {str(e)}")
            return {"error": str(e)}
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from AI"""
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
    
    def calculate_job_complexity_score(self, job_analysis: Dict[str, Any]) -> float:
        """Calculate job complexity score based on analysis"""
        
        score = 0
        
        # Experience requirements
        experience = job_analysis.get("experience", {})
        years_required = experience.get("years_required", 0)
        if years_required >= 5:
            score += 30
        elif years_required >= 3:
            score += 20
        elif years_required >= 1:
            score += 10
        
        # Education requirements
        education = job_analysis.get("education", {})
        if education.get("required", False):
            score += 15
        
        # Technical skills complexity
        skills = job_analysis.get("skills", {})
        technical_skills = skills.get("technical", [])
        if len(technical_skills) >= 10:
            score += 25
        elif len(technical_skills) >= 5:
            score += 15
        elif len(technical_skills) >= 3:
            score += 10
        
        # AI analysis
        ai_analysis = job_analysis.get("ai_analysis", {})
        seniority = ai_analysis.get("seniority_level", "").lower()
        if "senior" in seniority or "lead" in seniority or "principal" in seniority:
            score += 20
        elif "mid" in seniority:
            score += 10
        
        return min(score, 100)  # Cap at 100