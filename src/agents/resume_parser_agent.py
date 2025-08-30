from typing import Dict, Any
from .base_agent import BaseAgent
from ..models.resume import ResumeData, Contact, Skills, Experience, Education

class ResumeParserAgent(BaseAgent):
    """AI agent specialized in parsing and structuring resume data"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        super().__init__(api_key, model, temperature=0.1)  # Lower temperature for structured parsing
    
    def process(self, resume_text: str, filename: str) -> ResumeData:
        """Parse resume text into structured data"""
        
        prompt_template = """
        You are an expert resume parser. Extract structured information from the following resume text.
        
        Return a JSON object with the following structure:
        {{
            "candidate_name": "Full name of the candidate",
            "contact": {{
                "email": "email address",
                "phone": "phone number",
                "linkedin": "LinkedIn URL",
                "github": "GitHub URL",
                "website": "personal website",
                "location": "city, state/country"
            }},
            "summary": "Professional summary or objective",
            "skills": {{
                "technical": ["list of technical skills"],
                "soft": ["list of soft skills"],
                "languages": ["programming languages"],
                "frameworks": ["frameworks and libraries"],
                "tools": ["tools and software"],
                "certifications": ["certifications and licenses"]
            }},
            "experience": [
                {{
                    "title": "job title",
                    "company": "company name",
                    "start_date": "start date",
                    "end_date": "end date or 'Present'",
                    "duration_months": estimated_duration_in_months,
                    "description": "job description",
                    "technologies": ["technologies used"],
                    "achievements": ["key achievements with metrics if available"]
                }}
            ],
            "education": [
                {{
                    "degree": "degree type and major",
                    "institution": "school name",
                    "graduation_year": year_or_null,
                    "gpa": gpa_or_null,
                    "relevant_coursework": ["relevant courses"]
                }}
            ],
            "projects": [
                {{
                    "name": "project name",
                    "description": "project description",
                    "technologies": ["technologies used"],
                    "url": "project URL if available"
                }}
            ]
        }}
        
        Resume Text:
        {resume_text}
        
        Extract all available information accurately. If information is not available, use null or empty arrays.
        """
        
        formatted_prompt = prompt_template.format(resume_text=resume_text[:50000])
        response = self._invoke_llm(formatted_prompt)
        parsed_data = self._parse_json_response(response)
        
        # Convert to ResumeData object
        return self._convert_to_resume_data(parsed_data, filename, resume_text)
    
    def _convert_to_resume_data(self, parsed_data: Dict[str, Any], filename: str, raw_text: str) -> ResumeData:
        """Convert parsed JSON data to ResumeData object"""
        try:
            # Handle contact information
            contact_data = parsed_data.get('contact', {})
            contact = Contact(
                email=contact_data.get('email'),
                phone=contact_data.get('phone'),
                linkedin=contact_data.get('linkedin'),
                github=contact_data.get('github'),
                website=contact_data.get('website'),
                location=contact_data.get('location')
            )
            
            # Handle skills
            skills_data = parsed_data.get('skills', {})
            skills = Skills(
                technical=skills_data.get('technical', []),
                soft=skills_data.get('soft', []),
                languages=skills_data.get('languages', []),
                frameworks=skills_data.get('frameworks', []),
                tools=skills_data.get('tools', []),
                certifications=skills_data.get('certifications', [])
            )
            
            # Handle experience
            experience_list = []
            for exp_data in parsed_data.get('experience', []):
                experience = Experience(
                    title=exp_data.get('title', ''),
                    company=exp_data.get('company', ''),
                    start_date=exp_data.get('start_date'),
                    end_date=exp_data.get('end_date'),
                    duration_months=exp_data.get('duration_months'),
                    description=exp_data.get('description', ''),
                    technologies=exp_data.get('technologies', []),
                    achievements=exp_data.get('achievements', [])
                )
                experience_list.append(experience)
            
            # Handle education
            education_list = []
            for edu_data in parsed_data.get('education', []):
                education = Education(
                    degree=edu_data.get('degree', ''),
                    institution=edu_data.get('institution', ''),
                    graduation_year=edu_data.get('graduation_year'),
                    gpa=edu_data.get('gpa'),
                    relevant_coursework=edu_data.get('relevant_coursework', [])
                )
                education_list.append(education)
            
            return ResumeData(
                filename=filename,
                candidate_name=parsed_data.get('candidate_name', 'Unknown'),
                contact=contact,
                summary=parsed_data.get('summary', ''),
                skills=skills,
                experience=experience_list,
                education=education_list,
                projects=parsed_data.get('projects', []),
                raw_text=raw_text
            )
            
        except Exception as e:
            self.logger.error(f"Error converting parsed data: {str(e)}")
            # Return minimal ResumeData object
            return ResumeData(
                filename=filename,
                candidate_name=parsed_data.get('candidate_name', 'Unknown'),
                raw_text=raw_text
            )