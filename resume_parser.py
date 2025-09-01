import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

# Document parsing libraries
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from textstat import flesch_reading_ease, flesch_kincaid_grade

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedResumeParser:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse resume file and extract structured information"""
        
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.pdf':
                text = self._extract_pdf_text(file_path)
            elif file_ext == '.docx':
                text = self._extract_docx_text(file_path)
            elif file_ext in ['.txt', '.doc']:
                text = self._extract_text_file(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            if not text.strip():
                return {"error": "No text extracted from file"}
            
            # Parse the extracted text
            parsed_data = self._parse_resume_text(text)
            parsed_data["raw_text"] = text
            parsed_data["file_path"] = file_path
            parsed_data["file_type"] = file_ext
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            return {"error": str(e), "file_path": file_path}
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            return ""
    
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {str(e)}")
            return ""
    
    def _extract_text_file(self, file_path: str) -> str:
        """Extract text from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Error reading text file: {str(e)}")
                return ""
    
    def _parse_resume_text(self, text: str) -> Dict[str, Any]:
        """Parse resume text and extract structured information"""
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Extract different sections
        sections = self._extract_sections(cleaned_text)
        
        # Parse each section
        parsed_data = {
            "personal_info": self._extract_personal_info(cleaned_text),
            "contact_info": self._extract_contact_info(cleaned_text),
            "summary": self._extract_summary(sections),
            "experience": self._extract_experience(sections),
            "education": self._extract_education(sections),
            "skills": self._extract_skills(sections, cleaned_text),
            "certifications": self._extract_certifications(sections),
            "projects": self._extract_projects(sections),
            "languages": self._extract_languages(sections),
            "awards": self._extract_awards(sections),
            "publications": self._extract_publications(sections),
            "volunteer": self._extract_volunteer(sections),
            "text_analysis": self._analyze_text_quality(cleaned_text)
        }
        
        return parsed_data
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        return text.strip()
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract different sections of the resume"""
        
        # Common section headers
        section_patterns = {
            'summary': r'(?:summary|profile|objective|about)\s*:?\s*(.*?)(?=\n\s*(?:experience|education|skills|work|employment))',
            'experience': r'(?:experience|work\s*history|employment|professional\s*experience)\s*:?\s*(.*?)(?=\n\s*(?:education|skills|certification|project))',
            'education': r'(?:education|academic|qualification)\s*:?\s*(.*?)(?=\n\s*(?:skills|experience|certification|project))',
            'skills': r'(?:skills|technical\s*skills|competencies)\s*:?\s*(.*?)(?=\n\s*(?:experience|education|certification|project))',
            'certifications': r'(?:certification|certificate|license)\s*:?\s*(.*?)(?=\n\s*(?:experience|education|skills|project))',
            'projects': r'(?:project|portfolio)\s*:?\s*(.*?)(?=\n\s*(?:experience|education|skills|certification))',
            'languages': r'(?:language|linguistic)\s*:?\s*(.*?)(?=\n\s*(?:experience|education|skills|certification))',
            'awards': r'(?:award|honor|recognition)\s*:?\s*(.*?)(?=\n\s*(?:experience|education|skills|certification))',
            'publications': r'(?:publication|paper|article)\s*:?\s*(.*?)(?=\n\s*(?:experience|education|skills|certification))',
            'volunteer': r'(?:volunteer|community\s*service)\s*:?\s*(.*?)(?=\n\s*(?:experience|education|skills|certification))'
        }
        
        sections = {}
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()
        
        return sections
    
    def _extract_personal_info(self, text: str) -> Dict[str, str]:
        """Extract personal information"""
        
        # Name extraction (usually at the top)
        name_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        name_match = re.search(name_pattern, text)
        name = name_match.group(1) if name_match else ""
        
        return {
            "name": name,
            "full_name": name
        }
    
    def _extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information"""
        
        # Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        email = email_match.group(0) if email_match else ""
        
        # Phone
        phone_pattern = r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phone_match = re.search(phone_pattern, text)
        phone = phone_match.group(0) if phone_match else ""
        
        # LinkedIn
        linkedin_pattern = r'(?:linkedin\.com/in/|linkedin\.com/pub/)([a-zA-Z0-9-]+)'
        linkedin_match = re.search(linkedin_pattern, text)
        linkedin = linkedin_match.group(0) if linkedin_match else ""
        
        # GitHub
        github_pattern = r'(?:github\.com/)([a-zA-Z0-9-]+)'
        github_match = re.search(github_pattern, text)
        github = github_match.group(0) if github_match else ""
        
        return {
            "email": email,
            "phone": phone,
            "linkedin": linkedin,
            "github": github
        }
    
    def _extract_summary(self, sections: Dict[str, str]) -> str:
        """Extract professional summary"""
        return sections.get('summary', '')
    
    def _extract_experience(self, sections: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract work experience"""
        
        experience_text = sections.get('experience', '')
        if not experience_text:
            return []
        
        # Split by common job separators
        jobs = re.split(r'\n\s*(?=[A-Z][a-z]+.*\d{4})', experience_text)
        
        experiences = []
        for job in jobs:
            if not job.strip():
                continue
                
            # Extract job title
            title_pattern = r'^([A-Z][a-zA-Z\s]+(?:Engineer|Developer|Manager|Analyst|Consultant|Specialist|Coordinator|Director|Lead|Senior|Junior|Associate))'
            title_match = re.search(title_pattern, job)
            title = title_match.group(1).strip() if title_match else ""
            
            # Extract company
            company_pattern = r'at\s+([A-Za-z0-9\s&.,-]+?)(?:\s*[-–]\s*|\s*\(|\s*\d)'
            company_match = re.search(company_pattern, job, re.IGNORECASE)
            company = company_match.group(1).strip() if company_match else ""
            
            # Extract dates
            date_pattern = r'(\d{4})\s*[-–]\s*(\d{4}|present|current)'
            date_match = re.search(date_pattern, job, re.IGNORECASE)
            start_date = date_match.group(1) if date_match else ""
            end_date = date_match.group(2) if date_match else ""
            
            # Extract description
            description = job.replace(title, '').replace(company, '').strip()
            description = re.sub(r'\d{4}\s*[-–]\s*(?:\d{4}|present|current)', '', description, flags=re.IGNORECASE).strip()
            
            if title or company:  # Only add if we found meaningful data
                experiences.append({
                    "title": title,
                    "company": company,
                    "start_date": start_date,
                    "end_date": end_date,
                    "description": description,
                    "duration": self._calculate_duration(start_date, end_date)
                })
        
        return experiences
    
    def _extract_education(self, sections: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract education information"""
        
        education_text = sections.get('education', '')
        if not education_text:
            return []
        
        # Split by degree separators
        degrees = re.split(r'\n\s*(?=[A-Z][a-z]+.*\d{4})', education_text)
        
        education_list = []
        for degree in degrees:
            if not degree.strip():
                continue
            
            # Extract degree
            degree_pattern = r'(Bachelor|Master|PhD|Doctorate|Associate|Certificate|Diploma|B\.?S\.?|M\.?S\.?|B\.?A\.?|M\.?A\.?|B\.?E\.?|M\.?E\.?)'
            degree_match = re.search(degree_pattern, degree, re.IGNORECASE)
            degree_name = degree_match.group(1) if degree_match else ""
            
            # Extract field of study
            field_pattern = r'(?:in|of)\s+([A-Za-z\s]+?)(?:\s*[-–]\s*|\s*\(|\s*\d)'
            field_match = re.search(field_pattern, degree, re.IGNORECASE)
            field = field_match.group(1).strip() if field_match else ""
            
            # Extract institution
            institution_pattern = r'([A-Za-z\s&.,-]+?)(?:\s*[-–]\s*|\s*\(|\s*\d)'
            institution_match = re.search(institution_pattern, degree)
            institution = institution_match.group(1).strip() if institution_match else ""
            
            # Extract graduation year
            year_pattern = r'(\d{4})'
            year_match = re.search(year_pattern, degree)
            graduation_year = year_match.group(1) if year_match else ""
            
            if degree_name or institution:  # Only add if we found meaningful data
                education_list.append({
                    "degree": degree_name,
                    "field": field,
                    "institution": institution,
                    "graduation_year": graduation_year,
                    "gpa": self._extract_gpa(degree)
                })
        
        return education_list
    
    def _extract_skills(self, sections: Dict[str, str], full_text: str) -> Dict[str, List[str]]:
        """Extract skills and categorize them"""
        
        skills_text = sections.get('skills', '')
        if not skills_text:
            # Try to extract skills from full text
            skills_text = self._extract_skills_from_text(full_text)
        
        # Common skill categories
        skill_categories = {
            'programming_languages': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust', 'scala', 'r', 'matlab'],
            'frameworks': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'laravel', 'rails', 'asp.net', 'node.js'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite', 'cassandra', 'elasticsearch'],
            'cloud_platforms': ['aws', 'azure', 'gcp', 'google cloud', 'amazon web services', 'microsoft azure'],
            'tools': ['git', 'docker', 'kubernetes', 'jenkins', 'ci/cd', 'terraform', 'ansible', 'jira', 'confluence'],
            'methodologies': ['agile', 'scrum', 'kanban', 'devops', 'tdd', 'bdd', 'microservices', 'api development'],
            'soft_skills': ['leadership', 'communication', 'teamwork', 'problem solving', 'project management', 'mentoring']
        }
        
        # Extract skills from text
        found_skills = self._extract_skills_from_text(skills_text)
        
        # Categorize skills
        categorized_skills = {category: [] for category in skill_categories.keys()}
        
        for skill in found_skills:
            skill_lower = skill.lower()
            for category, keywords in skill_categories.items():
                if any(keyword in skill_lower for keyword in keywords):
                    categorized_skills[category].append(skill)
                    break
            else:
                # If not categorized, add to general skills
                if 'general' not in categorized_skills:
                    categorized_skills['general'] = []
                categorized_skills['general'].append(skill)
        
        return categorized_skills
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text using various patterns"""
        
        skills = []
        
        # Common skill patterns
        skill_patterns = [
            r'\b[A-Z][a-z]+\s*(?:\.js|\.net|\.py|\.java|\.cpp|\.cs)\b',  # Technologies with extensions
            r'\b(?:HTML|CSS|JavaScript|Python|Java|C\+\+|C#|Ruby|PHP|Swift|Kotlin|Go|Rust|Scala|R|MATLAB)\b',
            r'\b(?:React|Angular|Vue|Django|Flask|Spring|Express|Laravel|Rails|ASP\.NET|Node\.js)\b',
            r'\b(?:MySQL|PostgreSQL|MongoDB|Redis|Oracle|SQLite|Cassandra|Elasticsearch)\b',
            r'\b(?:AWS|Azure|GCP|Google Cloud|Amazon Web Services|Microsoft Azure)\b',
            r'\b(?:Git|Docker|Kubernetes|Jenkins|CI/CD|Terraform|Ansible|Jira|Confluence)\b',
            r'\b(?:Agile|Scrum|Kanban|DevOps|TDD|BDD|Microservices|API Development)\b'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend(matches)
        
        # Remove duplicates and clean up
        skills = list(set([skill.strip() for skill in skills if skill.strip()]))
        
        return skills
    
    def _extract_certifications(self, sections: Dict[str, str]) -> List[Dict[str, str]]:
        """Extract certifications"""
        
        cert_text = sections.get('certifications', '')
        if not cert_text:
            return []
        
        # Split by certification separators
        certs = re.split(r'\n\s*(?=[A-Z])', cert_text)
        
        certifications = []
        for cert in certs:
            if not cert.strip():
                continue
            
            # Extract certification name
            cert_name = cert.split(' - ')[0].strip() if ' - ' in cert else cert.strip()
            
            # Extract issuer
            issuer_pattern = r'(?:from|by|issued by)\s+([A-Za-z\s&.,-]+)'
            issuer_match = re.search(issuer_pattern, cert, re.IGNORECASE)
            issuer = issuer_match.group(1).strip() if issuer_match else ""
            
            # Extract date
            date_pattern = r'(\d{4})'
            date_match = re.search(date_pattern, cert)
            date = date_match.group(1) if date_match else ""
            
            if cert_name:
                certifications.append({
                    "name": cert_name,
                    "issuer": issuer,
                    "date": date
                })
        
        return certifications
    
    def _extract_projects(self, sections: Dict[str, str]) -> List[Dict[str, str]]:
        """Extract projects"""
        
        projects_text = sections.get('projects', '')
        if not projects_text:
            return []
        
        # Split by project separators
        projects = re.split(r'\n\s*(?=[A-Z])', projects_text)
        
        project_list = []
        for project in projects:
            if not project.strip():
                continue
            
            # Extract project name
            project_name = project.split(' - ')[0].strip() if ' - ' in project else project.strip()
            
            # Extract description
            description = project.replace(project_name, '').strip()
            if description.startswith(' - '):
                description = description[3:]
            
            if project_name:
                project_list.append({
                    "name": project_name,
                    "description": description
                })
        
        return project_list
    
    def _extract_languages(self, sections: Dict[str, str]) -> List[Dict[str, str]]:
        """Extract languages"""
        
        languages_text = sections.get('languages', '')
        if not languages_text:
            return []
        
        # Split by language separators
        languages = re.split(r'[,;\n]', languages_text)
        
        language_list = []
        for lang in languages:
            if not lang.strip():
                continue
            
            # Extract proficiency level
            proficiency_pattern = r'(native|fluent|advanced|intermediate|beginner|basic)'
            proficiency_match = re.search(proficiency_pattern, lang, re.IGNORECASE)
            proficiency = proficiency_match.group(1).lower() if proficiency_match else ""
            
            # Extract language name
            language_name = lang.replace(proficiency, '').strip()
            if language_name:
                language_list.append({
                    "language": language_name,
                    "proficiency": proficiency
                })
        
        return language_list
    
    def _extract_awards(self, sections: Dict[str, str]) -> List[str]:
        """Extract awards and honors"""
        
        awards_text = sections.get('awards', '')
        if not awards_text:
            return []
        
        # Split by award separators
        awards = re.split(r'\n\s*(?=[A-Z])', awards_text)
        
        return [award.strip() for award in awards if award.strip()]
    
    def _extract_publications(self, sections: Dict[str, str]) -> List[Dict[str, str]]:
        """Extract publications"""
        
        publications_text = sections.get('publications', '')
        if not publications_text:
            return []
        
        # Split by publication separators
        publications = re.split(r'\n\s*(?=[A-Z])', publications_text)
        
        publication_list = []
        for pub in publications:
            if not pub.strip():
                continue
            
            # Extract title
            title = pub.split(' - ')[0].strip() if ' - ' in pub else pub.strip()
            
            # Extract venue
            venue_pattern = r'(?:in|published in)\s+([A-Za-z\s&.,-]+)'
            venue_match = re.search(venue_pattern, pub, re.IGNORECASE)
            venue = venue_match.group(1).strip() if venue_match else ""
            
            # Extract date
            date_pattern = r'(\d{4})'
            date_match = re.search(date_pattern, pub)
            date = date_match.group(1) if date_match else ""
            
            if title:
                publication_list.append({
                    "title": title,
                    "venue": venue,
                    "date": date
                })
        
        return publication_list
    
    def _extract_volunteer(self, sections: Dict[str, str]) -> List[Dict[str, str]]:
        """Extract volunteer experience"""
        
        volunteer_text = sections.get('volunteer', '')
        if not volunteer_text:
            return []
        
        # Split by volunteer separators
        volunteer_experiences = re.split(r'\n\s*(?=[A-Z])', volunteer_text)
        
        volunteer_list = []
        for volunteer in volunteer_experiences:
            if not volunteer.strip():
                continue
            
            # Extract organization
            org_pattern = r'([A-Za-z\s&.,-]+?)(?:\s*[-–]\s*|\s*\(|\s*\d)'
            org_match = re.search(org_pattern, volunteer)
            organization = org_match.group(1).strip() if org_match else ""
            
            # Extract role
            role_pattern = r'(?:as|role:)\s+([A-Za-z\s&.,-]+)'
            role_match = re.search(role_pattern, volunteer, re.IGNORECASE)
            role = role_match.group(1).strip() if role_match else ""
            
            # Extract description
            description = volunteer.replace(organization, '').replace(role, '').strip()
            
            if organization:
                volunteer_list.append({
                    "organization": organization,
                    "role": role,
                    "description": description
                })
        
        return volunteer_list
    
    def _extract_gpa(self, text: str) -> str:
        """Extract GPA from education text"""
        
        gpa_pattern = r'GPA[:\s]*(\d+\.?\d*)'
        gpa_match = re.search(gpa_pattern, text, re.IGNORECASE)
        return gpa_match.group(1) if gpa_match else ""
    
    def _calculate_duration(self, start_date: str, end_date: str) -> str:
        """Calculate duration between dates"""
        
        try:
            if not start_date:
                return ""
            
            start_year = int(start_date)
            if end_date.lower() in ['present', 'current']:
                end_year = 2024  # Current year
            else:
                end_year = int(end_date)
            
            duration = end_year - start_year
            if duration == 0:
                return "Less than 1 year"
            elif duration == 1:
                return "1 year"
            else:
                return f"{duration} years"
                
        except (ValueError, TypeError):
            return ""
    
    def _analyze_text_quality(self, text: str) -> Dict[str, Any]:
        """Analyze text quality and readability"""
        
        try:
            # Basic text statistics
            word_count = len(text.split())
            sentence_count = len(sent_tokenize(text))
            avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
            
            # Readability scores
            flesch_score = flesch_reading_ease(text)
            fk_grade = flesch_kincaid_grade(text)
            
            # Keyword density analysis
            words = word_tokenize(text.lower())
            word_freq = {}
            for word in words:
                if word.isalpha() and word not in self.stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Top keywords
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_words_per_sentence": round(avg_words_per_sentence, 2),
                "flesch_reading_ease": round(flesch_score, 2),
                "flesch_kincaid_grade": round(fk_grade, 2),
                "top_keywords": [{"word": word, "count": count} for word, count in top_keywords],
                "readability_level": self._get_readability_level(flesch_score)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text quality: {str(e)}")
            return {"error": str(e)}
    
    def _get_readability_level(self, flesch_score: float) -> str:
        """Get readability level based on Flesch score"""
        
        if flesch_score >= 90:
            return "Very Easy"
        elif flesch_score >= 80:
            return "Easy"
        elif flesch_score >= 70:
            return "Fairly Easy"
        elif flesch_score >= 60:
            return "Standard"
        elif flesch_score >= 50:
            return "Fairly Difficult"
        elif flesch_score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"