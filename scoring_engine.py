import math
from typing import Dict, List, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedScoringEngine:
    def __init__(self):
        self.weights = settings.SCORING_WEIGHTS
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def calculate_comprehensive_score(self, 
                                    resume_data: Dict[str, Any], 
                                    job_analysis: Dict[str, Any],
                                    ai_agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive matching score using multiple algorithms"""
        
        try:
            # Individual component scores
            skills_score = self._calculate_skills_score(resume_data, job_analysis)
            experience_score = self._calculate_experience_score(resume_data, job_analysis)
            education_score = self._calculate_education_score(resume_data, job_analysis)
            cultural_fit_score = self._calculate_cultural_fit_score(resume_data, job_analysis)
            technical_score = self._calculate_technical_score(resume_data, job_analysis)
            
            # AI agent scores
            ai_scores = self._extract_ai_scores(ai_agent_results)
            
            # Semantic similarity score
            semantic_score = self._calculate_semantic_similarity(resume_data, job_analysis)
            
            # Weighted overall score
            weighted_score = self._calculate_weighted_score({
                'skills_match': skills_score,
                'experience_relevance': experience_score,
                'education': education_score,
                'cultural_fit': cultural_fit_score,
                'technical_skills': technical_score,
                'semantic_similarity': semantic_score
            })
            
            # AI-enhanced score
            ai_enhanced_score = self._calculate_ai_enhanced_score(weighted_score, ai_scores)
            
            # Final score with confidence
            final_score = self._calculate_final_score(ai_enhanced_score, ai_scores)
            
            return {
                'overall_score': final_score,
                'component_scores': {
                    'skills_match': skills_score,
                    'experience_relevance': experience_score,
                    'education': education_score,
                    'cultural_fit': cultural_fit_score,
                    'technical_skills': technical_score,
                    'semantic_similarity': semantic_score
                },
                'ai_scores': ai_scores,
                'weighted_score': weighted_score,
                'ai_enhanced_score': ai_enhanced_score,
                'confidence': self._calculate_confidence(ai_scores),
                'recommendations': self._generate_recommendations(resume_data, job_analysis, ai_scores)
            }
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive score: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_skills_score(self, resume_data: Dict[str, Any], job_analysis: Dict[str, Any]) -> float:
        """Calculate skills matching score"""
        
        resume_skills = resume_data.get('skills', {})
        job_skills = job_analysis.get('skills', {})
        
        # Extract all skills from resume
        all_resume_skills = []
        for category, skills in resume_skills.items():
            if isinstance(skills, list):
                all_resume_skills.extend([skill.lower() for skill in skills])
        
        # Extract required skills from job
        required_skills = job_skills.get('technical', []) + job_skills.get('tools', [])
        required_skills = [skill.lower() for skill in required_skills]
        
        # Extract preferred skills
        preferred_skills = job_skills.get('soft', [])
        preferred_skills = [skill.lower() for skill in preferred_skills]
        
        if not required_skills and not preferred_skills:
            return 50  # Neutral score if no skills specified
        
        # Calculate required skills match
        required_match = 0
        if required_skills:
            matched_required = sum(1 for skill in required_skills if skill in all_resume_skills)
            required_match = (matched_required / len(required_skills)) * 100
        
        # Calculate preferred skills match
        preferred_match = 0
        if preferred_skills:
            matched_preferred = sum(1 for skill in preferred_skills if skill in all_resume_skills)
            preferred_match = (matched_preferred / len(preferred_skills)) * 100
        
        # Weighted combination (required skills are more important)
        if required_skills and preferred_skills:
            return (required_match * 0.7) + (preferred_match * 0.3)
        elif required_skills:
            return required_match
        else:
            return preferred_match
    
    def _calculate_experience_score(self, resume_data: Dict[str, Any], job_analysis: Dict[str, Any]) -> float:
        """Calculate experience relevance score"""
        
        resume_experience = resume_data.get('experience', [])
        job_experience = job_analysis.get('experience', {})
        
        if not resume_experience:
            return 0
        
        # Job requirements
        years_required = job_experience.get('years_required', 0)
        experience_type = job_experience.get('type', [])
        level = job_experience.get('level', '')
        
        # Calculate total experience
        total_experience_years = 0
        relevant_experience_years = 0
        
        for exp in resume_experience:
            duration = exp.get('duration', '')
            if 'year' in duration.lower():
                try:
                    years = int(duration.split()[0])
                    total_experience_years += years
                    
                    # Check if experience is relevant
                    title = exp.get('title', '').lower()
                    company = exp.get('company', '').lower()
                    description = exp.get('description', '').lower()
                    
                    # Check for relevance based on job type
                    is_relevant = False
                    for exp_type in experience_type:
                        if exp_type.lower() in title or exp_type.lower() in description:
                            is_relevant = True
                            break
                    
                    if is_relevant:
                        relevant_experience_years += years
                        
                except (ValueError, IndexError):
                    continue
        
        # Calculate score based on requirements
        if years_required == 0:
            return 50  # Neutral if no experience requirement
        
        # Base score on total experience
        experience_ratio = min(total_experience_years / years_required, 2.0)  # Cap at 2x requirement
        base_score = (experience_ratio / 2.0) * 100
        
        # Bonus for relevant experience
        if relevant_experience_years > 0:
            relevance_bonus = min((relevant_experience_years / total_experience_years) * 20, 20)
            base_score += relevance_bonus
        
        # Level matching bonus
        if level:
            level_bonus = self._calculate_level_bonus(resume_experience, level)
            base_score += level_bonus
        
        return min(base_score, 100)
    
    def _calculate_level_bonus(self, experience: List[Dict[str, Any]], required_level: str) -> float:
        """Calculate bonus for level matching"""
        
        level_keywords = {
            'entry-level': ['junior', 'entry', 'associate', 'trainee', 'intern'],
            'mid-level': ['mid', 'intermediate', 'experienced'],
            'senior': ['senior', 'lead', 'principal', 'architect', 'staff'],
            'lead': ['lead', 'principal', 'architect', 'staff', 'senior'],
            'principal': ['principal', 'architect', 'staff', 'senior']
        }
        
        required_keywords = level_keywords.get(required_level.lower(), [])
        
        for exp in experience:
            title = exp.get('title', '').lower()
            for keyword in required_keywords:
                if keyword in title:
                    return 10  # Bonus for level match
        
        return 0
    
    def _calculate_education_score(self, resume_data: Dict[str, Any], job_analysis: Dict[str, Any]) -> float:
        """Calculate education matching score"""
        
        resume_education = resume_data.get('education', [])
        job_education = job_analysis.get('education', {})
        
        if not resume_education:
            return 0
        
        # Job requirements
        education_required = job_education.get('required', False)
        required_level = job_education.get('level', '')
        required_fields = job_education.get('fields', [])
        
        if not education_required and not required_level:
            return 50  # Neutral if no education requirement
        
        # Check education level
        level_score = 0
        if required_level:
            level_score = self._check_education_level(resume_education, required_level)
        
        # Check field relevance
        field_score = 0
        if required_fields:
            field_score = self._check_education_field(resume_education, required_fields)
        
        # Combine scores
        if level_score > 0 and field_score > 0:
            return (level_score + field_score) / 2
        elif level_score > 0:
            return level_score
        elif field_score > 0:
            return field_score
        else:
            return 0
    
    def _check_education_level(self, education: List[Dict[str, Any]], required_level: str) -> float:
        """Check if education level meets requirements"""
        
        level_hierarchy = {
            'certificate': 1,
            'diploma': 2,
            'associate': 3,
            'bachelor': 4,
            'master': 5,
            'phd': 6,
            'doctorate': 6
        }
        
        required_level_num = level_hierarchy.get(required_level.lower(), 0)
        
        for edu in education:
            degree = edu.get('degree', '').lower()
            for level, num in level_hierarchy.items():
                if level in degree and num >= required_level_num:
                    return 100
        
        return 0
    
    def _check_education_field(self, education: List[Dict[str, Any]], required_fields: List[str]) -> float:
        """Check if education field is relevant"""
        
        for edu in education:
            field = edu.get('field', '').lower()
            degree = edu.get('degree', '').lower()
            
            for required_field in required_fields:
                required_field_lower = required_field.lower()
                if (required_field_lower in field or 
                    required_field_lower in degree or
                    self._is_field_related(field, required_field_lower)):
                    return 100
        
        return 0
    
    def _is_field_related(self, field: str, required_field: str) -> bool:
        """Check if education field is related to required field"""
        
        field_mappings = {
            'computer science': ['cs', 'computing', 'software engineering', 'information technology', 'it'],
            'engineering': ['mechanical', 'electrical', 'civil', 'chemical', 'software'],
            'business': ['management', 'administration', 'finance', 'marketing', 'economics'],
            'mathematics': ['math', 'statistics', 'applied mathematics', 'pure mathematics'],
            'statistics': ['statistics', 'applied statistics', 'mathematical statistics']
        }
        
        for key, related_fields in field_mappings.items():
            if key in required_field:
                return any(related in field for related in related_fields)
        
        return False
    
    def _calculate_cultural_fit_score(self, resume_data: Dict[str, Any], job_analysis: Dict[str, Any]) -> float:
        """Calculate cultural fit score"""
        
        # Extract cultural indicators from resume
        resume_culture = self._extract_resume_culture(resume_data)
        
        # Extract cultural indicators from job
        job_culture = job_analysis.get('culture', {})
        culture_keywords = job_culture.get('culture_keywords', [])
        
        if not culture_keywords:
            return 50  # Neutral if no culture specified
        
        # Calculate cultural alignment
        alignment_score = 0
        for keyword in culture_keywords:
            if keyword.lower() in resume_culture:
                alignment_score += 1
        
        return (alignment_score / len(culture_keywords)) * 100
    
    def _extract_resume_culture(self, resume_data: Dict[str, Any]) -> str:
        """Extract cultural indicators from resume"""
        
        culture_text = ""
        
        # From summary
        summary = resume_data.get('summary', '')
        culture_text += summary + " "
        
        # From experience descriptions
        experience = resume_data.get('experience', [])
        for exp in experience:
            description = exp.get('description', '')
            culture_text += description + " "
        
        # From volunteer work
        volunteer = resume_data.get('volunteer', [])
        for vol in volunteer:
            description = vol.get('description', '')
            culture_text += description + " "
        
        return culture_text.lower()
    
    def _calculate_technical_score(self, resume_data: Dict[str, Any], job_analysis: Dict[str, Any]) -> float:
        """Calculate technical skills depth score"""
        
        resume_skills = resume_data.get('skills', {})
        job_skills = job_analysis.get('skills', {})
        
        # Get technical skills from resume
        technical_skills = resume_skills.get('technical', []) + resume_skills.get('programming_languages', [])
        technical_skills = [skill.lower() for skill in technical_skills]
        
        # Get required technical skills from job
        required_technical = job_skills.get('technical', [])
        required_technical = [skill.lower() for skill in required_technical]
        
        if not required_technical:
            return 50  # Neutral if no technical requirements
        
        # Calculate depth of technical skills
        depth_score = 0
        for skill in required_technical:
            if skill in technical_skills:
                depth_score += 1
        
        # Bonus for additional technical skills
        additional_skills = len(technical_skills) - len(required_technical)
        if additional_skills > 0:
            depth_score += min(additional_skills * 0.5, 10)
        
        return min((depth_score / len(required_technical)) * 100, 100)
    
    def _calculate_semantic_similarity(self, resume_data: Dict[str, Any], job_analysis: Dict[str, Any]) -> float:
        """Calculate semantic similarity between resume and job description"""
        
        try:
            # Combine resume text
            resume_text = self._combine_resume_text(resume_data)
            
            # Get job description text
            job_text = job_analysis.get('raw_text', '')
            
            if not resume_text or not job_text:
                return 50  # Neutral if no text available
            
            # Vectorize texts
            texts = [resume_text, job_text]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity * 100
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 50
    
    def _combine_resume_text(self, resume_data: Dict[str, Any]) -> str:
        """Combine resume text for semantic analysis"""
        
        text_parts = []
        
        # Add summary
        summary = resume_data.get('summary', '')
        if summary:
            text_parts.append(summary)
        
        # Add experience descriptions
        experience = resume_data.get('experience', [])
        for exp in experience:
            description = exp.get('description', '')
            if description:
                text_parts.append(description)
        
        # Add skills
        skills = resume_data.get('skills', {})
        for category, skill_list in skills.items():
            if isinstance(skill_list, list):
                text_parts.append(' '.join(skill_list))
        
        # Add education
        education = resume_data.get('education', [])
        for edu in education:
            field = edu.get('field', '')
            if field:
                text_parts.append(field)
        
        return ' '.join(text_parts)
    
    def _extract_ai_scores(self, ai_agent_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract scores from AI agent results"""
        
        scores = {}
        
        # Screener score
        screener = ai_agent_results.get('screener', {})
        if 'screening_score' in screener:
            scores['screener'] = float(screener['screening_score'])
        
        # Technical evaluator score
        technical = ai_agent_results.get('technical_evaluator', {})
        if 'technical_score' in technical:
            scores['technical'] = float(technical['technical_score'])
        
        # Cultural fit score
        cultural = ai_agent_results.get('cultural_fit_assessor', {})
        if 'cultural_fit_score' in cultural:
            scores['cultural'] = float(cultural['cultural_fit_score'])
        
        # Bias detection score
        bias = ai_agent_results.get('bias_detection', {})
        if 'bias_risk_score' in bias:
            scores['bias_risk'] = float(bias['bias_risk_score'])
        
        return scores
    
    def _calculate_weighted_score(self, component_scores: Dict[str, float]) -> float:
        """Calculate weighted score from component scores"""
        
        total_score = 0
        total_weight = 0
        
        for component, score in component_scores.items():
            weight = self.weights.get(component, 0.1)  # Default weight
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _calculate_ai_enhanced_score(self, weighted_score: float, ai_scores: Dict[str, float]) -> float:
        """Calculate AI-enhanced score"""
        
        if not ai_scores:
            return weighted_score
        
        # Calculate AI average
        ai_average = sum(ai_scores.values()) / len(ai_scores)
        
        # Combine weighted score with AI score
        return (weighted_score * 0.7) + (ai_average * 0.3)
    
    def _calculate_final_score(self, ai_enhanced_score: float, ai_scores: Dict[str, float]) -> float:
        """Calculate final score with confidence adjustment"""
        
        # Base score
        base_score = ai_enhanced_score
        
        # Confidence adjustment based on AI scores consistency
        if ai_scores:
            scores = list(ai_scores.values())
            if len(scores) > 1:
                std_dev = np.std(scores)
                if std_dev < 10:  # High consistency
                    base_score *= 1.05  # Slight boost
                elif std_dev > 20:  # Low consistency
                    base_score *= 0.95  # Slight reduction
        
        return min(max(base_score, 0), 100)
    
    def _calculate_confidence(self, ai_scores: Dict[str, float]) -> float:
        """Calculate confidence in the scoring"""
        
        if not ai_scores:
            return 0.5
        
        # Calculate confidence based on score consistency
        scores = list(ai_scores.values())
        if len(scores) > 1:
            std_dev = np.std(scores)
            # Lower standard deviation = higher confidence
            confidence = max(0.3, 1 - (std_dev / 50))
        else:
            confidence = 0.7
        
        return confidence
    
    def _generate_recommendations(self, resume_data: Dict[str, Any], 
                                job_analysis: Dict[str, Any], 
                                ai_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on analysis"""
        
        recommendations = []
        
        # Skills recommendations
        resume_skills = resume_data.get('skills', {})
        job_skills = job_analysis.get('skills', {})
        
        required_skills = job_skills.get('technical', [])
        resume_technical = resume_skills.get('technical', [])
        
        missing_skills = []
        for skill in required_skills:
            if skill.lower() not in [s.lower() for s in resume_technical]:
                missing_skills.append(skill)
        
        if missing_skills:
            recommendations.append(f"Consider gaining experience with: {', '.join(missing_skills[:3])}")
        
        # Experience recommendations
        experience = resume_data.get('experience', [])
        job_experience = job_analysis.get('experience', {})
        years_required = job_experience.get('years_required', 0)
        
        if years_required > 0:
            total_years = sum(self._extract_years_from_duration(exp.get('duration', '')) for exp in experience)
            if total_years < years_required:
                recommendations.append(f"Consider gaining {years_required - total_years} more years of experience")
        
        # Education recommendations
        education = resume_data.get('education', [])
        job_education = job_analysis.get('education', {})
        
        if job_education.get('required', False):
            required_level = job_education.get('level', '')
            if required_level and not self._has_required_education(education, required_level):
                recommendations.append(f"Consider obtaining a {required_level} degree")
        
        # AI-based recommendations
        if ai_scores:
            screener_score = ai_scores.get('screener', 0)
            if screener_score < 70:
                recommendations.append("Focus on addressing basic qualification requirements")
            
            technical_score = ai_scores.get('technical', 0)
            if technical_score < 70:
                recommendations.append("Strengthen technical skills and project experience")
            
            cultural_score = ai_scores.get('cultural', 0)
            if cultural_score < 70:
                recommendations.append("Highlight cultural fit and soft skills in resume")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _extract_years_from_duration(self, duration: str) -> int:
        """Extract years from duration string"""
        
        if not duration:
            return 0
        
        try:
            if 'year' in duration.lower():
                return int(duration.split()[0])
            elif 'month' in duration.lower():
                return int(duration.split()[0]) / 12
            else:
                return 0
        except (ValueError, IndexError):
            return 0
    
    def _has_required_education(self, education: List[Dict[str, Any]], required_level: str) -> bool:
        """Check if candidate has required education level"""
        
        level_hierarchy = {
            'certificate': 1,
            'diploma': 2,
            'associate': 3,
            'bachelor': 4,
            'master': 5,
            'phd': 6,
            'doctorate': 6
        }
        
        required_level_num = level_hierarchy.get(required_level.lower(), 0)
        
        for edu in education:
            degree = edu.get('degree', '').lower()
            for level, num in level_hierarchy.items():
                if level in degree and num >= required_level_num:
                    return True
        
        return False