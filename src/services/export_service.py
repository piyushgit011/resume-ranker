import pandas as pd
import json
from typing import List, Dict, Any
from io import BytesIO
import xlsxwriter
from datetime import datetime
from ..models.resume import ResumeScore

class ExportService:
    """Service for exporting resume analysis results in various formats"""
    
    def __init__(self):
        pass
    
    def export_to_csv(self, resume_scores: List[ResumeScore]) -> str:
        """Export resume scores to CSV format"""
        data = []
        
        for i, score in enumerate(resume_scores, 1):
            row = {
                'Rank': i,
                'Candidate Name': score.resume_data.candidate_name,
                'Filename': score.resume_data.filename,
                'Overall Score': round(score.overall_score, 2),
                'Skills Score': round(score.skill_score, 2),
                'Experience Score': round(score.experience_score, 2),
                'Education Score': round(score.education_score, 2),
                'Culture Fit Score': round(score.culture_fit_score, 2),
                'Career Progression Score': round(score.career_progression_score, 2),
                'Email': score.resume_data.contact.email or '',
                'Phone': score.resume_data.contact.phone or '',
                'Location': score.resume_data.contact.location or '',
                'LinkedIn': score.resume_data.contact.linkedin or '',
                'Technical Skills': ', '.join(score.resume_data.skills.technical),
                'Years of Experience': self._calculate_total_experience(score.resume_data),
                'Education': self._format_education(score.resume_data.education),
                'Top Achievements': self._get_top_achievements(score.resume_data),
                'Recommendations': '; '.join(score.recommendations),
                'Red Flags': '; '.join(score.red_flags),
                'Analysis Date': score.created_at.strftime('%Y-%m-%d %H:%M:%S')
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def export_to_excel(self, resume_scores: List[ResumeScore]) -> BytesIO:
        """Export resume scores to Excel with multiple sheets"""
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_data = []
            for i, score in enumerate(resume_scores, 1):
                summary_data.append({
                    'Rank': i,
                    'Candidate': score.resume_data.candidate_name,
                    'Overall Score': round(score.overall_score, 2),
                    'Skills': round(score.skill_score, 2),
                    'Experience': round(score.experience_score, 2),
                    'Education': round(score.education_score, 2),
                    'Culture Fit': round(score.culture_fit_score, 2),
                    'Email': score.resume_data.contact.email or '',
                    'Phone': score.resume_data.contact.phone or ''
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed analysis sheet
            detailed_data = []
            for score in resume_scores:
                detailed_data.append({
                    'Candidate': score.resume_data.candidate_name,
                    'Filename': score.resume_data.filename,
                    'Summary': score.resume_data.summary,
                    'Technical Skills': ', '.join(score.resume_data.skills.technical),
                    'Soft Skills': ', '.join(score.resume_data.skills.soft),
                    'Frameworks': ', '.join(score.resume_data.skills.frameworks),
                    'Tools': ', '.join(score.resume_data.skills.tools),
                    'Certifications': ', '.join(score.resume_data.skills.certifications),
                    'Recommendations': '; '.join(score.recommendations),
                    'Red Flags': '; '.join(score.red_flags)
                })
            
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='Detailed Analysis', index=False)
            
            # Contact information sheet
            contact_data = []
            for score in resume_scores:
                contact_data.append({
                    'Candidate': score.resume_data.candidate_name,
                    'Email': score.resume_data.contact.email or '',
                    'Phone': score.resume_data.contact.phone or '',
                    'LinkedIn': score.resume_data.contact.linkedin or '',
                    'GitHub': score.resume_data.contact.github or '',
                    'Website': score.resume_data.contact.website or '',
                    'Location': score.resume_data.contact.location or ''
                })
            
            contact_df = pd.DataFrame(contact_data)
            contact_df.to_excel(writer, sheet_name='Contact Info', index=False)
        
        output.seek(0)
        return output
    
    def export_to_json(self, resume_scores: List[ResumeScore]) -> str:
        """Export resume scores to JSON format"""
        data = {
            'export_date': datetime.now().isoformat(),
            'total_candidates': len(resume_scores),
            'candidates': [score.to_dict() for score in resume_scores]
        }
        
        return json.dumps(data, indent=2, default=str)
    
    def generate_hiring_report(self, resume_scores: List[ResumeScore], 
                             job_requirements: JobRequirements) -> Dict[str, Any]:
        """Generate comprehensive hiring report"""
        
        if not resume_scores:
            return {'error': 'No candidates to analyze'}
        
        # Calculate statistics
        scores = [score.overall_score for score in resume_scores]
        avg_score = sum(scores) / len(scores)
        
        # Categorize candidates
        excellent = [s for s in resume_scores if s.overall_score >= 90]
        very_good = [s for s in resume_scores if 80 <= s.overall_score < 90]
        good = [s for s in resume_scores if 70 <= s.overall_score < 80]
        average = [s for s in resume_scores if 60 <= s.overall_score < 70]
        below_average = [s for s in resume_scores if s.overall_score < 60]
        
        # Skill gap analysis
        all_required_skills = set(job_requirements.required_skills)
        candidate_skills = set()
        for score in resume_scores:
            candidate_skills.update(score.resume_data.skills.technical)
            candidate_skills.update(score.resume_data.skills.frameworks)
            candidate_skills.update(score.resume_data.skills.tools)
        
        skill_coverage = len(all_required_skills.intersection(candidate_skills)) / len(all_required_skills) * 100
        
        return {
            'job_info': {
                'title': job_requirements.title,
                'total_candidates': len(resume_scores),
                'analysis_date': datetime.now().isoformat()
            },
            'score_statistics': {
                'average_score': round(avg_score, 2),
                'highest_score': max(scores),
                'lowest_score': min(scores),
                'score_range': max(scores) - min(scores)
            },
            'candidate_distribution': {
                'excellent': len(excellent),
                'very_good': len(very_good),
                'good': len(good),
                'average': len(average),
                'below_average': len(below_average)
            },
            'top_candidates': [
                {
                    'name': score.resume_data.candidate_name,
                    'score': score.overall_score,
                    'email': score.resume_data.contact.email,
                    'key_strengths': score.recommendations[:3]
                }
                for score in resume_scores[:10]  # Top 10
            ],
            'skill_analysis': {
                'skill_coverage_percentage': round(skill_coverage, 2),
                'most_common_skills': self._get_most_common_skills(resume_scores),
                'skill_gaps': list(all_required_skills - candidate_skills)
            },
            'recommendations': {
                'immediate_interviews': len(excellent),
                'second_round_consideration': len(very_good),
                'skill_development_candidates': len(good),
                'not_recommended': len(below_average)
            }
        }
    
    def _calculate_total_experience(self, resume_data) -> float:
        """Calculate total years of experience"""
        total_months = sum(exp.duration_months or 0 for exp in resume_data.experience)
        return round(total_months / 12, 1) if total_months > 0 else 0
    
    def _format_education(self, education_list) -> str:
        """Format education information"""
        if not education_list:
            return ''
        
        edu_strings = []
        for edu in education_list:
            edu_str = f"{edu.degree} - {edu.institution}"
            if edu.graduation_year:
                edu_str += f" ({edu.graduation_year})"
            edu_strings.append(edu_str)
        
        return '; '.join(edu_strings)
    
    def _get_top_achievements(self, resume_data) -> str:
        """Get top achievements across all experiences"""
        all_achievements = []
        for exp in resume_data.experience:
            all_achievements.extend(exp.achievements)
        
        # Return top 3 achievements (by length as a proxy for detail)
        top_achievements = sorted(all_achievements, key=len, reverse=True)[:3]
        return '; '.join(top_achievements)
    
    def _get_most_common_skills(self, resume_scores: List[ResumeScore]) -> List[Dict[str, Any]]:
        """Get most common skills across all candidates"""
        skill_counts = {}
        
        for score in resume_scores:
            all_skills = (
                score.resume_data.skills.technical +
                score.resume_data.skills.frameworks +
                score.resume_data.skills.tools
            )
            
            for skill in all_skills:
                skill_lower = skill.lower().strip()
                skill_counts[skill_lower] = skill_counts.get(skill_lower, 0) + 1
        
        # Sort by frequency and return top 10
        sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'skill': skill, 'count': count} for skill, count in sorted_skills[:10]]