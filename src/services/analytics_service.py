import sqlite3
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

class AnalyticsService:
    """Advanced analytics service for business insights"""
    
    def __init__(self, db_path: str = "/workspace/data/resume_ranker.db"):
        self.db_path = db_path
    
    def get_hiring_funnel_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get hiring funnel metrics for the last N days"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get candidates in different stages
            query = """
                SELECT 
                    COUNT(*) as total_analyzed,
                    COUNT(CASE WHEN overall_score >= 90 THEN 1 END) as excellent,
                    COUNT(CASE WHEN overall_score >= 80 THEN 1 END) as very_good,
                    COUNT(CASE WHEN overall_score >= 70 THEN 1 END) as qualified,
                    COUNT(CASE WHEN overall_score < 60 THEN 1 END) as not_recommended,
                    AVG(overall_score) as avg_score
                FROM resume_analyses 
                WHERE created_at >= datetime('now', '-{} days')
            """.format(days)
            
            metrics = conn.execute(query).fetchone()
            
            # Get feedback data
            feedback_query = """
                SELECT 
                    f.feedback_type,
                    COUNT(*) as count,
                    AVG(r.overall_score) as avg_score_for_feedback
                FROM candidate_feedback f
                JOIN resume_analyses r ON f.resume_analysis_id = r.id
                WHERE r.created_at >= datetime('now', '-{} days')
                GROUP BY f.feedback_type
            """.format(days)
            
            feedback_data = conn.execute(feedback_query).fetchall()
            
            return {
                'period_days': days,
                'total_analyzed': metrics['total_analyzed'] if metrics else 0,
                'funnel_metrics': dict(metrics) if metrics else {},
                'feedback_breakdown': [dict(row) for row in feedback_data],
                'conversion_rates': self._calculate_conversion_rates(dict(metrics) if metrics else {})
            }
    
    def get_skills_demand_analysis(self, days: int = 90) -> Dict[str, Any]:
        """Analyze skills demand and supply in candidate pool"""
        with sqlite3.connect(self.db_path) as conn:
            # Get all resume analyses from the period
            query = """
                SELECT resume_data, overall_score
                FROM resume_analyses 
                WHERE created_at >= datetime('now', '-{} days')
            """.format(days)
            
            results = conn.execute(query).fetchall()
            
            skill_demand = {}
            skill_supply = {}
            high_performers = []  # Candidates with score >= 80
            
            for row in results:
                try:
                    resume_data = json.loads(row[0])
                    score = row[1]
                    
                    # Extract all skills
                    skills = resume_data.get('skills', {})
                    all_skills = (
                        skills.get('technical', []) +
                        skills.get('frameworks', []) +
                        skills.get('tools', []) +
                        skills.get('languages', [])
                    )
                    
                    # Count skill supply
                    for skill in all_skills:
                        skill_lower = skill.lower().strip()
                        skill_supply[skill_lower] = skill_supply.get(skill_lower, 0) + 1
                        
                        # Track high performer skills
                        if score >= 80:
                            if skill_lower not in skill_demand:
                                skill_demand[skill_lower] = 0
                            skill_demand[skill_lower] += 1
                
                except json.JSONDecodeError:
                    continue
            
            # Calculate skill scarcity (demand vs supply)
            skill_scarcity = {}
            for skill in skill_supply:
                demand = skill_demand.get(skill, 0)
                supply = skill_supply[skill]
                # Higher score means more scarce (high demand, low supply)
                scarcity_score = (demand / supply) * 100 if supply > 0 else 0
                skill_scarcity[skill] = scarcity_score
            
            # Sort skills by various metrics
            top_skills_by_supply = sorted(skill_supply.items(), key=lambda x: x[1], reverse=True)[:20]
            top_skills_by_demand = sorted(skill_demand.items(), key=lambda x: x[1], reverse=True)[:20]
            most_scarce_skills = sorted(skill_scarcity.items(), key=lambda x: x[1], reverse=True)[:20]
            
            return {
                'analysis_period_days': days,
                'total_candidates_analyzed': len(results),
                'top_skills_by_supply': top_skills_by_supply,
                'top_skills_by_demand': top_skills_by_demand,
                'most_scarce_skills': most_scarce_skills,
                'skill_insights': self._generate_skill_insights(skill_supply, skill_demand)
            }
    
    def get_performance_trends(self, days: int = 180) -> Dict[str, Any]:
        """Get performance trends over time"""
        with sqlite3.connect(self.db_path) as conn:
            # Weekly aggregated data
            query = """
                SELECT 
                    date(created_at, 'weekday 0', '-6 days') as week_start,
                    COUNT(*) as candidates_analyzed,
                    AVG(overall_score) as avg_score,
                    COUNT(CASE WHEN overall_score >= 80 THEN 1 END) as high_performers,
                    COUNT(CASE WHEN overall_score >= 70 THEN 1 END) as qualified
                FROM resume_analyses 
                WHERE created_at >= datetime('now', '-{} days')
                GROUP BY week_start
                ORDER BY week_start
            """.format(days)
            
            trends = conn.execute(query).fetchall()
            
            return {
                'period_days': days,
                'weekly_trends': [dict(row) for row in trends],
                'trend_analysis': self._analyze_trends([dict(row) for row in trends])
            }
    
    def get_comparative_analysis(self, job_posting_ids: List[int]) -> Dict[str, Any]:
        """Compare performance across different job postings"""
        if not job_posting_ids:
            return {'error': 'No job postings provided'}
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            comparative_data = {}
            
            for job_id in job_posting_ids:
                # Get job posting details
                job_query = "SELECT title, requirements FROM job_postings WHERE id = ?"
                job_info = conn.execute(job_query, (job_id,)).fetchone()
                
                # Get analysis metrics for this job
                metrics_query = """
                    SELECT 
                        COUNT(*) as total_candidates,
                        AVG(overall_score) as avg_score,
                        COUNT(CASE WHEN overall_score >= 80 THEN 1 END) as high_performers,
                        MAX(overall_score) as max_score,
                        MIN(overall_score) as min_score
                    FROM resume_analyses 
                    WHERE job_posting_id = ?
                """
                metrics = conn.execute(metrics_query, (job_id,)).fetchone()
                
                if job_info and metrics:
                    comparative_data[job_id] = {
                        'job_title': job_info['title'],
                        'metrics': dict(metrics)
                    }
            
            return {
                'job_comparisons': comparative_data,
                'insights': self._generate_comparative_insights(comparative_data)
            }
    
    def get_roi_analysis(self, days: int = 90) -> Dict[str, Any]:
        """Calculate ROI metrics for the resume ranking system"""
        with sqlite3.connect(self.db_path) as conn:
            # Get hiring outcomes
            query = """
                SELECT 
                    f.feedback_type,
                    COUNT(*) as count,
                    AVG(r.overall_score) as avg_score,
                    COUNT(CASE WHEN r.overall_score >= 80 THEN 1 END) as high_score_count
                FROM candidate_feedback f
                JOIN resume_analyses r ON f.resume_analysis_id = r.id
                WHERE r.created_at >= datetime('now', '-{} days')
                GROUP BY f.feedback_type
            """.format(days)
            
            outcomes = conn.execute(query).fetchall()
            
            # Calculate efficiency metrics
            total_analyzed = sum(row[1] for row in outcomes)
            hired_count = next((row[1] for row in outcomes if row[0] == 'hired'), 0)
            interviewed_count = next((row[1] for row in outcomes if row[0] == 'interviewed'), 0)
            
            hire_rate = (hired_count / total_analyzed * 100) if total_analyzed > 0 else 0
            interview_rate = (interviewed_count / total_analyzed * 100) if total_analyzed > 0 else 0
            
            return {
                'period_days': days,
                'total_candidates_analyzed': total_analyzed,
                'hiring_outcomes': [dict(zip(['feedback_type', 'count', 'avg_score', 'high_score_count'], row)) for row in outcomes],
                'efficiency_metrics': {
                    'hire_rate_percentage': hire_rate,
                    'interview_rate_percentage': interview_rate,
                    'time_saved_hours': total_analyzed * 0.5,  # Estimated 30 minutes saved per resume
                    'cost_savings_estimate': total_analyzed * 25  # Estimated $25 saved per resume in manual review time
                }
            }
    
    def _calculate_conversion_rates(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate conversion rates through hiring funnel"""
        total = metrics.get('total_analyzed', 0)
        
        if total == 0:
            return {}
        
        return {
            'excellent_rate': (metrics.get('excellent', 0) / total) * 100,
            'qualified_rate': (metrics.get('qualified', 0) / total) * 100,
            'recommendation_rate': ((total - metrics.get('not_recommended', 0)) / total) * 100
        }
    
    def _analyze_trends(self, weekly_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in the data"""
        if len(weekly_data) < 2:
            return {'message': 'Insufficient data for trend analysis'}
        
        # Calculate trend direction
        recent_avg = sum(week['avg_score'] for week in weekly_data[-4:]) / min(4, len(weekly_data))
        earlier_avg = sum(week['avg_score'] for week in weekly_data[:4]) / min(4, len(weekly_data))
        
        trend_direction = "improving" if recent_avg > earlier_avg else "declining" if recent_avg < earlier_avg else "stable"
        
        return {
            'trend_direction': trend_direction,
            'recent_average_score': recent_avg,
            'earlier_average_score': earlier_avg,
            'score_change': recent_avg - earlier_avg,
            'total_weeks_analyzed': len(weekly_data)
        }
    
    def _generate_skill_insights(self, skill_supply: Dict[str, int], 
                               skill_demand: Dict[str, int]) -> List[str]:
        """Generate insights about skill supply and demand"""
        insights = []
        
        # Most common skills
        top_skills = sorted(skill_supply.items(), key=lambda x: x[1], reverse=True)[:5]
        insights.append(f"Most common skills in candidate pool: {', '.join([skill for skill, _ in top_skills])}")
        
        # High demand skills
        if skill_demand:
            top_demand = sorted(skill_demand.items(), key=lambda x: x[1], reverse=True)[:3]
            insights.append(f"Skills most associated with high performers: {', '.join([skill for skill, _ in top_demand])}")
        
        # Skills with low supply
        low_supply = [skill for skill, count in skill_supply.items() if count <= 2]
        if low_supply:
            insights.append(f"Rare skills in candidate pool (potential recruiting opportunities): {', '.join(low_supply[:5])}")
        
        return insights
    
    def _generate_comparative_insights(self, comparative_data: Dict[int, Dict[str, Any]]) -> List[str]:
        """Generate insights from comparative analysis"""
        insights = []
        
        if len(comparative_data) < 2:
            return ["Need at least 2 job postings for comparison"]
        
        # Find job with highest average score
        best_job = max(comparative_data.items(), 
                      key=lambda x: x[1]['metrics']['avg_score'])
        
        insights.append(f"'{best_job[1]['job_title']}' has the highest average candidate score ({best_job[1]['metrics']['avg_score']:.1f})")
        
        # Find job with most candidates
        most_candidates = max(comparative_data.items(), 
                            key=lambda x: x[1]['metrics']['total_candidates'])
        
        insights.append(f"'{most_candidates[1]['job_title']}' attracted the most candidates ({most_candidates[1]['metrics']['total_candidates']})")
        
        return insights