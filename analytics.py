import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BusinessIntelligenceDashboard:
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'light': '#bcbd22',
            'dark': '#17becf'
        }
    
    def create_score_distribution_chart(self, score_data: Dict[str, int]) -> go.Figure:
        """Create score distribution chart"""
        
        ranges = list(score_data.keys())
        counts = list(score_data.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=ranges,
                y=counts,
                marker_color=[self.colors['primary'], self.colors['secondary'], 
                             self.colors['success'], self.colors['warning'], self.colors['info']],
                text=counts,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Candidate Score Distribution",
            xaxis_title="Score Range",
            yaxis_title="Number of Candidates",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def create_pipeline_chart(self, pipeline_data: List[Dict[str, Any]]) -> go.Figure:
        """Create candidate pipeline chart"""
        
        stages = [stage['stage'] for stage in pipeline_data]
        counts = [stage['count'] for stage in pipeline_data]
        avg_scores = [stage['avg_score'] for stage in pipeline_data]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Candidates by Stage', 'Average Score by Stage'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Pipeline counts
        fig.add_trace(
            go.Bar(
                x=stages,
                y=counts,
                name="Candidates",
                marker_color=self.colors['primary']
            ),
            row=1, col=1
        )
        
        # Average scores
        fig.add_trace(
            go.Scatter(
                x=stages,
                y=avg_scores,
                mode='lines+markers',
                name="Avg Score",
                line=dict(color=self.colors['secondary'], width=3),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Candidate Pipeline Analysis",
            template="plotly_white",
            height=400,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Pipeline Stage", row=1, col=1)
        fig.update_xaxes(title_text="Pipeline Stage", row=1, col=2)
        fig.update_yaxes(title_text="Number of Candidates", row=1, col=1)
        fig.update_yaxes(title_text="Average Score", row=1, col=2)
        
        return fig
    
    def create_skills_heatmap(self, skills_data: List[tuple]) -> go.Figure:
        """Create skills frequency heatmap"""
        
        if not skills_data:
            return self._create_empty_chart("No Skills Data Available")
        
        skills, counts = zip(*skills_data[:10])  # Top 10 skills
        
        fig = go.Figure(data=go.Heatmap(
            z=[counts],
            x=skills,
            y=['Frequency'],
            colorscale='Blues',
            showscale=True,
            text=counts,
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title="Top Skills Frequency",
            xaxis_title="Skills",
            yaxis_title="",
            template="plotly_white",
            height=300
        )
        
        return fig
    
    def create_conversion_funnel(self, conversion_data: Dict[str, float]) -> go.Figure:
        """Create conversion funnel chart"""
        
        stages = list(conversion_data.keys())
        rates = list(conversion_data.values())
        
        fig = go.Figure(go.Funnel(
            y=stages,
            x=rates,
            textinfo="value+percent initial",
            marker=dict(
                color=[self.colors['primary'], self.colors['secondary'], self.colors['success']],
                line=dict(width=2, color="white")
            )
        ))
        
        fig.update_layout(
            title="Candidate Conversion Funnel",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def create_score_trend_chart(self, trend_data: List[Dict[str, Any]]) -> go.Figure:
        """Create score trend over time chart"""
        
        if not trend_data:
            return self._create_empty_chart("No Trend Data Available")
        
        dates = [item['date'] for item in trend_data]
        scores = [item['score'] for item in trend_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=scores,
            mode='lines+markers',
            name='Average Score',
            line=dict(color=self.colors['primary'], width=3),
            marker=dict(size=6)
        ))
        
        # Add trend line
        if len(scores) > 1:
            z = np.polyfit(range(len(scores)), scores, 1)
            p = np.poly1d(z)
            trend_line = p(range(len(scores)))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(color=self.colors['secondary'], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title="Score Trend Over Time",
            xaxis_title="Date",
            yaxis_title="Average Score",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def create_component_scores_radar(self, component_scores: Dict[str, float]) -> go.Figure:
        """Create radar chart for component scores"""
        
        categories = list(component_scores.keys())
        values = list(component_scores.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Average Scores',
            line_color=self.colors['primary']
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Component Score Analysis",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def create_job_performance_chart(self, job_data: Dict[str, Any]) -> go.Figure:
        """Create job performance analysis chart"""
        
        if job_data.get('no_data'):
            return self._create_empty_chart("No Analysis Data for This Job")
        
        score_dist = job_data.get('score_distribution', {})
        ranges = list(score_dist.keys())
        counts = list(score_dist.values())
        
        fig = go.Figure(data=[
            go.Pie(
                labels=ranges,
                values=counts,
                hole=0.3,
                marker_colors=[self.colors['primary'], self.colors['secondary'], 
                              self.colors['success'], self.colors['warning'], self.colors['info']]
            )
        ])
        
        fig.update_layout(
            title=f"Score Distribution for {job_data.get('job', {}).get('title', 'Job')}",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def create_top_candidates_table(self, candidates_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create top candidates table"""
        
        if not candidates_data:
            return pd.DataFrame(columns=['Name', 'Email', 'Score', 'Skills', 'Experience', 'Education'])
        
        df_data = []
        for candidate in candidates_data:
            df_data.append({
                'Name': candidate.get('name', 'N/A'),
                'Email': candidate.get('email', 'N/A'),
                'Score': f"{candidate.get('overall_score', 0):.1f}",
                'Skills': f"{candidate.get('skills_match_score', 0):.1f}",
                'Experience': f"{candidate.get('experience_score', 0):.1f}",
                'Education': f"{candidate.get('education_score', 0):.1f}",
                'Status': candidate.get('status', 'N/A')
            })
        
        return pd.DataFrame(df_data)
    
    def create_bias_analysis_chart(self, bias_data: Dict[str, Any]) -> go.Figure:
        """Create bias analysis chart"""
        
        if not bias_data:
            return self._create_empty_chart("No Bias Analysis Data")
        
        bias_risks = bias_data.get('bias_risks', [])
        risk_scores = bias_data.get('risk_scores', {})
        
        if not risk_scores:
            return self._create_empty_chart("No Bias Risk Scores")
        
        categories = list(risk_scores.keys())
        scores = list(risk_scores.values())
        
        # Color based on risk level
        colors = []
        for score in scores:
            if score < 30:
                colors.append(self.colors['success'])
            elif score < 60:
                colors.append(self.colors['warning'])
            else:
                colors.append(self.colors['info'])
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=scores,
                marker_color=colors,
                text=scores,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Bias Risk Analysis",
            xaxis_title="Bias Category",
            yaxis_title="Risk Score",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def create_geographic_distribution_chart(self, location_data: List[Dict[str, Any]]) -> go.Figure:
        """Create geographic distribution chart"""
        
        if not location_data:
            return self._create_empty_chart("No Location Data Available")
        
        locations = [item['location'] for item in location_data]
        counts = [item['count'] for item in location_data]
        
        fig = go.Figure(data=[
            go.Bar(
                x=locations,
                y=counts,
                marker_color=self.colors['primary'],
                text=counts,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Candidate Geographic Distribution",
            xaxis_title="Location",
            yaxis_title="Number of Candidates",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def create_education_level_chart(self, education_data: Dict[str, int]) -> go.Figure:
        """Create education level distribution chart"""
        
        levels = list(education_data.keys())
        counts = list(education_data.values())
        
        fig = go.Figure(data=[
            go.Pie(
                labels=levels,
                values=counts,
                hole=0.3,
                marker_colors=px.colors.qualitative.Set3
            )
        ])
        
        fig.update_layout(
            title="Education Level Distribution",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def create_experience_distribution_chart(self, experience_data: Dict[str, int]) -> go.Figure:
        """Create experience level distribution chart"""
        
        levels = list(experience_data.keys())
        counts = list(experience_data.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=levels,
                y=counts,
                marker_color=self.colors['secondary'],
                text=counts,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Experience Level Distribution",
            xaxis_title="Experience Level",
            yaxis_title="Number of Candidates",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def create_dashboard_summary(self, analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create dashboard summary metrics"""
        
        return {
            "total_candidates": analytics_data.get("total_candidates", 0),
            "new_candidates": analytics_data.get("new_candidates", 0),
            "total_jobs": analytics_data.get("total_jobs", 0),
            "average_overall_score": analytics_data.get("average_scores", {}).get("overall", 0),
            "top_skill": analytics_data.get("top_skills", [{}])[0].get("word", "N/A") if analytics_data.get("top_skills") else "N/A",
            "score_distribution": analytics_data.get("score_distribution", {}),
            "status_distribution": analytics_data.get("status_distribution", {})
        }
    
    def create_performance_metrics(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance metrics"""
        
        conversion_rates = pipeline_data.get("conversion_rates", {})
        
        return {
            "screening_rate": f"{conversion_rates.get('screening_rate', 0):.1f}%",
            "interview_rate": f"{conversion_rates.get('interview_rate', 0):.1f}%",
            "hire_rate": f"{conversion_rates.get('hire_rate', 0):.1f}%",
            "time_to_hire": "15 days",  # This would be calculated from actual data
            "cost_per_hire": "$2,500",  # This would be calculated from actual data
            "quality_of_hire": "4.2/5"  # This would be calculated from actual data
        }
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font_size=16
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template="plotly_white",
            height=400
        )
        return fig
    
    def generate_insights(self, analytics_data: Dict[str, Any]) -> List[str]:
        """Generate business insights from analytics data"""
        
        insights = []
        
        # Score insights
        avg_score = analytics_data.get("average_scores", {}).get("overall", 0)
        if avg_score > 80:
            insights.append("High-quality candidate pool with average score above 80")
        elif avg_score < 60:
            insights.append("Candidate pool needs improvement - average score below 60")
        
        # Skills insights
        top_skills = analytics_data.get("top_skills", [])
        if top_skills:
            top_skill = top_skills[0]
            insights.append(f"Most common skill: {top_skill.get('word', 'N/A')} ({top_skill.get('count', 0)} candidates)")
        
        # Pipeline insights
        status_dist = analytics_data.get("status_distribution", {})
        new_candidates = status_dist.get("new", 0)
        screened_candidates = status_dist.get("screened", 0)
        
        if new_candidates > screened_candidates * 2:
            insights.append("High volume of new candidates - consider increasing screening capacity")
        
        # Score distribution insights
        score_dist = analytics_data.get("score_distribution", {})
        high_scores = score_dist.get("90-100", 0) + score_dist.get("80-89", 0)
        total_candidates = sum(score_dist.values())
        
        if total_candidates > 0:
            high_score_percentage = (high_scores / total_candidates) * 100
            if high_score_percentage > 30:
                insights.append(f"Strong candidate quality: {high_score_percentage:.1f}% score 80+")
            elif high_score_percentage < 10:
                insights.append(f"Low candidate quality: only {high_score_percentage:.1f}% score 80+")
        
        return insights