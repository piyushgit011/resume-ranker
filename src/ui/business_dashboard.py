import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta

from ..services.analytics_service import AnalyticsService
from ..services.database_service import DatabaseService

class BusinessDashboard:
    """Business intelligence dashboard for HR analytics"""
    
    def __init__(self, analytics_service: AnalyticsService, db_service: DatabaseService):
        self.analytics_service = analytics_service
        self.db_service = db_service
    
    def show_executive_summary(self):
        """Show executive summary dashboard"""
        st.title("📊 Executive Dashboard")
        
        # Time period selector
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### Key Performance Indicators")
        
        with col2:
            period = st.selectbox("Time Period", [30, 60, 90, 180], index=1)
        
        # Get metrics
        funnel_metrics = self.analytics_service.get_hiring_funnel_metrics(period)
        roi_analysis = self.analytics_service.get_roi_analysis(period)
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_analyzed = funnel_metrics['funnel_metrics'].get('total_analyzed', 0)
            st.metric("Candidates Analyzed", total_analyzed)
        
        with col2:
            qualified = funnel_metrics['funnel_metrics'].get('qualified', 0)
            qualification_rate = (qualified / total_analyzed * 100) if total_analyzed > 0 else 0
            st.metric("Qualification Rate", f"{qualification_rate:.1f}%")
        
        with col3:
            avg_score = funnel_metrics['funnel_metrics'].get('avg_score', 0)
            st.metric("Average Score", f"{avg_score:.1f}" if avg_score else "N/A")
        
        with col4:
            time_saved = roi_analysis['efficiency_metrics'].get('time_saved_hours', 0)
            st.metric("Time Saved", f"{time_saved:.0f} hours")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Hiring funnel
            if funnel_metrics['funnel_metrics']:
                self._show_hiring_funnel_chart(funnel_metrics['funnel_metrics'])
        
        with col2:
            # ROI metrics
            if roi_analysis['efficiency_metrics']:
                self._show_roi_chart(roi_analysis['efficiency_metrics'])
        
        # Skills analysis
        st.markdown("### 🛠️ Skills Market Analysis")
        skills_analysis = self.analytics_service.get_skills_demand_analysis(period)
        
        if skills_analysis['top_skills_by_supply']:
            col1, col2 = st.columns(2)
            
            with col1:
                self._show_skills_supply_chart(skills_analysis['top_skills_by_supply'])
            
            with col2:
                self._show_skills_demand_chart(skills_analysis['top_skills_by_demand'])
        
        # Insights and recommendations
        st.markdown("### 💡 Business Insights")
        
        if skills_analysis['skill_insights']:
            for insight in skills_analysis['skill_insights']:
                st.info(f"📈 {insight}")
        
        # Performance trends
        st.markdown("### 📈 Performance Trends")
        trends = self.analytics_service.get_performance_trends(period)
        
        if trends['weekly_trends']:
            self._show_performance_trends_chart(trends['weekly_trends'])
    
    def show_hiring_analytics(self):
        """Show detailed hiring analytics"""
        st.title("🎯 Hiring Analytics")
        
        # Hiring funnel analysis
        st.subheader("📊 Hiring Funnel Analysis")
        
        period = st.slider("Analysis Period (days)", 30, 365, 90)
        funnel_data = self.analytics_service.get_hiring_funnel_metrics(period)
        
        if funnel_data['funnel_metrics']:
            # Create funnel visualization
            metrics = funnel_data['funnel_metrics']
            
            funnel_stages = [
                ("Total Analyzed", metrics.get('total_analyzed', 0)),
                ("Qualified (70+)", metrics.get('qualified', 0)),
                ("Very Good (80+)", metrics.get('very_good', 0)),
                ("Excellent (90+)", metrics.get('excellent', 0))
            ]
            
            # Funnel chart
            fig_funnel = go.Figure(go.Funnel(
                y=[stage[0] for stage in funnel_stages],
                x=[stage[1] for stage in funnel_stages],
                textinfo="value+percent initial"
            ))
            
            fig_funnel.update_layout(title="Candidate Qualification Funnel")
            st.plotly_chart(fig_funnel, use_container_width=True)
            
            # Conversion rates
            if funnel_data['conversion_rates']:
                st.subheader("📈 Conversion Rates")
                
                col1, col2, col3 = st.columns(3)
                
                rates = funnel_data['conversion_rates']
                with col1:
                    st.metric("Excellent Rate", f"{rates.get('excellent_rate', 0):.1f}%")
                with col2:
                    st.metric("Qualified Rate", f"{rates.get('qualified_rate', 0):.1f}%")
                with col3:
                    st.metric("Recommendation Rate", f"{rates.get('recommendation_rate', 0):.1f}%")
        
        # Feedback analysis
        if funnel_data['feedback_breakdown']:
            st.subheader("📋 Hiring Outcomes")
            
            feedback_df = pd.DataFrame(funnel_data['feedback_breakdown'])
            
            # Pie chart of outcomes
            fig_pie = px.pie(
                feedback_df, 
                values='count', 
                names='feedback_type',
                title="Hiring Outcomes Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Outcome quality analysis
            st.markdown("#### Outcome Quality Analysis")
            st.dataframe(feedback_df, use_container_width=True)
    
    def show_recruiter_dashboard(self):
        """Show dashboard for recruiters"""
        st.title("👥 Recruiter Dashboard")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 New Analysis Session"):
                st.session_state.page = "🔍 Resume Analysis"
                st.rerun()
        
        with col2:
            if st.button("📊 View Analytics"):
                st.session_state.page = "📈 Analytics"
                st.rerun()
        
        with col3:
            if st.button("📤 Export Results"):
                st.session_state.show_export = True
        
        # Recent activity
        st.subheader("🕒 Recent Activity")
        
        recent_analyses = self.db_service.get_resume_analyses(limit=10)
        
        if recent_analyses:
            activity_data = []
            for analysis in recent_analyses:
                activity_data.append({
                    'Date': analysis['created_at'][:10],
                    'Candidate': analysis['candidate_name'],
                    'Score': f"{analysis['overall_score']:.1f}",
                    'Status': self._get_status_from_score(analysis['overall_score'])
                })
            
            activity_df = pd.DataFrame(activity_data)
            st.dataframe(activity_df, use_container_width=True)
        else:
            st.info("No recent activity found")
        
        # Performance summary
        st.subheader("📈 This Month's Performance")
        
        monthly_metrics = self.analytics_service.get_hiring_funnel_metrics(30)
        
        if monthly_metrics['funnel_metrics']:
            metrics = monthly_metrics['funnel_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Candidates Reviewed", metrics.get('total_analyzed', 0))
            with col2:
                qualified_rate = (metrics.get('qualified', 0) / metrics.get('total_analyzed', 1)) * 100
                st.metric("Qualification Rate", f"{qualified_rate:.1f}%")
            with col3:
                st.metric("High Performers", metrics.get('excellent', 0))
            with col4:
                avg_score = metrics.get('avg_score', 0)
                st.metric("Avg Score", f"{avg_score:.1f}" if avg_score else "N/A")
    
    def _show_hiring_funnel_chart(self, metrics: Dict[str, Any]):
        """Show hiring funnel chart"""
        funnel_data = [
            ("Total Candidates", metrics.get('total_analyzed', 0)),
            ("Qualified (70+)", metrics.get('qualified', 0)),
            ("Very Good (80+)", metrics.get('very_good', 0)),
            ("Excellent (90+)", metrics.get('excellent', 0))
        ]
        
        fig = go.Figure(go.Funnel(
            y=[item[0] for item in funnel_data],
            x=[item[1] for item in funnel_data],
            textinfo="value+percent initial"
        ))
        
        fig.update_layout(title="Hiring Funnel", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_roi_chart(self, roi_metrics: Dict[str, Any]):
        """Show ROI metrics chart"""
        metrics_data = [
            ("Time Saved (Hours)", roi_metrics.get('time_saved_hours', 0)),
            ("Cost Savings ($)", roi_metrics.get('cost_savings_estimate', 0)),
            ("Hire Rate (%)", roi_metrics.get('hire_rate_percentage', 0)),
            ("Interview Rate (%)", roi_metrics.get('interview_rate_percentage', 0))
        ]
        
        fig = go.Figure(data=[
            go.Bar(
                x=[item[0] for item in metrics_data],
                y=[item[1] for item in metrics_data],
                text=[f"{item[1]:.1f}" for item in metrics_data],
                textposition='auto'
            )
        ])
        
        fig.update_layout(title="ROI Metrics", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_skills_supply_chart(self, skills_data: List[tuple]):
        """Show skills supply chart"""
        if not skills_data:
            return
        
        skills, counts = zip(*skills_data[:10])
        
        fig = px.bar(
            x=list(counts),
            y=list(skills),
            orientation='h',
            title="Top Skills in Candidate Pool",
            labels={'x': 'Number of Candidates', 'y': 'Skills'}
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_skills_demand_chart(self, demand_data: List[tuple]):
        """Show skills demand chart"""
        if not demand_data:
            st.info("No high-performer skills data available")
            return
        
        skills, counts = zip(*demand_data[:10])
        
        fig = px.bar(
            x=list(counts),
            y=list(skills),
            orientation='h',
            title="Skills Among High Performers",
            labels={'x': 'High Performer Count', 'y': 'Skills'},
            color=list(counts),
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_performance_trends_chart(self, trends_data: List[Dict[str, Any]]):
        """Show performance trends over time"""
        if not trends_data:
            return
        
        df = pd.DataFrame(trends_data)
        df['week_start'] = pd.to_datetime(df['week_start'])
        
        fig = go.Figure()
        
        # Average score trend
        fig.add_trace(go.Scatter(
            x=df['week_start'],
            y=df['avg_score'],
            mode='lines+markers',
            name='Average Score',
            line=dict(color='blue', width=3)
        ))
        
        # Qualified candidates trend
        fig.add_trace(go.Scatter(
            x=df['week_start'],
            y=df['qualified'],
            mode='lines+markers',
            name='Qualified Candidates',
            yaxis='y2',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title="Performance Trends Over Time",
            xaxis_title="Week",
            yaxis=dict(title="Average Score", side="left"),
            yaxis2=dict(title="Qualified Candidates", side="right", overlaying="y"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _get_status_from_score(self, score: float) -> str:
        """Get status label from score"""
        if score >= 90:
            return "🟢 Excellent"
        elif score >= 80:
            return "🟡 Very Good"
        elif score >= 70:
            return "🟠 Good"
        elif score >= 60:
            return "🔵 Average"
        else:
            return "🔴 Below Average"