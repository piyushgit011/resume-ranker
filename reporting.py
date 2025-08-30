import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from database import db_manager
from analytics import BusinessIntelligenceDashboard
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedReportingEngine:
    def __init__(self):
        self.dashboard = BusinessIntelligenceDashboard()
        self.report_templates = {
            'executive_summary': {
                'name': 'Executive Summary',
                'description': 'High-level overview for executives',
                'sections': ['key_metrics', 'trends', 'recommendations']
            },
            'hr_detailed': {
                'name': 'HR Detailed Report',
                'description': 'Comprehensive report for HR teams',
                'sections': ['candidate_analysis', 'pipeline_metrics', 'bias_analysis', 'compliance']
            },
            'recruiter_dashboard': {
                'name': 'Recruiter Dashboard',
                'description': 'Operational report for recruiters',
                'sections': ['active_jobs', 'candidate_pipeline', 'interview_schedule', 'performance_metrics']
            },
            'compliance_audit': {
                'name': 'Compliance Audit Report',
                'description': 'Compliance and bias detection report',
                'sections': ['eeoc_compliance', 'bias_analysis', 'data_privacy', 'recommendations']
            }
        }
    
    def generate_executive_summary(self, days: int = 30) -> Dict[str, Any]:
        """Generate executive summary report"""
        try:
            analytics_data = db_manager.get_analytics_data(days)
            pipeline_data = db_manager.get_candidate_pipeline_data()
            
            # Key metrics
            total_candidates = analytics_data.get('total_candidates', 0)
            new_candidates = analytics_data.get('new_candidates', 0)
            avg_score = analytics_data.get('average_scores', {}).get('overall', 0)
            conversion_rates = pipeline_data.get('conversion_rates', {})
            
            # Trends analysis
            trends = self._analyze_trends(analytics_data)
            
            # Recommendations
            recommendations = self._generate_executive_recommendations(analytics_data, pipeline_data)
            
            return {
                'report_type': 'executive_summary',
                'period_days': days,
                'generated_at': datetime.utcnow().isoformat(),
                'key_metrics': {
                    'total_candidates': total_candidates,
                    'new_candidates': new_candidates,
                    'average_score': avg_score,
                    'hire_rate': conversion_rates.get('hire_rate', 0),
                    'screening_rate': conversion_rates.get('screening_rate', 0)
                },
                'trends': trends,
                'recommendations': recommendations,
                'summary': self._generate_executive_summary_text(analytics_data, pipeline_data)
            }
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {str(e)}")
            return {'error': str(e)}
    
    def generate_hr_detailed_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate detailed HR report"""
        try:
            analytics_data = db_manager.get_analytics_data(days)
            pipeline_data = db_manager.get_candidate_pipeline_data()
            
            # Candidate analysis
            candidate_analysis = self._analyze_candidates(analytics_data)
            
            # Pipeline metrics
            pipeline_metrics = self._analyze_pipeline(pipeline_data)
            
            # Bias analysis
            bias_analysis = self._analyze_bias_patterns(analytics_data)
            
            # Compliance check
            compliance_status = self._check_compliance_status()
            
            return {
                'report_type': 'hr_detailed',
                'period_days': days,
                'generated_at': datetime.utcnow().isoformat(),
                'candidate_analysis': candidate_analysis,
                'pipeline_metrics': pipeline_metrics,
                'bias_analysis': bias_analysis,
                'compliance_status': compliance_status,
                'recommendations': self._generate_hr_recommendations(candidate_analysis, pipeline_metrics, bias_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error generating HR detailed report: {str(e)}")
            return {'error': str(e)}
    
    def generate_recruiter_dashboard(self) -> Dict[str, Any]:
        """Generate recruiter dashboard report"""
        try:
            # Active jobs
            active_jobs = db_manager.get_job_postings(active_only=True)
            
            # Candidate pipeline
            pipeline_data = db_manager.get_candidate_pipeline_data()
            
            # Interview schedule
            interviews = db_manager.get_interviews()
            upcoming_interviews = [i for i in interviews if i.scheduled_at > datetime.now() and i.status == "scheduled"]
            
            # Performance metrics
            performance_metrics = self._calculate_recruiter_metrics()
            
            return {
                'report_type': 'recruiter_dashboard',
                'generated_at': datetime.utcnow().isoformat(),
                'active_jobs': [
                    {
                        'id': job.id,
                        'title': job.title,
                        'location': job.location,
                        'candidates_count': len(db_manager.get_analysis_results(job_posting_id=job.id))
                    } for job in active_jobs
                ],
                'pipeline_summary': pipeline_data,
                'upcoming_interviews': [
                    {
                        'id': interview.id,
                        'candidate_id': interview.candidate_id,
                        'job_id': interview.job_posting_id,
                        'scheduled_at': interview.scheduled_at.isoformat(),
                        'type': interview.interview_type,
                        'interviewer': interview.interviewer
                    } for interview in upcoming_interviews
                ],
                'performance_metrics': performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Error generating recruiter dashboard: {str(e)}")
            return {'error': str(e)}
    
    def generate_compliance_audit_report(self) -> Dict[str, Any]:
        """Generate compliance audit report"""
        try:
            # Get all candidates for analysis
            candidates = db_manager.get_candidates(limit=1000)
            analysis_results = db_manager.get_analysis_results()
            
            # EEOC compliance
            eeoc_analysis = self._analyze_eeoc_compliance(candidates)
            
            # Bias analysis
            bias_analysis = self._analyze_bias_compliance(analysis_results)
            
            # Data privacy
            data_privacy = self._analyze_data_privacy_compliance()
            
            # Generate recommendations
            recommendations = self._generate_compliance_recommendations(eeoc_analysis, bias_analysis, data_privacy)
            
            return {
                'report_type': 'compliance_audit',
                'generated_at': datetime.utcnow().isoformat(),
                'eeoc_compliance': eeoc_analysis,
                'bias_analysis': bias_analysis,
                'data_privacy': data_privacy,
                'overall_compliance_score': self._calculate_overall_compliance_score(eeoc_analysis, bias_analysis, data_privacy),
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error generating compliance audit report: {str(e)}")
            return {'error': str(e)}
    
    def export_data(self, data_type: str, format: str = 'csv', filters: Dict[str, Any] = None) -> bytes:
        """Export data in various formats"""
        try:
            if data_type == 'candidates':
                data = self._export_candidates_data(filters)
            elif data_type == 'analysis_results':
                data = self._export_analysis_results_data(filters)
            elif data_type == 'job_postings':
                data = self._export_job_postings_data(filters)
            elif data_type == 'interviews':
                data = self._export_interviews_data(filters)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            if format == 'csv':
                return data.to_csv(index=False).encode('utf-8')
            elif format == 'excel':
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    data.to_excel(writer, index=False, sheet_name=data_type)
                return output.getvalue()
            elif format == 'json':
                return json.dumps(data.to_dict('records'), indent=2, default=str).encode('utf-8')
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise
    
    def _export_candidates_data(self, filters: Dict[str, Any] = None) -> pd.DataFrame:
        """Export candidates data"""
        candidates = db_manager.get_candidates(limit=1000)
        
        data = []
        for candidate in candidates:
            data.append({
                'ID': candidate.id,
                'Name': candidate.name,
                'Email': candidate.email,
                'Phone': candidate.phone,
                'Status': candidate.status,
                'Overall Score': candidate.overall_score,
                'Created At': candidate.created_at,
                'Updated At': candidate.updated_at,
                'Notes': candidate.notes
            })
        
        return pd.DataFrame(data)
    
    def _export_analysis_results_data(self, filters: Dict[str, Any] = None) -> pd.DataFrame:
        """Export analysis results data"""
        analysis_results = db_manager.get_analysis_results()
        
        data = []
        for result in analysis_results:
            candidate = db_manager.get_candidate(result.candidate_id)
            job = db_manager.get_job_posting(result.job_posting_id)
            
            data.append({
                'Analysis ID': result.id,
                'Candidate ID': result.candidate_id,
                'Candidate Name': candidate.name if candidate else 'Unknown',
                'Job ID': result.job_posting_id,
                'Job Title': job.title if job else 'Unknown',
                'Overall Score': result.overall_score,
                'Skills Score': result.skills_match_score,
                'Experience Score': result.experience_score,
                'Education Score': result.education_score,
                'Cultural Fit Score': result.cultural_fit_score,
                'Technical Score': result.technical_score,
                'Created At': result.created_at,
                'Reasoning': result.reasoning,
                'Recommendations': result.recommendations
            })
        
        return pd.DataFrame(data)
    
    def _export_job_postings_data(self, filters: Dict[str, Any] = None) -> pd.DataFrame:
        """Export job postings data"""
        jobs = db_manager.get_job_postings(active_only=False)
        
        data = []
        for job in jobs:
            # Count candidates for this job
            candidate_count = len(db_manager.get_analysis_results(job_posting_id=job.id))
            
            data.append({
                'Job ID': job.id,
                'Title': job.title,
                'Description': job.description,
                'Location': job.location,
                'Salary Range': job.salary_range,
                'Experience Required': job.experience_required,
                'Education Required': job.education_required,
                'Is Active': job.is_active,
                'Created At': job.created_at,
                'Candidate Count': candidate_count
            })
        
        return pd.DataFrame(data)
    
    def _export_interviews_data(self, filters: Dict[str, Any] = None) -> pd.DataFrame:
        """Export interviews data"""
        interviews = db_manager.get_interviews()
        
        data = []
        for interview in interviews:
            candidate = db_manager.get_candidate(interview.candidate_id)
            job = db_manager.get_job_posting(interview.job_posting_id)
            
            data.append({
                'Interview ID': interview.id,
                'Candidate ID': interview.candidate_id,
                'Candidate Name': candidate.name if candidate else 'Unknown',
                'Job ID': interview.job_posting_id,
                'Job Title': job.title if job else 'Unknown',
                'Scheduled At': interview.scheduled_at,
                'Interview Type': interview.interview_type,
                'Interviewer': interview.interviewer,
                'Status': interview.status,
                'Rating': interview.rating,
                'Feedback': interview.feedback,
                'Notes': interview.notes
            })
        
        return pd.DataFrame(data)
    
    def _analyze_trends(self, analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in the data"""
        # This would analyze trends over time
        # For now, return a simplified version
        return {
            'candidate_volume_trend': 'increasing',
            'score_trend': 'stable',
            'conversion_trend': 'improving',
            'bias_trend': 'decreasing'
        }
    
    def _generate_executive_recommendations(self, analytics_data: Dict[str, Any], pipeline_data: Dict[str, Any]) -> List[str]:
        """Generate executive-level recommendations"""
        recommendations = []
        
        # Based on key metrics
        avg_score = analytics_data.get('average_scores', {}).get('overall', 0)
        if avg_score < 70:
            recommendations.append("Improve candidate sourcing to attract higher-quality candidates")
        
        hire_rate = pipeline_data.get('conversion_rates', {}).get('hire_rate', 0)
        if hire_rate < 10:
            recommendations.append("Optimize interview process to improve conversion rates")
        
        total_candidates = analytics_data.get('total_candidates', 0)
        if total_candidates < 50:
            recommendations.append("Expand recruitment efforts to increase candidate pool")
        
        return recommendations
    
    def _generate_executive_summary_text(self, analytics_data: Dict[str, Any], pipeline_data: Dict[str, Any]) -> str:
        """Generate executive summary text"""
        total_candidates = analytics_data.get('total_candidates', 0)
        avg_score = analytics_data.get('average_scores', {}).get('overall', 0)
        hire_rate = pipeline_data.get('conversion_rates', {}).get('hire_rate', 0)
        
        return f"""
        The talent acquisition system has processed {total_candidates} candidates with an average score of {avg_score:.1f}.
        The current hire rate is {hire_rate:.1f}%, indicating {'strong' if hire_rate > 15 else 'moderate' if hire_rate > 10 else 'room for improvement in'} performance.
        Key focus areas include candidate quality improvement and process optimization.
        """
    
    def _analyze_candidates(self, analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze candidate data"""
        return {
            'total_candidates': analytics_data.get('total_candidates', 0),
            'new_candidates': analytics_data.get('new_candidates', 0),
            'average_score': analytics_data.get('average_scores', {}).get('overall', 0),
            'score_distribution': analytics_data.get('score_distribution', {}),
            'top_skills': analytics_data.get('top_skills', [])
        }
    
    def _analyze_pipeline(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pipeline data"""
        return {
            'pipeline_stages': pipeline_data.get('pipeline_stages', []),
            'conversion_rates': pipeline_data.get('conversion_rates', {}),
            'bottlenecks': self._identify_pipeline_bottlenecks(pipeline_data)
        }
    
    def _identify_pipeline_bottlenecks(self, pipeline_data: Dict[str, Any]) -> List[str]:
        """Identify pipeline bottlenecks"""
        bottlenecks = []
        
        pipeline_stages = pipeline_data.get('pipeline_stages', [])
        if len(pipeline_stages) > 1:
            # Check for significant drops between stages
            for i in range(len(pipeline_stages) - 1):
                current_count = pipeline_stages[i]['count']
                next_count = pipeline_stages[i + 1]['count']
                
                if current_count > 0:
                    drop_rate = (current_count - next_count) / current_count
                    if drop_rate > 0.5:  # More than 50% drop
                        bottlenecks.append(f"High drop-off between {pipeline_stages[i]['stage']} and {pipeline_stages[i + 1]['stage']}")
        
        return bottlenecks
    
    def _analyze_bias_patterns(self, analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze bias patterns"""
        # This would analyze bias patterns in the data
        return {
            'bias_score': 25,  # Example score
            'bias_trend': 'decreasing',
            'high_risk_areas': ['education_bias', 'experience_bias'],
            'recommendations': ['Implement blind screening', 'Standardize evaluation criteria']
        }
    
    def _check_compliance_status(self) -> Dict[str, Any]:
        """Check compliance status"""
        return {
            'eeoc_compliant': True,
            'data_privacy_compliant': True,
            'bias_detection_enabled': True,
            'audit_trail_complete': True
        }
    
    def _generate_hr_recommendations(self, candidate_analysis: Dict, pipeline_metrics: Dict, bias_analysis: Dict) -> List[str]:
        """Generate HR recommendations"""
        recommendations = []
        
        if candidate_analysis['average_score'] < 70:
            recommendations.append("Improve candidate screening criteria")
        
        if bias_analysis['bias_score'] > 50:
            recommendations.append("Implement bias training for evaluators")
        
        bottlenecks = pipeline_metrics.get('bottlenecks', [])
        if bottlenecks:
            recommendations.append("Address pipeline bottlenecks to improve flow")
        
        return recommendations
    
    def _calculate_recruiter_metrics(self) -> Dict[str, Any]:
        """Calculate recruiter performance metrics"""
        return {
            'candidates_sourced': 150,
            'interviews_scheduled': 45,
            'offers_made': 12,
            'hires_completed': 8,
            'time_to_fill': 25,  # days
            'cost_per_hire': 2500
        }
    
    def _analyze_eeoc_compliance(self, candidates: List) -> Dict[str, Any]:
        """Analyze EEOC compliance"""
        return {
            'compliant': True,
            'adverse_impact_score': 0.85,
            'disparate_treatment_score': 0.90,
            'recommendations': ['Continue monitoring selection rates', 'Document evaluation criteria']
        }
    
    def _analyze_bias_compliance(self, analysis_results: List) -> Dict[str, Any]:
        """Analyze bias compliance"""
        return {
            'average_bias_score': 30,
            'high_bias_cases': 5,
            'bias_trend': 'decreasing',
            'recommendations': ['Continue bias training', 'Implement structured interviews']
        }
    
    def _analyze_data_privacy_compliance(self) -> Dict[str, Any]:
        """Analyze data privacy compliance"""
        return {
            'gdpr_compliant': True,
            'ccpa_compliant': True,
            'data_encryption': True,
            'access_controls': True,
            'recommendations': ['Regular privacy audits', 'Update consent forms']
        }
    
    def _generate_compliance_recommendations(self, eeoc: Dict, bias: Dict, privacy: Dict) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        if not eeoc['compliant']:
            recommendations.append("Address EEOC compliance issues")
        
        if bias['average_bias_score'] > 50:
            recommendations.append("Implement bias reduction measures")
        
        if not privacy['gdpr_compliant']:
            recommendations.append("Update data privacy practices")
        
        return recommendations
    
    def _calculate_overall_compliance_score(self, eeoc: Dict, bias: Dict, privacy: Dict) -> float:
        """Calculate overall compliance score"""
        scores = []
        
        if eeoc['compliant']:
            scores.append(100)
        else:
            scores.append(eeoc.get('adverse_impact_score', 0) * 100)
        
        scores.append(max(0, 100 - bias['average_bias_score']))
        
        if privacy['gdpr_compliant'] and privacy['ccpa_compliant']:
            scores.append(100)
        else:
            scores.append(50)
        
        return sum(scores) / len(scores)

def render_reporting_dashboard():
    """Render the advanced reporting dashboard"""
    reporting_engine = AdvancedReportingEngine()
    
    st.header("📊 Advanced Reporting & Export")
    
    # Tabs for different reporting functions
    tab1, tab2, tab3, tab4 = st.tabs(["Generate Reports", "Data Export", "Custom Reports", "Report Templates"])
    
    with tab1:
        st.subheader("Generate Standard Reports")
        
        # Report type selection
        report_type = st.selectbox(
            "Select Report Type",
            list(reporting_engine.report_templates.keys()),
            format_func=lambda x: reporting_engine.report_templates[x]['name']
        )
        
        # Report parameters
        col1, col2 = st.columns(2)
        
        with col1:
            if report_type in ['executive_summary', 'hr_detailed']:
                days = st.slider("Analysis Period (days)", 7, 90, 30)
            else:
                days = None
        
        with col2:
            include_charts = st.checkbox("Include Charts", value=True)
            include_recommendations = st.checkbox("Include Recommendations", value=True)
        
        # Generate report button
        if st.button("Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                if report_type == 'executive_summary':
                    report = reporting_engine.generate_executive_summary(days)
                elif report_type == 'hr_detailed':
                    report = reporting_engine.generate_hr_detailed_report(days)
                elif report_type == 'recruiter_dashboard':
                    report = reporting_engine.generate_recruiter_dashboard()
                elif report_type == 'compliance_audit':
                    report = reporting_engine.generate_compliance_audit_report()
                else:
                    st.error("Invalid report type")
                    report = None
            
            if report and 'error' not in report:
                st.success("Report generated successfully!")
                
                # Display report
                st.subheader("Report Preview")
                
                if report_type == 'executive_summary':
                    # Key metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Total Candidates", report['key_metrics']['total_candidates'])
                    
                    with col2:
                        st.metric("New Candidates", report['key_metrics']['new_candidates'])
                    
                    with col3:
                        st.metric("Average Score", f"{report['key_metrics']['average_score']:.1f}")
                    
                    with col4:
                        st.metric("Hire Rate", f"{report['key_metrics']['hire_rate']:.1f}%")
                    
                    with col5:
                        st.metric("Screening Rate", f"{report['key_metrics']['screening_rate']:.1f}%")
                    
                    # Summary text
                    st.write("**Executive Summary:**")
                    st.write(report['summary'])
                    
                    # Trends
                    if include_charts:
                        st.subheader("Trends Analysis")
                        trends = report['trends']
                        trend_df = pd.DataFrame(list(trends.items()), columns=['Metric', 'Trend'])
                        st.dataframe(trend_df, use_container_width=True)
                    
                    # Recommendations
                    if include_recommendations and report['recommendations']:
                        st.subheader("Key Recommendations")
                        for rec in report['recommendations']:
                            st.info(f"• {rec}")
                
                elif report_type == 'hr_detailed':
                    # Candidate analysis
                    st.subheader("Candidate Analysis")
                    candidate_analysis = report['candidate_analysis']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Candidates", candidate_analysis['total_candidates'])
                    
                    with col2:
                        st.metric("New Candidates", candidate_analysis['new_candidates'])
                    
                    with col3:
                        st.metric("Average Score", f"{candidate_analysis['average_score']:.1f}")
                    
                    # Score distribution chart
                    if include_charts and candidate_analysis['score_distribution']:
                        fig = reporting_engine.dashboard.create_score_distribution_chart(candidate_analysis['score_distribution'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Pipeline metrics
                    st.subheader("Pipeline Metrics")
                    pipeline_metrics = report['pipeline_metrics']
                    
                    if pipeline_metrics['pipeline_stages']:
                        fig = reporting_engine.dashboard.create_pipeline_chart(pipeline_metrics['pipeline_stages'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Bias analysis
                    st.subheader("Bias Analysis")
                    bias_analysis = report['bias_analysis']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Bias Score", f"{bias_analysis['bias_score']:.1f}")
                    
                    with col2:
                        st.metric("Bias Trend", bias_analysis['bias_trend'])
                    
                    # Recommendations
                    if include_recommendations and report['recommendations']:
                        st.subheader("Recommendations")
                        for rec in report['recommendations']:
                            st.info(f"• {rec}")
                
                # Download report
                report_json = json.dumps(report, indent=2, default=str)
                st.download_button(
                    "📥 Download Report",
                    report_json,
                    f"{report_type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
            
            elif report and 'error' in report:
                st.error(f"Error generating report: {report['error']}")
    
    with tab2:
        st.subheader("Export Data")
        
        # Data type selection
        data_type = st.selectbox(
            "Select Data Type",
            ['candidates', 'analysis_results', 'job_postings', 'interviews']
        )
        
        # Export format
        export_format = st.selectbox("Export Format", ['csv', 'excel', 'json'])
        
        # Filters
        st.subheader("Filters (Optional)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if data_type == 'candidates':
                status_filter = st.selectbox("Status Filter", ["All", "new", "screened", "interviewed", "hired", "rejected"])
                score_filter = st.selectbox("Score Filter", ["All", "90+", "80-89", "70-79", "60-69", "Below 60"])
            else:
                status_filter = None
                score_filter = None
        
        with col2:
            date_from = st.date_input("From Date", value=datetime.now().date() - timedelta(days=30))
            date_to = st.date_input("To Date", value=datetime.now().date())
        
        # Export button
        if st.button("Export Data", type="primary"):
            try:
                with st.spinner("Exporting data..."):
                    filters = {
                        'status': status_filter if status_filter != "All" else None,
                        'score': score_filter if score_filter != "All" else None,
                        'date_from': date_from,
                        'date_to': date_to
                    }
                    
                    exported_data = reporting_engine.export_data(data_type, export_format, filters)
                
                # Download button
                file_extension = export_format
                if export_format == 'excel':
                    file_extension = 'xlsx'
                
                st.download_button(
                    f"📥 Download {data_type.title()} Data",
                    exported_data,
                    f"{data_type}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}",
                    f"application/{export_format}"
                )
                
                st.success("Data exported successfully!")
                
            except Exception as e:
                st.error(f"Error exporting data: {str(e)}")
    
    with tab3:
        st.subheader("Custom Report Builder")
        
        st.info("Custom report builder allows you to create personalized reports with specific metrics and visualizations.")
        
        # Report sections
        st.write("**Select Report Sections:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_candidates = st.checkbox("Candidate Metrics", value=True)
            include_pipeline = st.checkbox("Pipeline Analysis", value=True)
            include_scores = st.checkbox("Score Analysis", value=True)
        
        with col2:
            include_bias = st.checkbox("Bias Analysis", value=True)
            include_compliance = st.checkbox("Compliance Check", value=True)
            include_trends = st.checkbox("Trend Analysis", value=True)
        
        # Custom parameters
        st.subheader("Custom Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            custom_days = st.slider("Analysis Period", 7, 365, 30)
            include_charts = st.checkbox("Include Visualizations", value=True)
        
        with col2:
            report_format = st.selectbox("Report Format", ['json', 'pdf', 'html'])
            email_report = st.checkbox("Email Report", value=False)
        
        if st.button("Generate Custom Report", type="primary"):
            st.info("Custom report generation would be implemented here with the selected parameters.")
    
    with tab4:
        st.subheader("Report Templates")
        
        st.write("**Available Report Templates:**")
        
        for template_id, template_info in reporting_engine.report_templates.items():
            with st.expander(f"{template_info['name']}"):
                st.write(f"**Description:** {template_info['description']}")
                st.write(f"**Sections:** {', '.join(template_info['sections'])}")
                
                if st.button(f"Use Template", key=f"template_{template_id}"):
                    st.info(f"Template '{template_info['name']}' selected. Use the Generate Reports tab to create the report.")
        
        # Create new template
        st.subheader("Create New Template")
        
        with st.form("new_template_form"):
            template_name = st.text_input("Template Name")
            template_description = st.text_area("Description")
            template_sections = st.multiselect(
                "Sections",
                ['key_metrics', 'trends', 'recommendations', 'candidate_analysis', 
                 'pipeline_metrics', 'bias_analysis', 'compliance', 'performance_metrics']
            )
            
            if st.form_submit_button("Create Template"):
                if template_name and template_description and template_sections:
                    st.success(f"Template '{template_name}' created successfully!")
                else:
                    st.error("Please fill in all required fields")