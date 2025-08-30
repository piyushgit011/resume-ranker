import streamlit as st
import os
import tempfile
import shutil
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px

# Add src to path for imports
import sys
sys.path.append('/workspace/src')

from agents.ranking_orchestrator import RankingOrchestrator
from services.pdf_service import PDFService
from services.database_service import DatabaseService
from services.export_service import ExportService
from services.auth_service import AuthService, show_login_form, require_auth
from models.resume import ScoringCriteria, JobRequirements, ResumeScore
from utils.helpers import (
    create_temp_directory, cleanup_temp_directory, format_score,
    create_score_visualization, create_score_distribution_chart,
    create_skills_comparison_chart, display_candidate_card,
    validate_api_key, calculate_hiring_priority
)
from config.settings import settings, JOB_TEMPLATES
import plotly.graph_objects as go

class AdvancedResumeRankerApp:
    """Advanced Resume Ranker Application with multiple AI agents"""
    
    def __init__(self):
        self.pdf_service = PDFService()
        self.db_service = DatabaseService()
        self.export_service = ExportService()
        self.auth_service = AuthService()
        self.orchestrator = None
        
        # Initialize session state
        if 'current_job_requirements' not in st.session_state:
            st.session_state.current_job_requirements = None
        if 'current_scores' not in st.session_state:
            st.session_state.current_scores = []
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user' not in st.session_state:
            st.session_state.user = None
    
    def run(self):
        """Run the main application"""
        st.set_page_config(
            page_title=settings.page_title,
            page_icon=settings.page_icon,
            layout=settings.layout
        )
        
        # Check authentication
        if not st.session_state.authenticated:
            self._show_login_page()
            return
        
        # Sidebar for navigation
        st.sidebar.title("🎯 Resume Ranker Pro")
        
        # User info in sidebar
        if st.session_state.user:
            st.sidebar.markdown(f"**Welcome, {st.session_state.user['full_name']}**")
            st.sidebar.markdown(f"Role: {st.session_state.user['role'].title()}")
            
            if st.sidebar.button("🚪 Logout"):
                self._logout()
                return
        
        page = st.sidebar.selectbox(
            "Navigation",
            ["📊 Dashboard", "🔍 Resume Analysis", "📈 Analytics", "👥 User Management", "⚙️ Settings", "📚 Help"]
        )
        
        # API Key validation
        if not self._validate_setup():
            return
        
        # Initialize orchestrator
        self.orchestrator = RankingOrchestrator(settings.openai_api_key, settings.openai_model)
        
        # Route to appropriate page
        if page == "📊 Dashboard":
            self._show_dashboard()
        elif page == "🔍 Resume Analysis":
            self._show_resume_analysis()
        elif page == "📈 Analytics":
            self._show_analytics()
        elif page == "👥 User Management":
            self._show_user_management()
        elif page == "⚙️ Settings":
            self._show_settings()
        elif page == "📚 Help":
            self._show_help()
    
    def _show_login_page(self):
        """Show login page"""
        st.title("🔐 Resume Ranker Pro - Login")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            show_login_form(self.auth_service)
    
    def _logout(self):
        """Logout current user"""
        if 'session_token' in st.session_state:
            self.auth_service.logout_user(st.session_state.session_token)
        
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        st.rerun()
    
    def _show_user_management(self):
        """Show user management page"""
        from services.auth_service import show_user_management
        show_user_management(self.auth_service)
    
    def _validate_setup(self) -> bool:
        """Validate application setup"""
        validation_errors = settings.validate()
        
        if validation_errors:
            st.error("⚠️ Configuration Issues Detected")
            
            for field, error in validation_errors.items():
                st.error(f"**{field}**: {error}")
            
            if 'openai_api_key' in validation_errors:
                st.info("💡 Please set your OpenAI API key as an environment variable:")
                st.code("export OPENAI_API_KEY='your-api-key-here'")
            
            return False
        
        return True
    
    def _show_dashboard(self):
        """Show main dashboard"""
        st.title("📊 Resume Ranker Dashboard")
        
        # Get recent analytics
        analytics = self.db_service.get_analytics_data()
        
        if analytics['statistics']:
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Candidates", analytics['statistics'].get('total_candidates', 0))
            with col2:
                avg_score = analytics['statistics'].get('avg_score', 0)
                st.metric("Average Score", f"{avg_score:.1f}" if avg_score else "N/A")
            with col3:
                max_score = analytics['statistics'].get('max_score', 0)
                st.metric("Highest Score", f"{max_score:.1f}" if max_score else "N/A")
            with col4:
                min_score = analytics['statistics'].get('min_score', 0)
                st.metric("Lowest Score", f"{min_score:.1f}" if min_score else "N/A")
            
            # Score distribution chart
            if analytics['score_distribution']:
                st.subheader("📈 Score Distribution")
                
                score_data = pd.DataFrame(analytics['score_distribution'])
                fig = px.bar(
                    score_data, 
                    x='score_range', 
                    y='count',
                    title="Candidate Score Distribution",
                    labels={'score_range': 'Score Range', 'count': 'Number of Candidates'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📝 No analysis data available yet. Start by analyzing some resumes!")
        
        # Recent sessions
        st.subheader("🕒 Recent Analysis Sessions")
        recent_sessions = self.db_service.get_ranking_sessions(limit=10)
        
        if recent_sessions:
            sessions_df = pd.DataFrame(recent_sessions)
            st.dataframe(sessions_df, use_container_width=True)
        else:
            st.info("No recent sessions found.")
    
    def _show_resume_analysis(self):
        """Show resume analysis page"""
        st.title("🔍 Resume Analysis")
        
        # Job requirements setup
        st.subheader("1️⃣ Job Requirements")
        
        # Quick templates
        col1, col2 = st.columns([2, 1])
        
        with col1:
            template_choice = st.selectbox(
                "Use a job template (optional)",
                ["Custom"] + list(JOB_TEMPLATES.keys())
            )
        
        with col2:
            if st.button("Load Template"):
                if template_choice != "Custom":
                    template = JOB_TEMPLATES[template_choice]
                    st.session_state.template_data = template
                    st.rerun()
        
        # Job requirements form
        with st.form("job_requirements_form"):
            job_title = st.text_input("Job Title", value="Software Engineer")
            job_description = st.text_area("Job Description", height=100)
            
            col1, col2 = st.columns(2)
            
            with col1:
                required_skills = st.text_area(
                    "Required Skills (one per line)",
                    value="\n".join(getattr(st.session_state, 'template_data', {}).get('required_skills', [])),
                    height=150
                )
                
                min_experience = st.number_input(
                    "Minimum Years of Experience",
                    min_value=0,
                    max_value=20,
                    value=getattr(st.session_state, 'template_data', {}).get('min_experience', 2)
                )
            
            with col2:
                preferred_skills = st.text_area(
                    "Preferred Skills (one per line)",
                    value="\n".join(getattr(st.session_state, 'template_data', {}).get('preferred_skills', [])),
                    height=150
                )
                
                education_req = st.text_input(
                    "Education Requirements",
                    value=getattr(st.session_state, 'template_data', {}).get('education_requirements', "Bachelor's degree")
                )
            
            # Advanced criteria
            with st.expander("🔧 Advanced Scoring Criteria"):
                st.markdown("**Scoring Weights** (must sum to 1.0)")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    skills_weight = st.slider("Skills Match", 0.0, 1.0, 0.30, 0.05)
                    experience_weight = st.slider("Experience", 0.0, 1.0, 0.25, 0.05)
                
                with col2:
                    education_weight = st.slider("Education", 0.0, 1.0, 0.15, 0.05)
                    culture_weight = st.slider("Culture Fit", 0.0, 1.0, 0.15, 0.05)
                
                with col3:
                    career_weight = st.slider("Career Progression", 0.0, 1.0, 0.15, 0.05)
                
                total_weight = skills_weight + experience_weight + education_weight + culture_weight + career_weight
                
                if abs(total_weight - 1.0) > 0.01:
                    st.error(f"⚠️ Weights must sum to 1.0 (current: {total_weight:.2f})")
                else:
                    st.success(f"✅ Weights sum to {total_weight:.2f}")
            
            industry = st.text_input("Industry", value="Technology")
            company_culture = st.text_area(
                "Company Culture Keywords (one per line)",
                value="\n".join(getattr(st.session_state, 'template_data', {}).get('company_culture', [])),
                height=80
            )
            
            submit_job_req = st.form_submit_button("💾 Save Job Requirements")
            
            if submit_job_req:
                # Create job requirements object
                job_requirements = JobRequirements(
                    title=job_title,
                    description=job_description,
                    required_skills=[s.strip() for s in required_skills.split('\n') if s.strip()],
                    preferred_skills=[s.strip() for s in preferred_skills.split('\n') if s.strip()],
                    min_experience=min_experience,
                    education_requirements=education_req,
                    industry=industry,
                    location="",  # Can be added later
                    company_culture=[c.strip() for c in company_culture.split('\n') if c.strip()]
                )
                
                # Create scoring criteria
                scoring_criteria = ScoringCriteria(
                    required_skills=job_requirements.required_skills,
                    preferred_skills=job_requirements.preferred_skills,
                    min_experience_years=min_experience,
                    required_education=education_req,
                    industry_experience=[industry],
                    weights={
                        'skills_match': skills_weight,
                        'experience_relevance': experience_weight,
                        'education_match': education_weight,
                        'culture_fit': culture_weight,
                        'career_progression': career_weight
                    }
                )
                
                st.session_state.current_job_requirements = job_requirements
                st.session_state.current_scoring_criteria = scoring_criteria
                st.success("✅ Job requirements saved!")
        
        # Resume upload and processing
        st.subheader("2️⃣ Upload Resumes")
        
        uploaded_zip = st.file_uploader(
            "Upload ZIP file containing resumes (PDF format)",
            type=["zip"],
            help="Upload a ZIP file containing PDF resumes. Maximum file size: 50MB"
        )
        
        if uploaded_zip and st.session_state.current_job_requirements:
            st.success(f"📁 Uploaded: {uploaded_zip.name} ({len(uploaded_zip.getbuffer())/1024:.1f} KB)")
            
            if st.button("🚀 Start Analysis", type="primary"):
                self._process_resumes(uploaded_zip)
        
        # Display results if available
        if st.session_state.processing_complete and st.session_state.current_scores:
            self._display_results()
    
    def _process_resumes(self, uploaded_zip):
        """Process uploaded resumes"""
        temp_dir = create_temp_directory()
        
        try:
            with st.spinner("🔄 Processing resumes..."):
                # Extract resume texts
                resume_texts = self.pdf_service.process_zip_file(
                    uploaded_zip.getbuffer(), temp_dir
                )
                
                st.info(f"📄 Found {len(resume_texts)} valid PDF resumes")
                
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process resumes with orchestrator
                resume_scores = self.orchestrator.process_resume_batch(
                    resume_texts,
                    st.session_state.current_scoring_criteria,
                    st.session_state.current_job_requirements
                )
                
                # Save to database
                job_id = self.db_service.save_job_posting(st.session_state.current_job_requirements)
                
                for i, score in enumerate(resume_scores):
                    self.db_service.save_resume_analysis(score, job_id)
                    progress_bar.progress((i + 1) / len(resume_scores))
                    status_text.text(f"Saving results... ({i + 1}/{len(resume_scores)})")
                
                st.session_state.current_scores = resume_scores
                st.session_state.processing_complete = True
                
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis complete!")
                
                st.success(f"🎉 Successfully analyzed {len(resume_scores)} resumes!")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error processing resumes: {str(e)}")
        finally:
            cleanup_temp_directory(temp_dir)
    
    def _display_results(self):
        """Display analysis results"""
        st.subheader("3️⃣ Analysis Results")
        
        scores = st.session_state.current_scores
        
        if not scores:
            st.warning("No results to display")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Candidates", len(scores))
        with col2:
            avg_score = sum(s.overall_score for s in scores) / len(scores)
            st.metric("Average Score", f"{avg_score:.1f}")
        with col3:
            top_score = max(s.overall_score for s in scores)
            st.metric("Top Score", f"{top_score:.1f}")
        with col4:
            qualified = len([s for s in scores if s.overall_score >= 70])
            st.metric("Qualified Candidates", qualified)
        
        # Visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Rankings", "📊 Visualizations", "📋 Detailed View", "📤 Export"])
        
        with tab1:
            self._show_rankings_tab(scores)
        
        with tab2:
            self._show_visualizations_tab(scores)
        
        with tab3:
            self._show_detailed_view_tab(scores)
        
        with tab4:
            self._show_export_tab(scores)
    
    def _show_rankings_tab(self, scores: List[ResumeScore]):
        """Show rankings table"""
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_score = st.slider("Minimum Score", 0, 100, 0)
        with col2:
            max_results = st.slider("Max Results", 5, 100, 20)
        with col3:
            sort_by = st.selectbox("Sort By", ["Overall Score", "Skills Score", "Experience Score"])
        
        # Filter and sort
        filtered_scores = [s for s in scores if s.overall_score >= min_score][:max_results]
        
        if sort_by == "Skills Score":
            filtered_scores.sort(key=lambda x: x.skill_score, reverse=True)
        elif sort_by == "Experience Score":
            filtered_scores.sort(key=lambda x: x.experience_score, reverse=True)
        
        # Display candidate cards
        for i, score in enumerate(filtered_scores, 1):
            with st.container():
                st.markdown("---")
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"### #{i} {score.resume_data.candidate_name}")
                    st.markdown(f"**File:** {score.resume_data.filename}")
                    
                    # Priority indicator
                    priority = calculate_hiring_priority(score)
                    st.markdown(f"**Priority:** {priority}")
                    
                    # Contact info
                    contact = score.resume_data.contact
                    if contact.email:
                        st.markdown(f"📧 {contact.email}")
                    if contact.phone:
                        st.markdown(f"📞 {contact.phone}")
                
                with col2:
                    st.metric("Overall", f"{score.overall_score:.1f}")
                    st.metric("Skills", f"{score.skill_score:.1f}")
                    st.metric("Experience", f"{score.experience_score:.1f}")
                
                with col3:
                    st.metric("Education", f"{score.education_score:.1f}")
                    st.metric("Culture Fit", f"{score.culture_fit_score:.1f}")
                    st.metric("Career", f"{score.career_progression_score:.1f}")
                
                # Expandable details
                with st.expander(f"📋 Detailed Analysis - {score.resume_data.candidate_name}"):
                    self._show_detailed_candidate_analysis(score)
    
    def _show_visualizations_tab(self, scores: List[ResumeScore]):
        """Show data visualizations"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution
            fig_dist = create_score_distribution_chart(scores)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Skills comparison
            fig_skills = create_skills_comparison_chart(scores)
            st.plotly_chart(fig_skills, use_container_width=True)
        
        # Top candidates radar chart
        st.subheader("🎯 Top 5 Candidates Comparison")
        
        top_5 = scores[:5]
        
        if len(top_5) > 0:
            # Create comparison radar chart
            categories = ['Skills', 'Experience', 'Education', 'Culture Fit', 'Career']
            
            fig = go.Figure()
            
            for i, score in enumerate(top_5):
                values = [
                    score.skill_score,
                    score.experience_score,
                    score.education_score,
                    score.culture_fit_score,
                    score.career_progression_score
                ]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=score.resume_data.candidate_name,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                title="Top 5 Candidates Score Comparison",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_detailed_view_tab(self, scores: List[ResumeScore]):
        """Show detailed candidate view"""
        
        if not scores:
            st.warning("No candidates to display")
            return
        
        # Candidate selector
        candidate_names = [f"{i+1}. {score.resume_data.candidate_name}" for i, score in enumerate(scores)]
        selected_candidate = st.selectbox("Select Candidate", candidate_names)
        
        if selected_candidate:
            candidate_index = int(selected_candidate.split('.')[0]) - 1
            selected_score = scores[candidate_index]
            
            self._show_detailed_candidate_analysis(selected_score)
    
    def _show_detailed_candidate_analysis(self, score: ResumeScore):
        """Show detailed analysis for a single candidate"""
        
        # Basic info
        st.markdown(f"## {score.resume_data.candidate_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Scores")
            
            # Create radar chart for individual candidate
            categories = ['Skills', 'Experience', 'Education', 'Culture Fit', 'Career']
            values = [
                score.skill_score,
                score.experience_score,
                score.education_score,
                score.culture_fit_score,
                score.career_progression_score
            ]
            
            fig = create_score_visualization(values, categories)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 👤 Contact Information")
            contact = score.resume_data.contact
            
            if contact.email:
                st.markdown(f"**Email:** {contact.email}")
            if contact.phone:
                st.markdown(f"**Phone:** {contact.phone}")
            if contact.location:
                st.markdown(f"**Location:** {contact.location}")
            if contact.linkedin:
                st.markdown(f"**LinkedIn:** [Profile]({contact.linkedin})")
            if contact.github:
                st.markdown(f"**GitHub:** [Profile]({contact.github})")
        
        # Skills breakdown
        st.markdown("### 🛠️ Skills")
        
        skill_cols = st.columns(3)
        
        with skill_cols[0]:
            st.markdown("**Technical Skills**")
            for skill in score.resume_data.skills.technical:
                st.markdown(f"• {skill}")
        
        with skill_cols[1]:
            st.markdown("**Frameworks**")
            for framework in score.resume_data.skills.frameworks:
                st.markdown(f"• {framework}")
        
        with skill_cols[2]:
            st.markdown("**Tools**")
            for tool in score.resume_data.skills.tools:
                st.markdown(f"• {tool}")
        
        # Experience
        st.markdown("### 💼 Experience")
        for exp in score.resume_data.experience:
            with st.expander(f"{exp.title} at {exp.company}"):
                st.markdown(f"**Duration:** {exp.start_date} - {exp.end_date}")
                st.markdown(f"**Description:** {exp.description}")
                
                if exp.technologies:
                    st.markdown(f"**Technologies:** {', '.join(exp.technologies)}")
                
                if exp.achievements:
                    st.markdown("**Key Achievements:**")
                    for achievement in exp.achievements:
                        st.markdown(f"• {achievement}")
        
        # Recommendations and red flags
        col1, col2 = st.columns(2)
        
        with col1:
            if score.recommendations:
                st.markdown("### ✅ Recommendations")
                for rec in score.recommendations:
                    st.markdown(f"• {rec}")
        
        with col2:
            if score.red_flags:
                st.markdown("### ⚠️ Red Flags")
                for flag in score.red_flags:
                    st.markdown(f"• {flag}")
    
    def _show_export_tab(self, scores: List[ResumeScore]):
        """Show export options"""
        
        st.markdown("### 📤 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download CSV"):
                csv_data = self.export_service.export_to_csv(scores)
                st.download_button(
                    "Download CSV File",
                    csv_data,
                    f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
        
        with col2:
            if st.button("📊 Download Excel"):
                excel_data = self.export_service.export_to_excel(scores)
                st.download_button(
                    "Download Excel File",
                    excel_data.getvalue(),
                    f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col3:
            if st.button("📋 Download JSON"):
                json_data = self.export_service.export_to_json(scores)
                st.download_button(
                    "Download JSON File",
                    json_data,
                    f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
        
        # Hiring report
        st.markdown("### 📈 Hiring Report")
        if st.button("Generate Hiring Report"):
            report = self.export_service.generate_hiring_report(
                scores, st.session_state.current_job_requirements
            )
            
            st.json(report)
    
    def _show_analytics(self):
        """Show analytics page"""
        st.title("📈 Analytics & Insights")
        
        analytics = self.db_service.get_analytics_data()
        
        if not analytics['statistics']:
            st.info("📊 No analytics data available yet. Analyze some resumes first!")
            return
        
        # Key metrics
        st.subheader("📊 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        stats = analytics['statistics']
        with col1:
            st.metric("Total Candidates Analyzed", stats.get('total_candidates', 0))
        with col2:
            avg_score = stats.get('avg_score', 0)
            st.metric("Average Score", f"{avg_score:.1f}" if avg_score else "N/A")
        with col3:
            max_score = stats.get('max_score', 0)
            st.metric("Highest Score", f"{max_score:.1f}" if max_score else "N/A")
        with col4:
            qualified = len([s for s in st.session_state.current_scores if s.overall_score >= 70])
            st.metric("Qualified Candidates", qualified)
        
        # Charts
        if analytics['score_distribution']:
            st.subheader("📈 Score Distribution")
            
            score_data = pd.DataFrame(analytics['score_distribution'])
            fig = px.bar(
                score_data, 
                x='score_range', 
                y='count',
                title="Candidate Score Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _show_settings(self):
        """Show settings page"""
        st.title("⚙️ Settings")
        
        st.subheader("🔧 Configuration")
        
        with st.form("settings_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**AI Model Settings**")
                model_choice = st.selectbox(
                    "OpenAI Model",
                    ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
                    index=0
                )
                
                temperature = st.slider(
                    "AI Temperature",
                    0.0, 1.0, settings.ai_temperature, 0.1
                )
            
            with col2:
                st.markdown("**Processing Settings**")
                max_file_size = st.number_input(
                    "Max File Size (MB)",
                    1, 100, settings.max_file_size_mb
                )
                
                max_batch_size = st.number_input(
                    "Max Resumes per Batch",
                    1, 500, settings.max_resumes_per_batch
                )
            
            if st.form_submit_button("💾 Save Settings"):
                st.success("Settings saved! (Note: Restart app to apply changes)")
    
    def _show_help(self):
        """Show help page"""
        st.title("📚 Help & Documentation")
        
        st.markdown("""
        ## 🎯 Advanced Resume Ranker
        
        This application uses multiple AI agents to provide comprehensive resume analysis and ranking.
        
        ### 🤖 AI Agents
        
        1. **Resume Parser Agent**: Extracts structured data from resume PDFs
        2. **Skills Matcher Agent**: Matches candidate skills with job requirements
        3. **Experience Analyzer Agent**: Analyzes work experience and career progression
        4. **Culture Fit Agent**: Assesses cultural fit and soft skills
        
        ### 📊 Scoring System
        
        The system uses a weighted scoring approach:
        - **Skills Match (30%)**: How well candidate skills match requirements
        - **Experience Relevance (25%)**: Relevance and quality of work experience
        - **Education Match (15%)**: Education background alignment
        - **Culture Fit (15%)**: Soft skills and cultural alignment
        - **Career Progression (15%)**: Career growth and advancement patterns
        
        ### 🔍 Features
        
        - **Multi-Agent Analysis**: Comprehensive evaluation using specialized AI agents
        - **Advanced PDF Processing**: Enhanced text extraction from complex resume formats
        - **Database Persistence**: Store and track analysis results over time
        - **Export Options**: CSV, Excel, and JSON export formats
        - **Analytics Dashboard**: Insights and trends from candidate data
        - **Job Templates**: Pre-configured templates for common roles
        - **Customizable Scoring**: Adjust weights based on role requirements
        
        ### 💡 Best Practices
        
        1. **Prepare Clean PDFs**: Ensure resumes are text-based PDFs (not scanned images)
        2. **Define Clear Criteria**: Be specific about required vs. preferred skills
        3. **Use Appropriate Weights**: Adjust scoring weights based on role importance
        4. **Review Red Flags**: Pay attention to identified concerns and gaps
        5. **Consider Context**: Use detailed analysis to understand candidate fit
        
        ### 🚀 Getting Started
        
        1. Navigate to **Resume Analysis**
        2. Define your job requirements (or use a template)
        3. Upload a ZIP file containing PDF resumes
        4. Review results and export for further use
        
        ### 🔧 Troubleshooting
        
        - **No text extracted**: Ensure PDFs are text-based, not scanned images
        - **Low scores**: Review and adjust scoring criteria and weights
        - **API errors**: Check OpenAI API key and rate limits
        - **Large files**: Break large batches into smaller uploads
        """)

def main():
    """Main application entry point"""
    app = AdvancedResumeRankerApp()
    app.run()

if __name__ == "__main__":
    main()