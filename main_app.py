import streamlit as st
import zipfile
import os
import tempfile
import shutil
import io
import json
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Import our custom modules
from config import settings
from models import Candidate, JobPosting, AnalysisResult, Interview
from database import db_manager
from resume_parser import AdvancedResumeParser
from job_analyzer import JobDescriptionAnalyzer
from ai_agents import AIAgentOrchestrator
from scoring_engine import AdvancedScoringEngine
from analytics import BusinessIntelligenceDashboard

# Set page config
st.set_page_config(
    page_title="Advanced Resume Ranker Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_components():
    return {
        'parser': AdvancedResumeParser(),
        'job_analyzer': JobDescriptionAnalyzer(),
        'ai_orchestrator': AIAgentOrchestrator(),
        'scoring_engine': AdvancedScoringEngine(),
        'dashboard': BusinessIntelligenceDashboard()
    }

components = initialize_components()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🚀 Advanced Resume Ranker Pro</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        AI-Powered Talent Acquisition Platform with Advanced Analytics & Business Intelligence
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["🏠 Dashboard", "📄 Resume Analysis", "💼 Job Management", "👥 Candidate Pipeline", 
     "📊 Analytics", "⚙️ Settings", "🔧 API Tools"]
)

# API Key check
if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "sk-your-api-key-here":
    st.error("⚠️ Please set your OpenAI API key in the environment variables or config.py")
    st.stop()

# Dashboard Page
if page == "🏠 Dashboard":
    st.header("📊 Business Intelligence Dashboard")
    
    # Get analytics data
    analytics_data = db_manager.get_analytics_data(30)
    pipeline_data = db_manager.get_candidate_pipeline_data()
    
    if analytics_data:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Candidates",
                value=analytics_data.get('total_candidates', 0),
                delta=analytics_data.get('new_candidates', 0)
            )
        
        with col2:
            st.metric(
                label="Active Jobs",
                value=analytics_data.get('total_jobs', 0)
            )
        
        with col3:
            avg_score = analytics_data.get('average_scores', {}).get('overall', 0)
            st.metric(
                label="Avg Score",
                value=f"{avg_score:.1f}",
                delta=f"{avg_score - 70:.1f}" if avg_score > 0 else None
            )
        
        with col4:
            conversion_rate = pipeline_data.get('conversion_rates', {}).get('hire_rate', 0)
            st.metric(
                label="Hire Rate",
                value=f"{conversion_rate:.1f}%"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution
            score_dist = analytics_data.get('score_distribution', {})
            if score_dist:
                fig = components['dashboard'].create_score_distribution_chart(score_dist)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pipeline chart
            pipeline_stages = pipeline_data.get('pipeline_stages', [])
            if pipeline_stages:
                fig = components['dashboard'].create_pipeline_chart(pipeline_stages)
                st.plotly_chart(fig, use_container_width=True)
        
        # Skills and insights
        col1, col2 = st.columns(2)
        
        with col1:
            # Top skills
            top_skills = analytics_data.get('top_skills', [])
            if top_skills:
                fig = components['dashboard'].create_skills_heatmap(top_skills)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Business insights
            st.subheader("💡 Business Insights")
            insights = components['dashboard'].generate_insights(analytics_data)
            for insight in insights:
                st.info(f"• {insight}")
    
    else:
        st.info("No data available. Start by uploading resumes and creating job postings.")

# Resume Analysis Page
elif page == "📄 Resume Analysis":
    st.header("📄 Advanced Resume Analysis")
    
    # File upload section
    st.subheader("📁 Upload Resumes")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload resume files (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt", "doc"],
            accept_multiple_files=True,
            help="Upload multiple resume files for batch processing"
        )
    
    with col2:
        st.info("""
        **Supported Formats:**
        - PDF files
        - Word documents (.docx, .doc)
        - Text files (.txt)
        
        **File Size Limit:** 50MB per file
        """)
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded successfully")
        
        # Job description input
        st.subheader("💼 Job Description")
        
        job_title = st.text_input("Job Title", placeholder="e.g., Senior Python Developer")
        job_description = st.text_area(
            "Job Description",
            height=200,
            placeholder="Paste the complete job description here..."
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.selectbox("AI Model", ["gpt-4o", "gpt-4", "gpt-3.5-turbo"], index=0)
                st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
            
            with col2:
                st.checkbox("Enable Bias Detection", value=True)
                st.checkbox("Enable Cultural Fit Analysis", value=True)
                st.checkbox("Enable Technical Deep Dive", value=True)
        
        # Process button
        if st.button("🚀 Analyze Resumes", type="primary"):
            if not job_description:
                st.error("Please provide a job description")
            else:
                # Create job posting
                job_data = {
                    "title": job_title or "Untitled Position",
                    "description": job_description,
                    "requirements": {},
                    "skills_required": [],
                    "experience_required": 0,
                    "education_required": "",
                    "location": "",
                    "salary_range": ""
                }
                
                job_posting = db_manager.create_job_posting(job_data)
                
                if job_posting:
                    st.success(f"✅ Job posting created: {job_posting.title}")
                    
                    # Process files
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
                        
                        # Save file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            tmp_file_path = tmp_file.name
                        
                        try:
                            # Parse resume
                            parsed_resume = components['parser'].parse_file(tmp_file_path)
                            
                            if 'error' not in parsed_resume:
                                # Analyze job description
                                job_analysis = components['job_analyzer'].analyze_job_description(job_description)
                                
                                # AI agent analysis
                                ai_results = components['ai_orchestrator'].analyze_candidate(
                                    parsed_resume['raw_text'], 
                                    job_description
                                )
                                
                                # Calculate comprehensive score
                                scoring_results = components['scoring_engine'].calculate_comprehensive_score(
                                    parsed_resume, 
                                    job_analysis, 
                                    ai_results
                                )
                                
                                # Save to database
                                candidate_data = {
                                    "name": parsed_resume.get('personal_info', {}).get('name', 'Unknown'),
                                    "email": parsed_resume.get('contact_info', {}).get('email', ''),
                                    "phone": parsed_resume.get('contact_info', {}).get('phone', ''),
                                    "resume_text": parsed_resume['raw_text'],
                                    "resume_file_path": tmp_file_path,
                                    "skills": parsed_resume.get('skills', {}),
                                    "experience": parsed_resume.get('experience', []),
                                    "education": parsed_resume.get('education', []),
                                    "overall_score": scoring_results.get('overall_score', 0)
                                }
                                
                                candidate = db_manager.create_candidate(candidate_data)
                                
                                if candidate:
                                    # Save analysis result
                                    analysis_data = {
                                        "candidate_id": candidate.id,
                                        "job_posting_id": job_posting.id,
                                        "skills_match_score": scoring_results.get('component_scores', {}).get('skills_match', 0),
                                        "experience_score": scoring_results.get('component_scores', {}).get('experience_relevance', 0),
                                        "education_score": scoring_results.get('component_scores', {}).get('education', 0),
                                        "cultural_fit_score": scoring_results.get('component_scores', {}).get('cultural_fit', 0),
                                        "technical_score": scoring_results.get('component_scores', {}).get('technical_skills', 0),
                                        "overall_score": scoring_results.get('overall_score', 0),
                                        "analysis_details": scoring_results,
                                        "reasoning": ai_results.get('screener', {}).get('reasoning', ''),
                                        "recommendations": '; '.join(scoring_results.get('recommendations', [])),
                                        "screener_notes": ai_results.get('screener', {}).get('reasoning', ''),
                                        "technical_evaluator_notes": ai_results.get('technical_evaluator', {}).get('recommendations', ''),
                                        "cultural_assessor_notes": ai_results.get('cultural_fit_assessor', {}).get('overall_assessment', '')
                                    }
                                    
                                    db_manager.create_analysis_result(analysis_data)
                                
                                results.append({
                                    'filename': uploaded_file.name,
                                    'candidate_name': candidate_data['name'],
                                    'overall_score': scoring_results.get('overall_score', 0),
                                    'component_scores': scoring_results.get('component_scores', {}),
                                    'ai_results': ai_results,
                                    'recommendations': scoring_results.get('recommendations', [])
                                })
                            
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        
                        finally:
                            # Clean up temp file
                            if os.path.exists(tmp_file_path):
                                os.unlink(tmp_file_path)
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    status_text.text("✅ Analysis complete!")
                    
                    # Display results
                    if results:
                        st.subheader("📊 Analysis Results")
                        
                        # Sort by overall score
                        results.sort(key=lambda x: x['overall_score'], reverse=True)
                        
                        # Results table
                        df_data = []
                        for i, result in enumerate(results):
                            df_data.append({
                                'Rank': i + 1,
                                'Name': result['candidate_name'],
                                'File': result['filename'],
                                'Overall Score': f"{result['overall_score']:.1f}",
                                'Skills': f"{result['component_scores'].get('skills_match', 0):.1f}",
                                'Experience': f"{result['component_scores'].get('experience_relevance', 0):.1f}",
                                'Education': f"{result['component_scores'].get('education', 0):.1f}",
                                'Cultural Fit': f"{result['component_scores'].get('cultural_fit', 0):.1f}",
                                'Technical': f"{result['component_scores'].get('technical_skills', 0):.1f}"
                            })
                        
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results as CSV",
                            csv,
                            f"resume_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv"
                        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Advanced Resume Ranker Pro v2.0.0 | Powered by AI & Advanced Analytics</p>
    <p>Built with Streamlit, OpenAI, and Advanced Machine Learning</p>
</div>
""", unsafe_allow_html=True)