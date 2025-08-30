import streamlit as st
import tempfile
import shutil
import os
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def create_temp_directory() -> str:
    """Create a temporary directory for file processing"""
    return tempfile.mkdtemp()

def cleanup_temp_directory(temp_dir: str):
    """Clean up temporary directory"""
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        st.warning(f"Could not clean up temporary directory: {str(e)}")

def format_score(score: float) -> str:
    """Format score with color coding"""
    if score >= 90:
        return f"🟢 {score:.1f}"
    elif score >= 80:
        return f"🟡 {score:.1f}"
    elif score >= 70:
        return f"🟠 {score:.1f}"
    else:
        return f"🔴 {score:.1f}"

def create_score_visualization(scores: List[float], labels: List[str]) -> go.Figure:
    """Create a radar chart for score visualization"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=labels,
        fill='toself',
        name='Scores'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Candidate Score Breakdown"
    )
    
    return fig

def create_score_distribution_chart(resume_scores: List[Any]) -> go.Figure:
    """Create score distribution histogram"""
    scores = [score.overall_score for score in resume_scores]
    
    fig = px.histogram(
        x=scores,
        bins=20,
        title="Score Distribution",
        labels={'x': 'Overall Score', 'y': 'Number of Candidates'}
    )
    
    fig.update_layout(
        xaxis_title="Overall Score",
        yaxis_title="Number of Candidates",
        showlegend=False
    )
    
    return fig

def create_skills_comparison_chart(resume_scores: List[Any], top_n: int = 10) -> go.Figure:
    """Create skills frequency comparison chart"""
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
    
    # Get top skills
    sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    skills, counts = zip(*sorted_skills) if sorted_skills else ([], [])
    
    fig = px.bar(
        x=list(counts),
        y=list(skills),
        orientation='h',
        title=f"Top {top_n} Most Common Skills",
        labels={'x': 'Number of Candidates', 'y': 'Skills'}
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def display_candidate_card(resume_score: Any, rank: int):
    """Display a candidate card with key information"""
    
    score_color = "green" if resume_score.overall_score >= 80 else "orange" if resume_score.overall_score >= 60 else "red"
    
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"### #{rank} {resume_score.resume_data.candidate_name}")
            st.markdown(f"**File:** {resume_score.resume_data.filename}")
            
            # Contact info
            contact = resume_score.resume_data.contact
            contact_info = []
            if contact.email:
                contact_info.append(f"📧 {contact.email}")
            if contact.phone:
                contact_info.append(f"📞 {contact.phone}")
            if contact.location:
                contact_info.append(f"📍 {contact.location}")
            
            if contact_info:
                st.markdown(" | ".join(contact_info))
        
        with col2:
            st.metric("Overall Score", f"{resume_score.overall_score:.1f}", delta=None)
        
        with col3:
            if contact.linkedin:
                st.markdown(f"[LinkedIn]({contact.linkedin})")
            if contact.github:
                st.markdown(f"[GitHub]({contact.github})")

def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format"""
    return api_key and api_key.startswith('sk-') and len(api_key) > 20

def calculate_hiring_priority(resume_score: Any) -> str:
    """Calculate hiring priority based on scores and analysis"""
    overall = resume_score.overall_score
    red_flags_count = len(resume_score.red_flags)
    
    if overall >= 90 and red_flags_count == 0:
        return "🟢 Immediate Interview"
    elif overall >= 80 and red_flags_count <= 1:
        return "🟡 Strong Candidate"
    elif overall >= 70 and red_flags_count <= 2:
        return "🟠 Consider for Interview"
    elif overall >= 60:
        return "🔵 Potential with Development"
    else:
        return "🔴 Not Recommended"

def format_duration(months: Optional[int]) -> str:
    """Format duration in months to human readable format"""
    if not months:
        return "Unknown"
    
    if months < 12:
        return f"{months} month{'s' if months != 1 else ''}"
    else:
        years = months // 12
        remaining_months = months % 12
        
        if remaining_months == 0:
            return f"{years} year{'s' if years != 1 else ''}"
        else:
            return f"{years} year{'s' if years != 1 else ''}, {remaining_months} month{'s' if remaining_months != 1 else ''}"

def extract_keywords_from_text(text: str, min_length: int = 3) -> List[str]:
    """Extract keywords from text for analysis"""
    import re
    
    # Remove common stop words
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'among', 'through', 'during',
        'before', 'after', 'above', 'below', 'between'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Filter words
    keywords = [
        word for word in words 
        if len(word) >= min_length and word not in stop_words
    ]
    
    # Count frequency and return most common
    from collections import Counter
    word_counts = Counter(keywords)
    
    return [word for word, count in word_counts.most_common(50)]

def create_comparison_table(resume_scores: List[Any], criteria: List[str]) -> Dict[str, Any]:
    """Create comparison table data for multiple candidates"""
    
    table_data = []
    
    for i, score in enumerate(resume_scores[:10], 1):  # Top 10 candidates
        row = {
            'Rank': i,
            'Candidate': score.resume_data.candidate_name,
            'Overall': f"{score.overall_score:.1f}",
            'Skills': f"{score.skill_score:.1f}",
            'Experience': f"{score.experience_score:.1f}",
            'Education': f"{score.education_score:.1f}",
            'Culture Fit': f"{score.culture_fit_score:.1f}",
            'Priority': calculate_hiring_priority(score)
        }
        table_data.append(row)
    
    return table_data