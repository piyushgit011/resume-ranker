import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Dict, List, Any, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import json
import os
from database import db_manager
from models import Interview
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterviewScheduler:
    def __init__(self):
        self.time_slots = self._generate_time_slots()
        self.interview_types = ["Phone", "Video", "In-Person", "Technical", "HR", "Panel"]
    
    def _generate_time_slots(self) -> List[Dict[str, Any]]:
        """Generate available time slots for the next 30 days"""
        slots = []
        start_date = datetime.now().date()
        
        for i in range(30):  # Next 30 days
            current_date = start_date + timedelta(days=i)
            
            # Skip weekends
            if current_date.weekday() >= 5:
                continue
            
            # Generate time slots for each day (9 AM to 5 PM, 1-hour slots)
            for hour in range(9, 17):
                slot_time = time(hour, 0)
                slots.append({
                    'date': current_date,
                    'time': slot_time,
                    'datetime': datetime.combine(current_date, slot_time),
                    'available': True,
                    'formatted': f"{current_date.strftime('%Y-%m-%d')} {slot_time.strftime('%H:%M')}"
                })
        
        return slots
    
    def schedule_interview(self, candidate_id: int, job_posting_id: int, 
                          scheduled_datetime: datetime, interview_type: str, 
                          interviewer: str, notes: str = "") -> bool:
        """Schedule an interview"""
        try:
            interview_data = {
                "candidate_id": candidate_id,
                "job_posting_id": job_posting_id,
                "scheduled_at": scheduled_datetime,
                "interview_type": interview_type,
                "interviewer": interviewer,
                "status": "scheduled",
                "notes": notes
            }
            
            interview = db_manager.create_interview(interview_data)
            return interview is not None
            
        except Exception as e:
            logger.error(f"Error scheduling interview: {str(e)}")
            return False
    
    def get_available_slots(self, date: datetime.date = None) -> List[Dict[str, Any]]:
        """Get available time slots for a specific date"""
        if date is None:
            date = datetime.now().date()
        
        # Get existing interviews for the date
        existing_interviews = db_manager.get_interviews()
        scheduled_times = []
        
        for interview in existing_interviews:
            if interview.scheduled_at.date() == date:
                scheduled_times.append(interview.scheduled_at.time())
        
        # Filter available slots
        available_slots = []
        for slot in self.time_slots:
            if (slot['date'] == date and 
                slot['time'] not in scheduled_times and 
                slot['datetime'] > datetime.now()):
                available_slots.append(slot)
        
        return available_slots
    
    def get_interview_calendar(self, days: int = 7) -> pd.DataFrame:
        """Get interview calendar for the next N days"""
        start_date = datetime.now().date()
        end_date = start_date + timedelta(days=days)
        
        interviews = db_manager.get_interviews()
        calendar_data = []
        
        for interview in interviews:
            if start_date <= interview.scheduled_at.date() <= end_date:
                candidate = db_manager.get_candidate(interview.candidate_id)
                job = db_manager.get_job_posting(interview.job_posting_id)
                
                calendar_data.append({
                    'Date': interview.scheduled_at.date(),
                    'Time': interview.scheduled_at.time(),
                    'Candidate': candidate.name if candidate else 'Unknown',
                    'Job': job.title if job else 'Unknown',
                    'Type': interview.interview_type,
                    'Interviewer': interview.interviewer,
                    'Status': interview.status
                })
        
        return pd.DataFrame(calendar_data)
    
    def send_interview_invitation(self, candidate_email: str, candidate_name: str,
                                 interview_datetime: datetime, interview_type: str,
                                 interviewer: str, job_title: str) -> bool:
        """Send interview invitation email"""
        try:
            # Email configuration (you would set these in environment variables)
            smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
            smtp_port = int(os.getenv("SMTP_PORT", "587"))
            email_user = os.getenv("EMAIL_USER", "")
            email_password = os.getenv("EMAIL_PASSWORD", "")
            
            if not email_user or not email_password:
                logger.warning("Email credentials not configured")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_user
            msg['To'] = candidate_email
            msg['Subject'] = f"Interview Invitation - {job_title}"
            
            # Email body
            body = f"""
            Dear {candidate_name},
            
            Thank you for your interest in the {job_title} position. We are pleased to invite you for an interview.
            
            Interview Details:
            - Date: {interview_datetime.strftime('%B %d, %Y')}
            - Time: {interview_datetime.strftime('%I:%M %p')}
            - Type: {interview_type}
            - Interviewer: {interviewer}
            
            Please confirm your availability by replying to this email.
            
            Best regards,
            HR Team
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email_user, email_password)
            text = msg.as_string()
            server.sendmail(email_user, candidate_email, text)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return False
    
    def send_interview_reminder(self, candidate_email: str, candidate_name: str,
                               interview_datetime: datetime, job_title: str) -> bool:
        """Send interview reminder email"""
        try:
            smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
            smtp_port = int(os.getenv("SMTP_PORT", "587"))
            email_user = os.getenv("EMAIL_USER", "")
            email_password = os.getenv("EMAIL_PASSWORD", "")
            
            if not email_user or not email_password:
                return False
            
            msg = MIMEMultipart()
            msg['From'] = email_user
            msg['To'] = candidate_email
            msg['Subject'] = f"Interview Reminder - {job_title}"
            
            body = f"""
            Dear {candidate_name},
            
            This is a reminder about your upcoming interview for the {job_title} position.
            
            Interview Details:
            - Date: {interview_datetime.strftime('%B %d, %Y')}
            - Time: {interview_datetime.strftime('%I:%M %p')}
            
            Please arrive 10 minutes early and bring a copy of your resume.
            
            Best regards,
            HR Team
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email_user, email_password)
            text = msg.as_string()
            server.sendmail(email_user, candidate_email, text)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending reminder: {str(e)}")
            return False
    
    def update_interview_status(self, interview_id: int, status: str, 
                               rating: int = None, feedback: str = None) -> bool:
        """Update interview status and add feedback"""
        try:
            session = db_manager.get_session()
            interview = session.query(Interview).filter(Interview.id == interview_id).first()
            
            if interview:
                interview.status = status
                if rating is not None:
                    interview.rating = rating
                if feedback is not None:
                    interview.feedback = feedback
                
                session.commit()
                session.close()
                return True
            
            session.close()
            return False
            
        except Exception as e:
            logger.error(f"Error updating interview: {str(e)}")
            return False
    
    def get_interview_statistics(self) -> Dict[str, Any]:
        """Get interview statistics"""
        try:
            interviews = db_manager.get_interviews()
            
            total_interviews = len(interviews)
            completed_interviews = len([i for i in interviews if i.status == "completed"])
            scheduled_interviews = len([i for i in interviews if i.status == "scheduled"])
            cancelled_interviews = len([i for i in interviews if i.status == "cancelled"])
            
            # Average rating
            rated_interviews = [i for i in interviews if i.rating is not None]
            avg_rating = sum(i.rating for i in rated_interviews) / len(rated_interviews) if rated_interviews else 0
            
            # Interview type distribution
            type_distribution = {}
            for interview in interviews:
                interview_type = interview.interview_type
                type_distribution[interview_type] = type_distribution.get(interview_type, 0) + 1
            
            return {
                "total_interviews": total_interviews,
                "completed_interviews": completed_interviews,
                "scheduled_interviews": scheduled_interviews,
                "cancelled_interviews": cancelled_interviews,
                "completion_rate": (completed_interviews / total_interviews * 100) if total_interviews > 0 else 0,
                "average_rating": avg_rating,
                "type_distribution": type_distribution
            }
            
        except Exception as e:
            logger.error(f"Error getting interview statistics: {str(e)}")
            return {}

def render_interview_scheduler():
    """Render the interview scheduler interface"""
    scheduler = InterviewScheduler()
    
    st.header("📅 Interview Scheduler")
    
    # Tabs for different functions
    tab1, tab2, tab3, tab4 = st.tabs(["Schedule Interview", "Interview Calendar", "Send Communications", "Interview Analytics"])
    
    with tab1:
        st.subheader("Schedule New Interview")
        
        # Get candidates and jobs
        candidates = db_manager.get_candidates(limit=100)
        jobs = db_manager.get_job_postings(active_only=True)
        
        if not candidates:
            st.warning("No candidates available. Please add candidates first.")
            return
        
        if not jobs:
            st.warning("No active job postings available. Please create job postings first.")
            return
        
        with st.form("schedule_interview_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                candidate_options = {f"{c.name} ({c.email})": c.id for c in candidates}
                selected_candidate = st.selectbox("Select Candidate", list(candidate_options.keys()))
                candidate_id = candidate_options[selected_candidate]
                
                job_options = {f"{j.title}": j.id for j in jobs}
                selected_job = st.selectbox("Select Job", list(job_options.keys()))
                job_id = job_options[selected_job]
                
                interview_type = st.selectbox("Interview Type", scheduler.interview_types)
            
            with col2:
                # Date selection
                selected_date = st.date_input("Select Date", min_value=datetime.now().date())
                
                # Time selection
                available_slots = scheduler.get_available_slots(selected_date)
                if available_slots:
                    time_options = {slot['formatted']: slot['datetime'] for slot in available_slots}
                    selected_time = st.selectbox("Select Time", list(time_options.keys()))
                    scheduled_datetime = time_options[selected_time]
                else:
                    st.warning("No available time slots for the selected date.")
                    scheduled_datetime = None
                
                interviewer = st.text_input("Interviewer Name", placeholder="e.g., John Smith")
            
            notes = st.text_area("Notes", placeholder="Additional notes for the interview...")
            
            submitted = st.form_submit_button("Schedule Interview", type="primary")
            
            if submitted and scheduled_datetime:
                if not interviewer:
                    st.error("Please enter interviewer name")
                else:
                    success = scheduler.schedule_interview(
                        candidate_id, job_id, scheduled_datetime, 
                        interview_type, interviewer, notes
                    )
                    
                    if success:
                        st.success("✅ Interview scheduled successfully!")
                        
                        # Send invitation email
                        candidate = db_manager.get_candidate(candidate_id)
                        job = db_manager.get_job_posting(job_id)
                        
                        if candidate and candidate.email:
                            email_sent = scheduler.send_interview_invitation(
                                candidate.email, candidate.name, scheduled_datetime,
                                interview_type, interviewer, job.title
                            )
                            
                            if email_sent:
                                st.success("📧 Interview invitation email sent!")
                            else:
                                st.warning("⚠️ Interview scheduled but email could not be sent")
                    else:
                        st.error("❌ Failed to schedule interview")
    
    with tab2:
        st.subheader("Interview Calendar")
        
        # Calendar view
        days_ahead = st.slider("Show next N days", 1, 30, 7)
        calendar_df = scheduler.get_interview_calendar(days_ahead)
        
        if not calendar_df.empty:
            st.dataframe(calendar_df, use_container_width=True)
            
            # Download calendar
            csv = calendar_df.to_csv(index=False)
            st.download_button(
                "📥 Download Calendar",
                csv,
                f"interview_calendar_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        else:
            st.info("No interviews scheduled for the selected period.")
        
        # Interview management
        st.subheader("Manage Interviews")
        
        interviews = db_manager.get_interviews()
        if interviews:
            for interview in interviews:
                with st.expander(f"Interview - {interview.scheduled_at.strftime('%Y-%m-%d %H:%M')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        candidate = db_manager.get_candidate(interview.candidate_id)
                        job = db_manager.get_job_posting(interview.job_posting_id)
                        
                        st.write(f"**Candidate:** {candidate.name if candidate else 'Unknown'}")
                        st.write(f"**Job:** {job.title if job else 'Unknown'}")
                        st.write(f"**Type:** {interview.interview_type}")
                        st.write(f"**Interviewer:** {interview.interviewer}")
                        st.write(f"**Status:** {interview.status}")
                    
                    with col2:
                        # Update status
                        new_status = st.selectbox(
                            "Update Status",
                            ["scheduled", "completed", "cancelled"],
                            index=["scheduled", "completed", "cancelled"].index(interview.status),
                            key=f"status_{interview.id}"
                        )
                        
                        if new_status == "completed":
                            rating = st.slider("Rating (1-5)", 1, 5, 3, key=f"rating_{interview.id}")
                            feedback = st.text_area("Feedback", key=f"feedback_{interview.id}")
                        else:
                            rating = None
                            feedback = None
                        
                        if st.button(f"Update", key=f"update_{interview.id}"):
                            success = scheduler.update_interview_status(
                                interview.id, new_status, rating, feedback
                            )
                            
                            if success:
                                st.success("Interview updated successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to update interview")
    
    with tab3:
        st.subheader("Send Communications")
        
        # Get scheduled interviews
        interviews = db_manager.get_interviews()
        upcoming_interviews = [i for i in interviews if i.scheduled_at > datetime.now() and i.status == "scheduled"]
        
        if upcoming_interviews:
            st.write("**Upcoming Interviews:**")
            
            for interview in upcoming_interviews:
                candidate = db_manager.get_candidate(interview.candidate_id)
                job = db_manager.get_job_posting(interview.job_posting_id)
                
                if candidate and candidate.email:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"{candidate.name} - {job.title if job else 'Unknown'} - {interview.scheduled_at.strftime('%Y-%m-%d %H:%M')}")
                    
                    with col2:
                        if st.button(f"Send Reminder", key=f"reminder_{interview.id}"):
                            success = scheduler.send_interview_reminder(
                                candidate.email, candidate.name, 
                                interview.scheduled_at, job.title if job else "Position"
                            )
                            
                            if success:
                                st.success("Reminder sent!")
                            else:
                                st.error("Failed to send reminder")
                    
                    with col3:
                        if st.button(f"Reschedule", key=f"reschedule_{interview.id}"):
                            st.info("Reschedule functionality would be implemented here")
        else:
            st.info("No upcoming interviews to send communications for.")
    
    with tab4:
        st.subheader("Interview Analytics")
        
        stats = scheduler.get_interview_statistics()
        
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Interviews", stats['total_interviews'])
            
            with col2:
                st.metric("Completed", stats['completed_interviews'])
            
            with col3:
                st.metric("Completion Rate", f"{stats['completion_rate']:.1f}%")
            
            with col4:
                st.metric("Average Rating", f"{stats['average_rating']:.1f}")
            
            # Interview type distribution
            if stats['type_distribution']:
                st.subheader("Interview Type Distribution")
                type_df = pd.DataFrame(list(stats['type_distribution'].items()), 
                                     columns=['Type', 'Count'])
                st.bar_chart(type_df.set_index('Type'))
        else:
            st.info("No interview data available.")