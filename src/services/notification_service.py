import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any, Optional
import logging

class NotificationService:
    """Service for sending notifications about hiring activities"""
    
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", self.smtp_username)
        self.logger = logging.getLogger(__name__)
    
    def send_analysis_complete_notification(self, recipient_email: str, 
                                          job_title: str, 
                                          candidate_count: int,
                                          top_candidates: List[Dict[str, Any]]):
        """Send notification when resume analysis is complete"""
        
        subject = f"Resume Analysis Complete - {job_title}"
        
        # Create HTML email content
        html_content = f"""
        <html>
        <body>
            <h2>Resume Analysis Complete</h2>
            <p>The resume analysis for <strong>{job_title}</strong> has been completed.</p>
            
            <h3>Summary:</h3>
            <ul>
                <li>Total candidates analyzed: {candidate_count}</li>
                <li>Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
            </ul>
            
            <h3>Top Candidates:</h3>
            <table border="1" style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th>Rank</th>
                    <th>Candidate Name</th>
                    <th>Overall Score</th>
                    <th>Email</th>
                </tr>
        """
        
        for i, candidate in enumerate(top_candidates[:5], 1):
            html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{candidate.get('name', 'N/A')}</td>
                    <td>{candidate.get('score', 0):.1f}</td>
                    <td>{candidate.get('email', 'N/A')}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <p>Please log into the Resume Ranker system to view detailed analysis and export results.</p>
            
            <p>Best regards,<br>Resume Ranker System</p>
        </body>
        </html>
        """
        
        self._send_email(recipient_email, subject, html_content)
    
    def send_high_priority_candidate_alert(self, recipient_email: str,
                                         candidate_name: str,
                                         score: float,
                                         job_title: str,
                                         key_highlights: List[str]):
        """Send alert for high-priority candidates"""
        
        subject = f"🚨 High Priority Candidate Alert - {candidate_name}"
        
        html_content = f"""
        <html>
        <body>
            <h2>🚨 High Priority Candidate Detected</h2>
            <p>A high-scoring candidate has been identified for <strong>{job_title}</strong>.</p>
            
            <h3>Candidate Details:</h3>
            <ul>
                <li><strong>Name:</strong> {candidate_name}</li>
                <li><strong>Overall Score:</strong> {score:.1f}/100</li>
                <li><strong>Priority Level:</strong> {"🟢 Immediate Interview" if score >= 90 else "🟡 Strong Candidate"}</li>
            </ul>
            
            <h3>Key Highlights:</h3>
            <ul>
        """
        
        for highlight in key_highlights[:5]:
            html_content += f"<li>{highlight}</li>"
        
        html_content += """
            </ul>
            
            <p><strong>Recommendation:</strong> Schedule interview as soon as possible to avoid losing this candidate.</p>
            
            <p>Best regards,<br>Resume Ranker System</p>
        </body>
        </html>
        """
        
        self._send_email(recipient_email, subject, html_content)
    
    def send_weekly_report(self, recipient_email: str, analytics_data: Dict[str, Any]):
        """Send weekly hiring analytics report"""
        
        subject = "📊 Weekly Hiring Analytics Report"
        
        metrics = analytics_data.get('funnel_metrics', {})
        
        html_content = f"""
        <html>
        <body>
            <h2>📊 Weekly Hiring Analytics Report</h2>
            <p>Here's your weekly summary of hiring activities:</p>
            
            <h3>Key Metrics (Last 7 Days):</h3>
            <table border="1" style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr><td>Total Candidates Analyzed</td><td>{metrics.get('total_analyzed', 0)}</td></tr>
                <tr><td>Average Score</td><td>{metrics.get('avg_score', 0):.1f}</td></tr>
                <tr><td>Qualified Candidates (70+)</td><td>{metrics.get('qualified', 0)}</td></tr>
                <tr><td>Excellent Candidates (90+)</td><td>{metrics.get('excellent', 0)}</td></tr>
            </table>
            
            <h3>Recommendations:</h3>
            <ul>
                <li>Review top candidates for immediate interviews</li>
                <li>Consider adjusting criteria if qualification rates are too low/high</li>
                <li>Export results for detailed analysis</li>
            </ul>
            
            <p>Access the full dashboard for detailed insights and candidate profiles.</p>
            
            <p>Best regards,<br>Resume Ranker System</p>
        </body>
        </html>
        """
        
        self._send_email(recipient_email, subject, html_content)
    
    def _send_email(self, recipient: str, subject: str, html_content: str):
        """Send email using SMTP"""
        
        if not self.smtp_username or not self.smtp_password:
            self.logger.warning("SMTP credentials not configured. Email not sent.")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = recipient
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            self.logger.info(f"Email sent successfully to {recipient}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email to {recipient}: {str(e)}")
            return False
    
    def is_configured(self) -> bool:
        """Check if notification service is properly configured"""
        return bool(self.smtp_username and self.smtp_password)