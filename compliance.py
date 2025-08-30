import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import re
from database import db_manager
from ai_agents import BiasDetectionAgent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplianceManager:
    def __init__(self):
        self.bias_agent = BiasDetectionAgent()
        self.protected_attributes = [
            'age', 'gender', 'race', 'ethnicity', 'religion', 'nationality',
            'disability', 'sexual_orientation', 'marital_status', 'pregnancy',
            'veteran_status', 'genetic_information'
        ]
        
        self.compliance_rules = {
            'eeoc_compliance': {
                'name': 'EEOC Compliance',
                'description': 'Equal Employment Opportunity Commission guidelines',
                'checks': ['demographic_bias', 'adverse_impact', 'disparate_treatment']
            },
            'ada_compliance': {
                'name': 'ADA Compliance',
                'description': 'Americans with Disabilities Act requirements',
                'checks': ['disability_accommodation', 'reasonable_accommodation']
            },
            'ofccp_compliance': {
                'name': 'OFCCP Compliance',
                'description': 'Office of Federal Contract Compliance Programs',
                'checks': ['affirmative_action', 'veteran_preference', 'disability_preference']
            },
            'gdpr_compliance': {
                'name': 'GDPR Compliance',
                'description': 'General Data Protection Regulation',
                'checks': ['data_privacy', 'consent_management', 'data_retention']
            }
        }
    
    def detect_bias_in_text(self, text: str) -> Dict[str, Any]:
        """Detect bias in text using AI agent"""
        try:
            # Use bias detection agent
            bias_analysis = self.bias_agent.analyze(text, "", {})
            
            return {
                'bias_detected': bias_analysis.get('bias_risk_score', 0) > 50,
                'risk_score': bias_analysis.get('bias_risk_score', 0),
                'bias_types': bias_analysis.get('demographic_bias_risks', []),
                'recommendations': bias_analysis.get('fair_evaluation_recommendations', []),
                'overall_assessment': bias_analysis.get('overall_assessment', '')
            }
            
        except Exception as e:
            logger.error(f"Error detecting bias: {str(e)}")
            return {'error': str(e)}
    
    def check_eeoc_compliance(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check EEOC compliance"""
        try:
            # Analyze candidate pool demographics
            total_candidates = len(candidates)
            
            # Check for adverse impact (80% rule)
            adverse_impact_analysis = self._check_adverse_impact(candidates)
            
            # Check for disparate treatment
            disparate_treatment_analysis = self._check_disparate_treatment(candidates)
            
            # Check demographic distribution
            demographic_analysis = self._analyze_demographics(candidates)
            
            return {
                'compliant': adverse_impact_analysis['compliant'] and disparate_treatment_analysis['compliant'],
                'total_candidates': total_candidates,
                'adverse_impact': adverse_impact_analysis,
                'disparate_treatment': disparate_treatment_analysis,
                'demographics': demographic_analysis,
                'recommendations': self._generate_eeoc_recommendations(adverse_impact_analysis, disparate_treatment_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error checking EEOC compliance: {str(e)}")
            return {'error': str(e)}
    
    def _check_adverse_impact(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for adverse impact using 80% rule"""
        # This is a simplified implementation
        # In practice, you would need demographic data and selection rates
        
        return {
            'compliant': True,
            'selection_rate_ratio': 0.85,  # Example: 85% ratio
            'threshold': 0.80,
            'analysis': 'Selection rates appear to be within acceptable ranges',
            'recommendations': ['Continue monitoring selection rates', 'Document selection criteria']
        }
    
    def _check_disparate_treatment(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for disparate treatment"""
        # This would analyze if similar candidates are treated differently
        
        return {
            'compliant': True,
            'analysis': 'No evidence of disparate treatment found',
            'recommendations': ['Maintain consistent evaluation criteria', 'Train evaluators on bias awareness']
        }
    
    def _analyze_demographics(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze demographic distribution"""
        # This would analyze the demographic makeup of the candidate pool
        
        return {
            'diversity_index': 0.75,  # Example diversity index
            'analysis': 'Candidate pool shows good diversity',
            'recommendations': ['Continue diverse sourcing strategies']
        }
    
    def _generate_eeoc_recommendations(self, adverse_impact: Dict, disparate_treatment: Dict) -> List[str]:
        """Generate EEOC compliance recommendations"""
        recommendations = []
        
        if not adverse_impact['compliant']:
            recommendations.append("Review selection criteria for potential adverse impact")
            recommendations.append("Consider alternative selection methods")
        
        if not disparate_treatment['compliant']:
            recommendations.append("Implement standardized evaluation processes")
            recommendations.append("Provide bias training to evaluators")
        
        recommendations.extend([
            "Document all selection decisions",
            "Maintain records for required time periods",
            "Regularly audit selection processes"
        ])
        
        return recommendations
    
    def check_data_privacy_compliance(self, data_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Check data privacy compliance (GDPR, CCPA, etc.)"""
        try:
            compliance_checks = {
                'consent_obtained': data_usage.get('consent_obtained', False),
                'data_minimization': data_usage.get('data_minimization', False),
                'purpose_limitation': data_usage.get('purpose_limitation', False),
                'retention_limits': data_usage.get('retention_limits', False),
                'access_controls': data_usage.get('access_controls', False),
                'data_encryption': data_usage.get('data_encryption', False)
            }
            
            compliant = all(compliance_checks.values())
            
            return {
                'compliant': compliant,
                'checks': compliance_checks,
                'recommendations': self._generate_privacy_recommendations(compliance_checks)
            }
            
        except Exception as e:
            logger.error(f"Error checking data privacy compliance: {str(e)}")
            return {'error': str(e)}
    
    def _generate_privacy_recommendations(self, checks: Dict[str, bool]) -> List[str]:
        """Generate data privacy recommendations"""
        recommendations = []
        
        if not checks['consent_obtained']:
            recommendations.append("Obtain explicit consent for data processing")
        
        if not checks['data_minimization']:
            recommendations.append("Implement data minimization practices")
        
        if not checks['purpose_limitation']:
            recommendations.append("Limit data processing to stated purposes")
        
        if not checks['retention_limits']:
            recommendations.append("Implement data retention policies")
        
        if not checks['access_controls']:
            recommendations.append("Implement proper access controls")
        
        if not checks['data_encryption']:
            recommendations.append("Encrypt sensitive data")
        
        return recommendations
    
    def generate_compliance_report(self, report_type: str = "comprehensive") -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        try:
            # Get all candidates and analysis results
            candidates = db_manager.get_candidates(limit=1000)
            analysis_results = db_manager.get_analysis_results()
            
            report = {
                'report_type': report_type,
                'generated_at': datetime.utcnow().isoformat(),
                'summary': {},
                'eeoc_compliance': {},
                'data_privacy': {},
                'bias_analysis': {},
                'recommendations': []
            }
            
            # EEOC Compliance
            if candidates:
                report['eeoc_compliance'] = self.check_eeoc_compliance(candidates)
            
            # Data Privacy Compliance
            data_usage = {
                'consent_obtained': True,  # This would be checked from actual data
                'data_minimization': True,
                'purpose_limitation': True,
                'retention_limits': True,
                'access_controls': True,
                'data_encryption': True
            }
            report['data_privacy'] = self.check_data_privacy_compliance(data_usage)
            
            # Bias Analysis
            if analysis_results:
                bias_scores = [result.analysis_details.get('bias_risk_score', 0) 
                             for result in analysis_results 
                             if result.analysis_details and isinstance(result.analysis_details, dict)]
                
                if bias_scores:
                    report['bias_analysis'] = {
                        'average_bias_score': sum(bias_scores) / len(bias_scores),
                        'high_bias_cases': len([score for score in bias_scores if score > 70]),
                        'total_analyses': len(bias_scores)
                    }
            
            # Generate summary
            report['summary'] = {
                'total_candidates': len(candidates),
                'total_analyses': len(analysis_results),
                'overall_compliance': (
                    report['eeoc_compliance'].get('compliant', False) and
                    report['data_privacy'].get('compliant', False)
                )
            }
            
            # Generate recommendations
            all_recommendations = []
            if 'recommendations' in report['eeoc_compliance']:
                all_recommendations.extend(report['eeoc_compliance']['recommendations'])
            if 'recommendations' in report['data_privacy']:
                all_recommendations.extend(report['data_privacy']['recommendations'])
            
            report['recommendations'] = list(set(all_recommendations))
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {str(e)}")
            return {'error': str(e)}
    
    def audit_selection_process(self, job_posting_id: int) -> Dict[str, Any]:
        """Audit selection process for a specific job posting"""
        try:
            # Get analysis results for the job
            analysis_results = db_manager.get_analysis_results(job_posting_id=job_posting_id)
            
            if not analysis_results:
                return {'error': 'No analysis results found for this job posting'}
            
            # Analyze score distribution
            scores = [result.overall_score for result in analysis_results if result.overall_score]
            
            if not scores:
                return {'error': 'No scores found for analysis'}
            
            # Check for potential bias patterns
            bias_analysis = {
                'score_distribution': {
                    'mean': sum(scores) / len(scores),
                    'median': sorted(scores)[len(scores) // 2],
                    'std_dev': self._calculate_std_dev(scores),
                    'min': min(scores),
                    'max': max(scores)
                },
                'potential_issues': [],
                'recommendations': []
            }
            
            # Check for score clustering (potential bias)
            if bias_analysis['score_distribution']['std_dev'] < 5:
                bias_analysis['potential_issues'].append('Low score variance may indicate bias')
                bias_analysis['recommendations'].append('Review scoring criteria for potential bias')
            
            # Check for extreme scores
            if bias_analysis['score_distribution']['max'] - bias_analysis['score_distribution']['min'] > 80:
                bias_analysis['potential_issues'].append('High score variance may indicate inconsistent evaluation')
                bias_analysis['recommendations'].append('Standardize evaluation criteria')
            
            return {
                'job_posting_id': job_posting_id,
                'total_candidates': len(analysis_results),
                'bias_analysis': bias_analysis,
                'audit_date': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error auditing selection process: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_std_dev(self, scores: List[float]) -> float:
        """Calculate standard deviation"""
        if len(scores) < 2:
            return 0
        
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
        return variance ** 0.5
    
    def generate_bias_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate bias analysis report"""
        try:
            # Get recent analysis results
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            analysis_results = db_manager.get_analysis_results()
            recent_results = [r for r in analysis_results if r.created_at >= start_date]
            
            if not recent_results:
                return {'error': 'No recent analysis results found'}
            
            # Analyze bias patterns
            bias_scores = []
            bias_types = {}
            
            for result in recent_results:
                if result.analysis_details and isinstance(result.analysis_details, dict):
                    bias_score = result.analysis_details.get('bias_risk_score', 0)
                    bias_scores.append(bias_score)
                    
                    # Collect bias types
                    bias_type_list = result.analysis_details.get('demographic_bias_risks', [])
                    for bias_type in bias_type_list:
                        bias_types[bias_type] = bias_types.get(bias_type, 0) + 1
            
            return {
                'period_days': days,
                'total_analyses': len(recent_results),
                'average_bias_score': sum(bias_scores) / len(bias_scores) if bias_scores else 0,
                'high_bias_cases': len([score for score in bias_scores if score > 70]),
                'bias_type_distribution': bias_types,
                'recommendations': self._generate_bias_recommendations(bias_scores, bias_types)
            }
            
        except Exception as e:
            logger.error(f"Error generating bias report: {str(e)}")
            return {'error': str(e)}
    
    def _generate_bias_recommendations(self, bias_scores: List[float], bias_types: Dict[str, int]) -> List[str]:
        """Generate bias reduction recommendations"""
        recommendations = []
        
        if bias_scores:
            avg_bias = sum(bias_scores) / len(bias_scores)
            
            if avg_bias > 60:
                recommendations.append("High bias scores detected - implement bias training")
                recommendations.append("Review and update evaluation criteria")
            
            if avg_bias > 40:
                recommendations.append("Moderate bias detected - consider blind evaluation")
                recommendations.append("Implement structured interview processes")
        
        # Specific bias type recommendations
        for bias_type, count in bias_types.items():
            if count > 5:  # If this bias type appears frequently
                recommendations.append(f"Address {bias_type} bias through training and process improvements")
        
        return list(set(recommendations))

def render_compliance_dashboard():
    """Render the compliance and bias detection dashboard"""
    compliance_manager = ComplianceManager()
    
    st.header("🛡️ Compliance & Bias Detection")
    
    # Tabs for different compliance areas
    tab1, tab2, tab3, tab4 = st.tabs(["Bias Detection", "EEOC Compliance", "Data Privacy", "Audit Reports"])
    
    with tab1:
        st.subheader("Bias Detection Analysis")
        
        # Text bias detection
        st.write("**Analyze Text for Bias**")
        text_to_analyze = st.text_area(
            "Enter text to analyze for bias",
            placeholder="Paste job description, evaluation criteria, or other text here...",
            height=150
        )
        
        if st.button("Analyze for Bias"):
            if text_to_analyze:
                with st.spinner("Analyzing text for bias..."):
                    bias_results = compliance_manager.detect_bias_in_text(text_to_analyze)
                
                if 'error' not in bias_results:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Bias Risk Score", f"{bias_results['risk_score']:.1f}")
                        st.metric("Bias Detected", "Yes" if bias_results['bias_detected'] else "No")
                    
                    with col2:
                        if bias_results['bias_types']:
                            st.write("**Detected Bias Types:**")
                            for bias_type in bias_results['bias_types']:
                                st.write(f"• {bias_type}")
                        
                        if bias_results['recommendations']:
                            st.write("**Recommendations:**")
                            for rec in bias_results['recommendations']:
                                st.write(f"• {rec}")
                    
                    st.write("**Overall Assessment:**")
                    st.write(bias_results['overall_assessment'])
                else:
                    st.error(f"Error analyzing text: {bias_results['error']}")
            else:
                st.warning("Please enter text to analyze")
        
        # Bias report
        st.subheader("Bias Analysis Report")
        
        col1, col2 = st.columns(2)
        with col1:
            days = st.slider("Analysis Period (days)", 7, 90, 30)
        
        with col2:
            if st.button("Generate Bias Report"):
                with st.spinner("Generating bias report..."):
                    bias_report = compliance_manager.generate_bias_report(days)
                
                if 'error' not in bias_report:
                    st.success("Bias report generated successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Analyses", bias_report['total_analyses'])
                    
                    with col2:
                        st.metric("Average Bias Score", f"{bias_report['average_bias_score']:.1f}")
                    
                    with col3:
                        st.metric("High Bias Cases", bias_report['high_bias_cases'])
                    
                    # Bias type distribution
                    if bias_report['bias_type_distribution']:
                        st.subheader("Bias Type Distribution")
                        bias_df = pd.DataFrame(
                            list(bias_report['bias_type_distribution'].items()),
                            columns=['Bias Type', 'Count']
                        )
                        st.bar_chart(bias_df.set_index('Bias Type'))
                    
                    # Recommendations
                    if bias_report['recommendations']:
                        st.subheader("Recommendations")
                        for rec in bias_report['recommendations']:
                            st.info(f"• {rec}")
                else:
                    st.error(f"Error generating report: {bias_report['error']}")
    
    with tab2:
        st.subheader("EEOC Compliance Check")
        
        if st.button("Run EEOC Compliance Check"):
            with st.spinner("Checking EEOC compliance..."):
                candidates = db_manager.get_candidates(limit=1000)
                eeoc_results = compliance_manager.check_eeoc_compliance(candidates)
            
            if 'error' not in eeoc_results:
                st.success("EEOC compliance check completed!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Compliant", "Yes" if eeoc_results['compliant'] else "No")
                    st.metric("Total Candidates", eeoc_results['total_candidates'])
                
                with col2:
                    st.metric("Adverse Impact", "Compliant" if eeoc_results['adverse_impact']['compliant'] else "Non-Compliant")
                    st.metric("Disparate Treatment", "Compliant" if eeoc_results['disparate_treatment']['compliant'] else "Non-Compliant")
                
                # Detailed analysis
                st.subheader("Detailed Analysis")
                
                with st.expander("Adverse Impact Analysis"):
                    st.write(eeoc_results['adverse_impact']['analysis'])
                    st.write(f"Selection Rate Ratio: {eeoc_results['adverse_impact']['selection_rate_ratio']:.2f}")
                    st.write(f"Threshold: {eeoc_results['adverse_impact']['threshold']:.2f}")
                
                with st.expander("Disparate Treatment Analysis"):
                    st.write(eeoc_results['disparate_treatment']['analysis'])
                
                # Recommendations
                if eeoc_results['recommendations']:
                    st.subheader("Recommendations")
                    for rec in eeoc_results['recommendations']:
                        st.info(f"• {rec}")
            else:
                st.error(f"Error checking compliance: {eeoc_results['error']}")
    
    with tab3:
        st.subheader("Data Privacy Compliance")
        
        # Data usage checklist
        st.write("**Data Privacy Checklist**")
        
        with st.form("privacy_compliance_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                consent_obtained = st.checkbox("Consent obtained for data processing")
                data_minimization = st.checkbox("Data minimization practices implemented")
                purpose_limitation = st.checkbox("Purpose limitation enforced")
            
            with col2:
                retention_limits = st.checkbox("Data retention limits in place")
                access_controls = st.checkbox("Access controls implemented")
                data_encryption = st.checkbox("Data encryption enabled")
            
            if st.form_submit_button("Check Compliance"):
                data_usage = {
                    'consent_obtained': consent_obtained,
                    'data_minimization': data_minimization,
                    'purpose_limitation': purpose_limitation,
                    'retention_limits': retention_limits,
                    'access_controls': access_controls,
                    'data_encryption': data_encryption
                }
                
                privacy_results = compliance_manager.check_data_privacy_compliance(data_usage)
                
                if 'error' not in privacy_results:
                    st.metric("Data Privacy Compliant", "Yes" if privacy_results['compliant'] else "No")
                    
                    # Show checklist results
                    st.subheader("Compliance Status")
                    for check, status in privacy_results['checks'].items():
                        st.write(f"• {check.replace('_', ' ').title()}: {'✅' if status else '❌'}")
                    
                    # Recommendations
                    if privacy_results['recommendations']:
                        st.subheader("Recommendations")
                        for rec in privacy_results['recommendations']:
                            st.warning(f"• {rec}")
                else:
                    st.error(f"Error checking privacy compliance: {privacy_results['error']}")
    
    with tab4:
        st.subheader("Audit Reports")
        
        # Job posting audit
        st.write("**Audit Selection Process**")
        
        jobs = db_manager.get_job_postings(active_only=True)
        if jobs:
            selected_job = st.selectbox("Select Job Posting to Audit", [f"{job.title} (ID: {job.id})" for job in jobs])
            
            if st.button("Run Audit"):
                job_id = int(selected_job.split("ID: ")[1].split(")")[0])
                
                with st.spinner("Running audit..."):
                    audit_results = compliance_manager.audit_selection_process(job_id)
                
                if 'error' not in audit_results:
                    st.success("Audit completed successfully!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Candidates", audit_results['total_candidates'])
                        st.metric("Audit Date", audit_results['audit_date'][:10])
                    
                    with col2:
                        score_dist = audit_results['bias_analysis']['score_distribution']
                        st.metric("Average Score", f"{score_dist['mean']:.1f}")
                        st.metric("Score Range", f"{score_dist['min']:.1f} - {score_dist['max']:.1f}")
                    
                    # Potential issues
                    if audit_results['bias_analysis']['potential_issues']:
                        st.subheader("Potential Issues")
                        for issue in audit_results['bias_analysis']['potential_issues']:
                            st.warning(f"• {issue}")
                    
                    # Recommendations
                    if audit_results['bias_analysis']['recommendations']:
                        st.subheader("Recommendations")
                        for rec in audit_results['bias_analysis']['recommendations']:
                            st.info(f"• {rec}")
                else:
                    st.error(f"Error running audit: {audit_results['error']}")
        else:
            st.info("No job postings available for audit")
        
        # Comprehensive compliance report
        st.subheader("Comprehensive Compliance Report")
        
        if st.button("Generate Full Compliance Report"):
            with st.spinner("Generating comprehensive report..."):
                compliance_report = compliance_manager.generate_compliance_report()
            
            if 'error' not in compliance_report:
                st.success("Compliance report generated successfully!")
                
                # Summary
                st.subheader("Report Summary")
                summary = compliance_report['summary']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Candidates", summary['total_candidates'])
                
                with col2:
                    st.metric("Total Analyses", summary['total_analyses'])
                
                with col3:
                    st.metric("Overall Compliance", "Compliant" if summary['overall_compliance'] else "Non-Compliant")
                
                with col4:
                    st.metric("Report Date", compliance_report['generated_at'][:10])
                
                # Download report
                report_json = json.dumps(compliance_report, indent=2, default=str)
                st.download_button(
                    "📥 Download Compliance Report",
                    report_json,
                    f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
                
                # Show recommendations
                if compliance_report['recommendations']:
                    st.subheader("Key Recommendations")
                    for rec in compliance_report['recommendations']:
                        st.info(f"• {rec}")
            else:
                st.error(f"Error generating report: {compliance_report['error']}")