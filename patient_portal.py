#!/usr/bin/env python3
"""
Patient Portal Module
====================

Secure patient-facing interface for cancer analysis platform
including results viewing, appointment scheduling, and communication.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import hashlib
import sqlite3
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import json
import time
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatientAuthentication:
    """Secure patient authentication system"""
    
    def __init__(self, db_path: str = "patient_portal.db"):
        self.db_path = db_path
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize patient authentication database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                mrn TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                date_of_birth DATE NOT NULL,
                phone TEXT,
                emergency_contact TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patient_sessions (
                session_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_patient(self, mrn: str, password: str) -> Optional[Dict]:
        """Verify patient credentials"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = self.hash_password(password)
        cursor.execute('''
            SELECT patient_id, mrn, first_name, last_name, email
            FROM patients 
            WHERE mrn = ? AND password_hash = ? AND is_active = 1
        ''', (mrn, password_hash))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'patient_id': result[0],
                'mrn': result[1],
                'first_name': result[2],
                'last_name': result[3],
                'email': result[4]
            }
        return None

class PatientDataManager:
    """Manage patient clinical data and results"""
    
    def __init__(self, db_path: str = "clinical_data.db"):
        self.db_path = db_path
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize clinical data database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_results (
                result_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                test_type TEXT NOT NULL,
                test_date DATE NOT NULL,
                results_json TEXT NOT NULL,
                status TEXT DEFAULT 'completed',
                physician_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS appointments (
                appointment_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                physician_id TEXT NOT NULL,
                appointment_date DATETIME NOT NULL,
                appointment_type TEXT NOT NULL,
                status TEXT DEFAULT 'scheduled',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_patient_results(self, patient_id: str) -> List[Dict]:
        """Get all test results for a patient"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT result_id, test_type, test_date, results_json, status, physician_notes
            FROM test_results 
            WHERE patient_id = ? 
            ORDER BY test_date DESC
        ''', (patient_id,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'result_id': row[0],
                'test_type': row[1],
                'test_date': row[2],
                'results': json.loads(row[3]),
                'status': row[4],
                'physician_notes': row[5]
            })
        
        conn.close()
        return results

class PatientPortalInterface:
    """Main patient portal interface"""
    
    def __init__(self):
        self.auth = PatientAuthentication()
        self.data_manager = PatientDataManager()
    
    def run(self):
        """Run the patient portal application"""
        st.set_page_config(
            page_title="Patient Portal - Cancer Analysis Platform",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #2c3e50, #3498db);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .result-card {
            border: 1px solid #ddd;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .status-normal { color: #27ae60; font-weight: bold; }
        .status-abnormal { color: #e74c3c; font-weight: bold; }
        .status-pending { color: #f39c12; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)
        
        # Check authentication
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        
        if not st.session_state.authenticated:
            self.show_login_page()
        else:
            self.show_dashboard()
    
    def show_login_page(self):
        """Display patient login page"""
        st.markdown("""
        <div class="main-header">
            <h1 style="color: white; text-align: center; margin: 0;">
                🏥 Patient Portal
            </h1>
            <p style="color: #ecf0f1; text-align: center; margin: 0;">
                Secure access to your cancer analysis results
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### Sign In")
            
            with st.form("login_form"):
                mrn = st.text_input("Medical Record Number (MRN)", placeholder="Enter your MRN")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                submitted = st.form_submit_button("Sign In", use_container_width=True)
                
                if submitted:
                    if mrn and password:
                        patient = self.auth.verify_patient(mrn, password)
                        if patient:
                            st.session_state.authenticated = True
                            st.session_state.patient = patient
                            st.rerun()
                        else:
                            st.error("Invalid credentials. Please check your MRN and password.")
                    else:
                        st.error("Please enter both MRN and password.")
            
            st.markdown("---")
            st.markdown("""
            **Need help accessing your account?**
            - Contact Patient Services: (555) 123-4567
            - Email: support@canceranalysis.com
            - Forgot password: Contact your care team
            """)
    
    def show_dashboard(self):
        """Display patient dashboard"""
        patient = st.session_state.patient
        
        # Header
        st.markdown(f"""
        <div class="main-header">
            <h1 style="color: white; margin: 0;">
                Welcome, {patient['first_name']} {patient['last_name']}
            </h1>
            <p style="color: #ecf0f1; margin: 0;">
                MRN: {patient['mrn']} | Last Updated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar navigation
        with st.sidebar:
            st.markdown("### Navigation")
            page = st.selectbox("Select a page:", [
                "🏠 Dashboard",
                "📊 Test Results",
                "📅 Appointments", 
                "💬 Messages",
                "📋 Health Summary",
                "⚙️ Settings"
            ])
            
            st.markdown("---")
            if st.button("🚪 Sign Out"):
                st.session_state.authenticated = False
                st.rerun()
        
        # Main content based on selected page
        if page == "🏠 Dashboard":
            self.show_dashboard_overview()
        elif page == "📊 Test Results":
            self.show_test_results()
        elif page == "📅 Appointments":
            self.show_appointments()
        elif page == "💬 Messages":
            self.show_messages()
        elif page == "📋 Health Summary":
            self.show_health_summary()
        elif page == "⚙️ Settings":
            self.show_settings()
    
    def show_dashboard_overview(self):
        """Display dashboard overview"""
        patient_id = st.session_state.patient['patient_id']
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>3</h3>
                <p>Recent Tests</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>1</h3>
                <p>Upcoming Appointments</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>2</h3>
                <p>New Messages</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>Stable</h3>
                <p>Current Status</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent activity
        st.markdown("### Recent Activity")
        
        # Simulated recent activity data
        activity_data = [
            {"date": "2024-01-15", "activity": "Blood test results available", "status": "new"},
            {"date": "2024-01-14", "activity": "Appointment scheduled with Dr. Smith", "status": "info"},
            {"date": "2024-01-12", "activity": "MRI scan completed", "status": "completed"},
            {"date": "2024-01-10", "activity": "Treatment plan updated", "status": "info"}
        ]
        
        for activity in activity_data:
            status_class = "status-normal" if activity["status"] == "completed" else "status-pending"
            st.markdown(f"""
            <div class="result-card">
                <strong>{activity['date']}</strong> - 
                <span class="{status_class}">{activity['activity']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    def show_test_results(self):
        """Display test results page"""
        st.markdown("### 📊 Test Results")
        
        # Generate sample test results
        sample_results = self.generate_sample_results()
        
        for result in sample_results:
            with st.expander(f"{result['test_type']} - {result['test_date']}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Status:** {result['status']}")
                    st.markdown(f"**Test Date:** {result['test_date']}")
                    
                    if result['test_type'] == 'WSI Analysis':
                        self.display_wsi_results(result['results'])
                    elif result['test_type'] == 'NGS Sequencing':
                        self.display_ngs_results(result['results'])
                    elif result['test_type'] == 'MRD Detection':
                        self.display_mrd_results(result['results'])
                
                with col2:
                    if result.get('physician_notes'):
                        st.markdown("**Physician Notes:**")
                        st.markdown(result['physician_notes'])
                    
                    st.download_button(
                        "📄 Download Report",
                        data=json.dumps(result, indent=2),
                        file_name=f"{result['test_type']}_{result['test_date']}.json",
                        mime="application/json"
                    )
    
    def display_wsi_results(self, results: Dict):
        """Display WSI analysis results"""
        st.markdown("**Whole Slide Imaging Analysis**")
        
        # Create visualization
        fig = go.Figure()
        
        # Sample tumor regions data
        regions = results.get('tumor_regions', [])
        if regions:
            for i, region in enumerate(regions):
                fig.add_trace(go.Scatter(
                    x=[region['x']],
                    y=[region['y']],
                    mode='markers',
                    marker=dict(
                        size=region['confidence'] * 20,
                        color=region['malignancy_score'],
                        colorscale='Reds',
                        showscale=True
                    ),
                    name=f'Region {i+1}'
                ))
        
        fig.update_layout(
            title="Tumor Region Analysis",
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tumor Area", f"{results.get('tumor_area', 0):.1f} mm²")
        with col2:
            st.metric("Confidence", f"{results.get('confidence', 0):.1%}")
        with col3:
            st.metric("Grade", results.get('grade', 'N/A'))
    
    def display_ngs_results(self, results: Dict):
        """Display NGS sequencing results"""
        st.markdown("**Next Generation Sequencing Results**")
        
        # Variant summary
        variants = results.get('variants', [])
        if variants:
            df = pd.DataFrame(variants)
            st.dataframe(df, use_container_width=True)
            
            # Variant frequency chart
            if 'allele_frequency' in df.columns:
                fig = px.bar(
                    df, 
                    x='gene', 
                    y='allele_frequency',
                    title="Variant Allele Frequencies",
                    color='pathogenicity'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Quality metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Coverage", f"{results.get('coverage', 0)}x")
        with col2:
            st.metric("Quality Score", f"{results.get('quality_score', 0):.1f}")
        with col3:
            st.metric("Variants Found", len(variants))
    
    def display_mrd_results(self, results: Dict):
        """Display MRD detection results"""
        st.markdown("**Minimal Residual Disease Detection**")
        
        # MRD trend over time
        time_points = results.get('time_series', [])
        if time_points:
            df = pd.DataFrame(time_points)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['mrd_level'],
                mode='lines+markers',
                name='MRD Level',
                line=dict(color='red', width=3)
            ))
            
            # Add threshold line
            fig.add_hline(
                y=0.01, 
                line_dash="dash", 
                line_color="orange",
                annotation_text="Detection Threshold"
            )
            
            fig.update_layout(
                title="MRD Levels Over Time",
                xaxis_title="Date",
                yaxis_title="MRD Level (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Current status
        current_level = results.get('current_level', 0)
        if current_level < 0.01:
            st.success(f"✅ Below detection threshold: {current_level:.4f}%")
        else:
            st.warning(f"⚠️ Above detection threshold: {current_level:.4f}%")
    
    def show_appointments(self):
        """Display appointments page"""
        st.markdown("### 📅 Appointments")
        
        tab1, tab2 = st.tabs(["Upcoming", "Schedule New"])
        
        with tab1:
            # Sample upcoming appointments
            appointments = [
                {
                    "date": "2024-01-20",
                    "time": "10:00 AM",
                    "physician": "Dr. Sarah Smith",
                    "type": "Follow-up Consultation",
                    "location": "Oncology Clinic - Room 205"
                },
                {
                    "date": "2024-01-25", 
                    "time": "2:30 PM",
                    "physician": "Dr. Michael Johnson",
                    "type": "Blood Draw",
                    "location": "Laboratory - 1st Floor"
                }
            ]
            
            for apt in appointments:
                st.markdown(f"""
                <div class="result-card">
                    <h4>{apt['type']}</h4>
                    <p><strong>Date:</strong> {apt['date']} at {apt['time']}</p>
                    <p><strong>Provider:</strong> {apt['physician']}</p>
                    <p><strong>Location:</strong> {apt['location']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("**Request New Appointment**")
            
            with st.form("appointment_form"):
                appointment_type = st.selectbox("Appointment Type", [
                    "Consultation",
                    "Follow-up",
                    "Lab Work",
                    "Imaging",
                    "Treatment"
                ])
                
                preferred_date = st.date_input("Preferred Date")
                preferred_time = st.selectbox("Preferred Time", [
                    "Morning (8:00 AM - 12:00 PM)",
                    "Afternoon (12:00 PM - 5:00 PM)"
                ])
                
                physician = st.selectbox("Preferred Physician", [
                    "Dr. Sarah Smith - Oncologist",
                    "Dr. Michael Johnson - Radiologist", 
                    "Dr. Emily Chen - Pathologist"
                ])
                
                notes = st.text_area("Additional Notes", placeholder="Any specific concerns or requests...")
                
                submitted = st.form_submit_button("Submit Request")
                
                if submitted:
                    st.success("Appointment request submitted! You will receive confirmation within 24 hours.")
    
    def show_messages(self):
        """Display messages page"""
        st.markdown("### 💬 Messages")
        
        # Sample messages
        messages = [
            {
                "from": "Dr. Sarah Smith",
                "subject": "Test Results Available",
                "date": "2024-01-15",
                "preview": "Your recent blood work results are now available...",
                "unread": True
            },
            {
                "from": "Scheduling Department", 
                "subject": "Appointment Confirmation",
                "date": "2024-01-14",
                "preview": "Your appointment on January 20th has been confirmed...",
                "unread": False
            }
        ]
        
        for msg in messages:
            status_indicator = "🔴" if msg["unread"] else "⚪"
            
            with st.expander(f"{status_indicator} {msg['subject']} - {msg['from']} ({msg['date']})"):
                st.markdown(f"**From:** {msg['from']}")
                st.markdown(f"**Date:** {msg['date']}")
                st.markdown(f"**Message:** {msg['preview']}")
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("Reply", key=f"reply_{msg['date']}"):
                        st.info("Reply functionality would open here")
    
    def show_health_summary(self):
        """Display health summary page"""
        st.markdown("### 📋 Health Summary")
        
        # Treatment timeline
        st.markdown("#### Treatment Timeline")
        
        timeline_data = [
            {"date": "2023-12-01", "event": "Initial Diagnosis", "type": "diagnosis"},
            {"date": "2023-12-15", "event": "Treatment Plan Created", "type": "planning"},
            {"date": "2024-01-05", "event": "First Treatment Cycle", "type": "treatment"},
            {"date": "2024-01-15", "event": "Follow-up Imaging", "type": "monitoring"}
        ]
        
        for event in timeline_data:
            color = {
                "diagnosis": "🔴",
                "planning": "🟡", 
                "treatment": "🟢",
                "monitoring": "🔵"
            }.get(event["type"], "⚪")
            
            st.markdown(f"{color} **{event['date']}** - {event['event']}")
        
        # Key metrics over time
        st.markdown("#### Key Health Metrics")
        
        # Sample trend data
        dates = pd.date_range(start='2023-12-01', end='2024-01-15', freq='W')
        tumor_markers = np.random.normal(50, 10, len(dates))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=tumor_markers,
            mode='lines+markers',
            name='Tumor Marker Levels',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="Tumor Marker Trends",
            xaxis_title="Date",
            yaxis_title="Marker Level",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_settings(self):
        """Display settings page"""
        st.markdown("### ⚙️ Settings")
        
        tab1, tab2, tab3 = st.tabs(["Profile", "Notifications", "Privacy"])
        
        with tab1:
            st.markdown("#### Profile Information")
            patient = st.session_state.patient
            
            with st.form("profile_form"):
                first_name = st.text_input("First Name", value=patient['first_name'])
                last_name = st.text_input("Last Name", value=patient['last_name'])
                email = st.text_input("Email", value=patient['email'])
                phone = st.text_input("Phone Number", placeholder="(555) 123-4567")
                
                if st.form_submit_button("Update Profile"):
                    st.success("Profile updated successfully!")
        
        with tab2:
            st.markdown("#### Notification Preferences")
            
            email_notifications = st.checkbox("Email notifications", value=True)
            sms_notifications = st.checkbox("SMS notifications", value=False)
            test_results = st.checkbox("Test result alerts", value=True)
            appointment_reminders = st.checkbox("Appointment reminders", value=True)
            
            if st.button("Save Notification Settings"):
                st.success("Notification preferences saved!")
        
        with tab3:
            st.markdown("#### Privacy & Security")
            
            st.markdown("**Data Sharing Preferences**")
            research_participation = st.checkbox("Participate in anonymized research", value=False)
            
            st.markdown("**Account Security**")
            if st.button("Change Password"):
                st.info("Password change functionality would be implemented here")
            
            if st.button("Download My Data"):
                st.info("Data export functionality would be implemented here")
    
    def generate_sample_results(self) -> List[Dict]:
        """Generate sample test results for demonstration"""
        return [
            {
                "result_id": "WSI_001",
                "test_type": "WSI Analysis",
                "test_date": "2024-01-15",
                "status": "Complete",
                "results": {
                    "tumor_area": 12.5,
                    "confidence": 0.94,
                    "grade": "II",
                    "tumor_regions": [
                        {"x": 100, "y": 150, "confidence": 0.9, "malignancy_score": 0.8},
                        {"x": 200, "y": 250, "confidence": 0.85, "malignancy_score": 0.7}
                    ]
                },
                "physician_notes": "Moderate tumor burden detected. Recommend follow-up in 3 months."
            },
            {
                "result_id": "NGS_001", 
                "test_type": "NGS Sequencing",
                "test_date": "2024-01-12",
                "status": "Complete",
                "results": {
                    "coverage": 150,
                    "quality_score": 98.5,
                    "variants": [
                        {"gene": "TP53", "variant": "c.542T>G", "allele_frequency": 0.45, "pathogenicity": "Pathogenic"},
                        {"gene": "EGFR", "variant": "c.2573T>G", "allele_frequency": 0.32, "pathogenicity": "Likely Pathogenic"}
                    ]
                },
                "physician_notes": "Significant mutations detected. Treatment plan to be adjusted accordingly."
            },
            {
                "result_id": "MRD_001",
                "test_type": "MRD Detection", 
                "test_date": "2024-01-10",
                "status": "Complete",
                "results": {
                    "current_level": 0.005,
                    "time_series": [
                        {"date": "2023-12-01", "mrd_level": 0.15},
                        {"date": "2023-12-15", "mrd_level": 0.08},
                        {"date": "2024-01-01", "mrd_level": 0.02},
                        {"date": "2024-01-10", "mrd_level": 0.005}
                    ]
                },
                "physician_notes": "Excellent response to treatment. MRD levels below detection threshold."
            }
        ]

def main():
    """Main application entry point"""
    portal = PatientPortalInterface()
    portal.run()

if __name__ == "__main__":
    main()
