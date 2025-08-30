#!/usr/bin/env python3
"""
Advanced Resume Ranker Pro - Startup Script
This script provides easy startup options for the application
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import streamlit
        import pandas
        import plotly
        import langchain_openai
        import sqlalchemy
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_api_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "sk-your-api-key-here":
        print("⚠️  OpenAI API key not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False
    print("✅ OpenAI API key is set")
    return True

def run_streamlit():
    """Run the Streamlit application"""
    print("🚀 Starting Advanced Resume Ranker Pro...")
    print("📊 Opening web interface at http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "main_app.py"])

def run_api():
    """Run the FastAPI server"""
    print("🔧 Starting API server...")
    print("📡 API available at http://localhost:8000")
    subprocess.run([sys.executable, "api.py"])

def run_both():
    """Run both Streamlit and API servers"""
    print("🚀 Starting both Streamlit and API servers...")
    print("📊 Web interface: http://localhost:8501")
    print("📡 API server: http://localhost:8000")
    
    # Start API server in background
    api_process = subprocess.Popen([sys.executable, "api.py"])
    
    # Start Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "main_app.py"])
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        api_process.terminate()

def setup_database():
    """Initialize the database"""
    print("🗄️  Setting up database...")
    try:
        from database import db_manager
        # This will create tables if they don't exist
        db_manager.get_session()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database setup failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Advanced Resume Ranker Pro")
    parser.add_argument(
        "mode",
        choices=["web", "api", "both", "setup"],
        help="Run mode: web (Streamlit), api (FastAPI), both, or setup (database)"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip dependency and API key checks"
    )
    
    args = parser.parse_args()
    
    print("🚀 Advanced Resume Ranker Pro v2.0.0")
    print("=" * 50)
    
    if not args.skip_checks:
        if not check_dependencies():
            sys.exit(1)
        
        if args.mode in ["web", "both"] and not check_api_key():
            print("\n⚠️  You can still run the API server without the API key")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    if args.mode == "setup":
        setup_database()
    elif args.mode == "web":
        run_streamlit()
    elif args.mode == "api":
        run_api()
    elif args.mode == "both":
        run_both()

if __name__ == "__main__":
    main()