#!/usr/bin/env python3
"""
Startup script for running the Resume Ranker API server
"""

import uvicorn
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.services.api_service import api_app

if __name__ == "__main__":
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    print(f"🚀 Starting Resume Ranker API server...")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    
    uvicorn.run(
        "src.services.api_service:api_app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )