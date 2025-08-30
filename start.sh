#!/bin/bash

# Advanced Resume Ranker Startup Script

echo "🎯 Starting Advanced Resume Ranker..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs static/css static/js

# Check for environment file
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment file..."
    cp .env.example .env
    echo "❗ Please edit .env with your OpenAI API key before running the application"
fi

# Check for OpenAI API key
if grep -q "sk-your-openai-api-key-here" .env 2>/dev/null; then
    echo "❌ Please set your OpenAI API key in the .env file"
    echo "📝 Edit .env and replace 'sk-your-openai-api-key-here' with your actual API key"
    exit 1
fi

# Initialize database
echo "🗄️ Initializing database..."
python -c "
import sys
sys.path.append('src')
from services.database_service import DatabaseService
db = DatabaseService()
print('Database initialized successfully')
"

echo "✅ Setup complete!"
echo ""
echo "🚀 Starting application..."
echo "📍 Application will be available at: http://localhost:8501"
echo "🔐 Default admin login: admin / admin123 (change immediately)"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Start the application
streamlit run app.py --server.port=8501 --server.address=0.0.0.0