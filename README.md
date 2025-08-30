# 🎯 Advanced Resume Ranker

An enterprise-grade resume ranking system powered by multiple AI agents for comprehensive candidate evaluation.

## 🚀 Features

### 🤖 Multi-Agent AI System
- **Resume Parser Agent**: Extracts structured data from PDF resumes
- **Skills Matcher Agent**: Matches candidate skills with job requirements
- **Experience Analyzer Agent**: Evaluates work experience and career progression
- **Culture Fit Agent**: Assesses soft skills and cultural alignment

### 📊 Advanced Analytics
- Comprehensive scoring with weighted criteria
- Real-time analytics dashboard
- Score distribution visualizations
- Skills gap analysis
- Hiring recommendations

### 💼 Business Features
- Job requirement templates for common roles
- Candidate database with persistence
- Export to CSV, Excel, and JSON formats
- Hiring reports and analytics
- User management and authentication
- Session tracking and audit logs

### 🎨 Modern UI
- Multi-page dashboard interface
- Interactive visualizations with Plotly
- Responsive design
- Real-time progress tracking
- Detailed candidate profiles

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- OpenAI API key
- 2GB+ RAM recommended

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd advanced-resume-ranker
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the application**
   - Open http://localhost:8501 in your browser
   - Default admin login: `admin` / `admin123` (change immediately)

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

2. **Access the application**
   - Open http://localhost:8501 in your browser

### Production Deployment

1. **Set environment variables**
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export APP_ENV="production"
   ```

2. **Use production profile**
   ```bash
   docker-compose --profile production up -d
   ```

## 📖 Usage Guide

### 1. Job Requirements Setup
- Choose from pre-built job templates or create custom requirements
- Define required and preferred skills
- Set minimum experience requirements
- Configure scoring weights based on role priorities

### 2. Resume Upload
- Upload ZIP files containing PDF resumes
- System supports batch processing of up to 100 resumes
- Advanced PDF text extraction handles complex layouts

### 3. AI Analysis
- Multiple AI agents analyze each resume simultaneously
- Comprehensive scoring across 5 dimensions
- Detailed analysis with recommendations and red flags

### 4. Results Review
- Interactive dashboard with rankings and visualizations
- Detailed candidate profiles with contact information
- Export options for further processing
- Hiring recommendations based on scores

## 🏗️ Architecture

```
├── src/
│   ├── agents/           # AI agents for specialized analysis
│   ├── services/         # Business logic and data services
│   ├── models/          # Data models and structures
│   ├── ui/              # User interface components
│   └── utils/           # Utility functions and helpers
├── config/              # Configuration and settings
├── data/               # Database and data files
├── static/             # Static assets (CSS, JS)
└── logs/               # Application logs
```

### AI Agents

1. **ResumeParserAgent**: Converts unstructured resume text into structured data
2. **SkillsMatcherAgent**: Matches and scores technical and soft skills
3. **ExperienceAnalyzerAgent**: Evaluates career progression and experience quality
4. **CultureFitAgent**: Assesses cultural alignment and soft skills
5. **RankingOrchestrator**: Coordinates all agents and compiles final scores

### Scoring System

The system uses a weighted scoring approach:
- **Skills Match (30%)**: Technical skills alignment
- **Experience Relevance (25%)**: Work experience quality and relevance
- **Education Match (15%)**: Educational background fit
- **Culture Fit (15%)**: Soft skills and cultural alignment
- **Career Progression (15%)**: Career growth and advancement

## 🔧 Configuration

### Scoring Weights
Customize scoring weights in the UI or via configuration:

```python
weights = {
    'skills_match': 0.30,
    'experience_relevance': 0.25,
    'education_match': 0.15,
    'culture_fit': 0.15,
    'career_progression': 0.15
}
```

### Job Templates
Pre-configured templates for common roles:
- Software Engineer
- Data Scientist
- DevOps Engineer
- Product Manager
- Frontend Developer

## 📊 Analytics Features

- **Candidate Metrics**: Total candidates, average scores, qualification rates
- **Score Distribution**: Visual breakdown of candidate score ranges
- **Skills Analysis**: Most common skills across candidate pool
- **Hiring Insights**: Recommendations for interview prioritization
- **Trend Analysis**: Performance trends over time

## 🔒 Security Features

- User authentication and role-based access
- Session management with expiration
- Permission-based feature access
- Audit logging for compliance
- Secure password hashing

## 🚀 API Integration

The system is designed for easy integration with:
- Applicant Tracking Systems (ATS)
- HR Information Systems (HRIS)
- Job boards and career sites
- Interview scheduling platforms

## 📈 Performance

- **Parallel Processing**: Multiple agents work simultaneously
- **Efficient PDF Processing**: Optimized text extraction
- **Database Optimization**: Indexed queries for fast retrieval
- **Caching**: Resume analysis results cached for quick access

## 🛡️ Compliance

- **Data Privacy**: Local data storage with optional encryption
- **GDPR Compliance**: Data export and deletion capabilities
- **Audit Trails**: Complete analysis history tracking
- **Access Controls**: Role-based permissions

## 🔄 Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
flake8 src/
```

### Adding New Agents
1. Inherit from `BaseAgent`
2. Implement `process()` method
3. Add to `RankingOrchestrator`
4. Update scoring weights

## 📞 Support

For technical support or feature requests, please create an issue in the repository.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.