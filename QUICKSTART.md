# 🚀 Quick Start Guide

Get the Advanced Resume Ranker up and running in 5 minutes!

## ⚡ Quick Setup

### 1. **Install and Test** (2 minutes)
```bash
# Run the installation script
python install_and_test.py

# Or use the startup script
./start.sh
```

### 2. **Set Your OpenAI API Key** (1 minute)
```bash
# Edit the .env file
nano .env

# Replace this line:
OPENAI_API_KEY=sk-your-openai-api-key-here

# With your actual API key:
OPENAI_API_KEY=sk-proj-your-actual-api-key
```

### 3. **Start the Application** (1 minute)
```bash
# Start the web application
streamlit run app.py

# The app will open at: http://localhost:8501
```

### 4. **First Login** (30 seconds)
- **Username**: `admin`
- **Password**: `admin123`
- **⚠️ Change this password immediately after login!**

### 5. **Analyze Your First Batch** (30 seconds)
1. Go to "🔍 Resume Analysis"
2. Choose a job template or create custom requirements
3. Upload a ZIP file with PDF resumes
4. Click "🚀 Start Analysis"
5. View results in the dashboard!

## 🎯 What You Get

### ✨ **Instant Features**
- **Multi-Agent AI Analysis**: 5 specialized AI agents analyze each resume
- **Advanced Scoring**: Weighted scoring across 5 key dimensions
- **Business Dashboard**: Executive KPIs and hiring funnel metrics
- **Export Options**: CSV, Excel, and JSON exports
- **Analytics**: Skills demand, performance trends, ROI analysis

### 🤖 **AI Agents Working for You**
1. **Resume Parser**: Extracts structured data from PDFs
2. **Skills Matcher**: Matches candidate skills with job requirements
3. **Experience Analyzer**: Evaluates career progression and experience quality
4. **Culture Fit Assessor**: Analyzes soft skills and cultural alignment
5. **Interview Generator**: Creates personalized interview questions

### 📊 **Business Intelligence**
- **Hiring Funnel Analysis**: Track candidates through evaluation stages
- **ROI Calculations**: Measure time and cost savings
- **Skills Market Insights**: Understand talent supply and demand
- **Performance Benchmarking**: Compare across roles and time periods

## 🎮 Try These Examples

### Example 1: Software Engineer Position
1. Select "Software Engineer" template
2. Upload resumes for developer positions
3. See how the system ranks candidates based on:
   - Python/JavaScript skills
   - Experience with frameworks
   - System design capabilities
   - Team collaboration indicators

### Example 2: Data Scientist Role
1. Select "Data Scientist" template
2. Upload data science resumes
3. Analysis focuses on:
   - ML/AI framework experience
   - Statistical analysis skills
   - Research and publication history
   - Data visualization capabilities

### Example 3: Custom Role
1. Create custom job requirements
2. Adjust scoring weights (e.g., 40% skills, 30% experience)
3. Upload relevant resumes
4. Get tailored analysis for your specific needs

## 📈 Advanced Features

### **Business Analytics**
- Navigate to "📈 Analytics" for insights
- View hiring funnel performance
- Analyze skills market trends
- Track ROI and efficiency metrics

### **Export and Integration**
- Export results in multiple formats
- Use the REST API for system integration
- Set up automated notifications
- Generate comprehensive hiring reports

### **User Management**
- Add team members with appropriate roles
- Set permissions for different user types
- Track user activity and audit trails

## 🔧 Customization

### **Scoring Weights**
Adjust the importance of different criteria:
- **Skills Match**: 30% (default)
- **Experience**: 25% (default)
- **Education**: 15% (default)
- **Culture Fit**: 15% (default)
- **Career Progression**: 15% (default)

### **Job Templates**
Pre-configured templates available:
- Software Engineer
- Data Scientist
- DevOps Engineer
- Product Manager
- Frontend Developer

### **Custom Criteria**
Create your own evaluation criteria:
- Define required vs. preferred skills
- Set minimum experience requirements
- Specify education requirements
- Configure cultural fit factors

## 🆘 Need Help?

### **Common Issues**
1. **"No text extracted from PDF"**
   - Ensure PDFs are text-based, not scanned images
   - Try different PDF files

2. **"OpenAI API error"**
   - Check your API key is correct
   - Verify you have API credits available

3. **"Database error"**
   - Ensure the `data/` directory exists
   - Check file permissions

### **Getting Support**
- Check the `📚 Help` section in the app
- Review `DEPLOYMENT.md` for detailed setup
- Create an issue in the repository

## 🎉 You're Ready!

The Advanced Resume Ranker is now ready to transform your hiring process with:

- **80% time savings** on initial resume screening
- **Consistent evaluation** across all candidates
- **Data-driven insights** for better hiring decisions
- **Comprehensive analytics** for strategic planning

Start analyzing resumes and see the difference AI-powered hiring can make! 🚀