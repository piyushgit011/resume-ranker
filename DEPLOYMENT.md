# 🚀 Deployment Guide

This guide covers various deployment options for the Advanced Resume Ranker system.

## 📋 Prerequisites

- Python 3.11+
- OpenAI API key
- 2GB+ RAM
- 10GB+ storage (for database and file processing)

## 🏠 Local Development

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd advanced-resume-ranker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env
```

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key
- `DATABASE_PATH`: Path to SQLite database
- `MAX_FILE_SIZE_MB`: Maximum upload file size
- `MAX_RESUMES_PER_BATCH`: Maximum resumes per analysis

### 3. Run Application

```bash
# Run Streamlit app
streamlit run app.py

# Or run API server
python run_api.py
```

## 🐳 Docker Deployment

### 1. Basic Docker

```bash
# Build image
docker build -t resume-ranker .

# Run container
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your-api-key \
  -v $(pwd)/data:/app/data \
  resume-ranker
```

### 2. Docker Compose

```bash
# Development
docker-compose up -d

# Production with nginx
docker-compose --profile production up -d
```

## ☁️ Cloud Deployment

### AWS Deployment

#### Option 1: ECS Fargate

1. **Build and push to ECR**
   ```bash
   # Create ECR repository
   aws ecr create-repository --repository-name resume-ranker
   
   # Get login token
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
   
   # Build and tag
   docker build -t resume-ranker .
   docker tag resume-ranker:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/resume-ranker:latest
   
   # Push to ECR
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/resume-ranker:latest
   ```

2. **Create ECS task definition**
   ```json
   {
     "family": "resume-ranker",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "1024",
     "memory": "2048",
     "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "resume-ranker",
         "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/resume-ranker:latest",
         "portMappings": [
           {
             "containerPort": 8501,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "OPENAI_API_KEY",
             "value": "your-api-key"
           }
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/resume-ranker",
             "awslogs-region": "us-east-1",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

3. **Create ECS service with Application Load Balancer**

#### Option 2: EC2 Instance

```bash
# Launch EC2 instance (t3.medium recommended)
# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Clone and run
git clone <repository-url>
cd advanced-resume-ranker
docker-compose up -d
```

### Google Cloud Platform

#### Cloud Run Deployment

```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT-ID/resume-ranker
gcloud run deploy --image gcr.io/PROJECT-ID/resume-ranker --platform managed
```

### Azure Container Instances

```bash
# Create resource group
az group create --name resume-ranker-rg --location eastus

# Deploy container
az container create \
  --resource-group resume-ranker-rg \
  --name resume-ranker \
  --image your-registry/resume-ranker:latest \
  --cpu 2 --memory 4 \
  --ports 8501 \
  --environment-variables OPENAI_API_KEY=your-api-key
```

## 🔧 Production Configuration

### 1. Environment Variables

```bash
# Production settings
export APP_ENV=production
export DEBUG=false
export LOG_LEVEL=WARNING

# Security
export SECRET_KEY=your-secret-key
export SESSION_TIMEOUT_HOURS=8

# Performance
export MAX_WORKERS=4
export CACHE_TTL=3600

# Monitoring
export SENTRY_DSN=your-sentry-dsn
export NEW_RELIC_LICENSE_KEY=your-newrelic-key
```

### 2. Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 3. SSL Configuration

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### 4. Database Backup

```bash
# Create backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
sqlite3 /app/data/resume_ranker.db ".backup /app/backups/backup_$DATE.db"

# Compress backup
gzip /app/backups/backup_$DATE.db

# Upload to S3 (optional)
aws s3 cp /app/backups/backup_$DATE.db.gz s3://your-backup-bucket/
```

## 📊 Monitoring and Logging

### 1. Application Logs

```python
# Configure logging in production
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/app.log'),
        logging.StreamHandler()
    ]
)
```

### 2. Health Checks

```bash
# Health check endpoint
curl http://localhost:8501/_stcore/health

# API health check
curl http://localhost:8000/api/v1/health
```

### 3. Performance Monitoring

```bash
# Monitor resource usage
docker stats resume-ranker

# Monitor application metrics
curl http://localhost:8501/_stcore/metrics
```

## 🔒 Security Considerations

### 1. API Security

- Use HTTPS in production
- Implement rate limiting
- Validate all inputs
- Use environment variables for secrets
- Regular security updates

### 2. Data Protection

- Encrypt sensitive data
- Implement data retention policies
- Regular database backups
- Access logging and audit trails

### 3. Network Security

```bash
# Firewall configuration
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

## 📈 Scaling Considerations

### Horizontal Scaling

- Use load balancer for multiple instances
- Implement session store (Redis)
- Database clustering for high availability
- CDN for static assets

### Performance Optimization

- Enable caching for resume analyses
- Implement background job processing
- Use connection pooling
- Optimize database queries

## 🆘 Troubleshooting

### Common Issues

1. **PDF extraction fails**
   - Ensure PDFs are text-based, not scanned images
   - Check file permissions and disk space

2. **OpenAI API errors**
   - Verify API key is valid
   - Check rate limits and usage quotas
   - Monitor API response times

3. **Database issues**
   - Check disk space for SQLite database
   - Verify database file permissions
   - Regular database maintenance

4. **Memory issues**
   - Increase container memory limits
   - Implement batch processing limits
   - Monitor memory usage patterns

### Logs and Debugging

```bash
# View application logs
docker logs resume-ranker

# Check system resources
htop
df -h

# Database integrity check
sqlite3 /app/data/resume_ranker.db "PRAGMA integrity_check;"
```

## 📞 Support

For deployment assistance:
1. Check the troubleshooting section
2. Review application logs
3. Create an issue in the repository
4. Contact technical support

## 🔄 Updates and Maintenance

### Regular Maintenance Tasks

1. **Weekly**
   - Review application logs
   - Check disk space usage
   - Monitor API usage and costs

2. **Monthly**
   - Update dependencies
   - Database optimization
   - Security patches

3. **Quarterly**
   - Full system backup
   - Performance review
   - Security audit