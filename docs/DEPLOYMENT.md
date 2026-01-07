# Deployment Guide

## Overview

This guide covers deploying the Deepfake Detection System in production environments.

## Prerequisites

- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+
- Python 3.11+
- Node.js 18+ (for tooling)
- Android NDK (for mobile builds)
- Xcode (for iOS builds)

## Cloud Deployment

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd DeepFake

# Copy environment file
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### 2. Database Setup

```bash
# Start PostgreSQL
docker-compose up -d postgres

# Wait for database to be ready
docker-compose logs -f postgres

# Run migrations
python cloud/database/migrate.py
```

### 3. Start Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

### 4. Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs
```

## Edge Agent Deployment

### Android

```bash
# Set Android NDK path
export ANDROID_NDK_HOME=/path/to/ndk

# Build
./scripts/build_android.sh

# Install on device
adb install build/android/deepfake-detector.apk
```

### iOS

```bash
# Build
./scripts/build_ios.sh

# Open in Xcode
open edge_agent/mobile/ios/DeepFakeDetector.xcworkspace

# Build and run from Xcode
```

## Production Configuration

### Security

1. **Change Default Passwords**
   ```bash
   # Generate secure passwords
   openssl rand -base64 32
   ```

2. **Enable HTTPS**
   - Configure SSL certificates
   - Update CORS origins
   - Enable secure cookies

3. **Configure MFA**
   ```env
   MFA_ENABLED=true
   MFA_ISSUER=YourOrganization
   ```

### Scaling

1. **Horizontal Scaling**
   ```yaml
   # docker-compose.yml
   api:
     deploy:
       replicas: 3
   ```

2. **Database Replication**
   - Configure read replicas
   - Enable connection pooling

3. **Caching**
   - Configure Redis cluster
   - Enable result caching

### Monitoring

1. **Prometheus**
   - Access: http://localhost:9090
   - Configure alerts

2. **Grafana**
   - Access: http://localhost:3000
   - Import dashboards

3. **Logging**
   ```bash
   # View logs
   docker-compose logs -f

   # Export logs
   docker-compose logs > logs.txt
   ```

## Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepfake-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: deepfake-api
  template:
    metadata:
      labels:
        app: deepfake-api
    spec:
      containers:
      - name: api
        image: deepfake-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: deepfake-secrets
              key: database-url
```

## Backup & Recovery

### Database Backup

```bash
# Backup
docker-compose exec postgres pg_dump -U deepfake_user deepfake_db > backup.sql

# Restore
docker-compose exec -T postgres psql -U deepfake_user deepfake_db < backup.sql
```

### Model Backup

```bash
# Backup models
tar -czf models-backup.tar.gz /app/model_storage

# Restore
tar -xzf models-backup.tar.gz -C /app/model_storage
```

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check DATABASE_URL
   - Verify PostgreSQL is running
   - Check network connectivity

2. **Model Loading Failed**
   - Verify model files exist
   - Check file permissions
   - Validate model format

3. **High Memory Usage**
   - Reduce batch size
   - Enable model quantization
   - Configure memory limits

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Restart services
docker-compose restart api
```

## Performance Tuning

### API Optimization

```env
# Increase workers
API_WORKERS=8

# Enable caching
REDIS_URL=redis://redis:6379/0
```

### Database Optimization

```sql
-- Create indexes
CREATE INDEX idx_detections_user_created ON detections(user_id, created_at DESC);

-- Analyze tables
ANALYZE detections;
```

## Security Hardening

1. **Firewall Rules**
   ```bash
   # Allow only necessary ports
   ufw allow 8000/tcp
   ufw allow 5432/tcp
   ufw enable
   ```

2. **Rate Limiting**
   ```env
   RATE_LIMIT_PER_MINUTE=60
   RATE_LIMIT_PER_HOUR=1000
   ```

3. **Audit Logging**
   ```env
   AUDIT_ENABLED=true
   AUDIT_RETENTION_DAYS=90
   ```

## Maintenance

### Regular Tasks

1. **Update Models**
   ```bash
   # Upload new model
   curl -X POST http://localhost:8000/api/v1/models/upload \
     -H "Authorization: Bearer $TOKEN" \
     -F "file=@model.tflite"
   ```

2. **Clean Old Data**
   ```sql
   DELETE FROM detections WHERE created_at < NOW() - INTERVAL '90 days';
   ```

3. **Monitor Performance**
   - Check API latency
   - Monitor database size
   - Review error logs
