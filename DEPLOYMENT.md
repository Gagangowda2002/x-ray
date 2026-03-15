# Deployment Guide

Complete step-by-step instructions for deploying the X-Ray Medical Image Classification system to popular cloud platforms.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Deployment to Render](#deployment-to-render)
3. [Deployment to Railway](#deployment-to-railway)
4. [Deployment to Heroku](#deployment-to-heroku)
5. [Local Docker Deployment](#local-docker-deployment)
6. [Post-Deployment Configuration](#post-deployment-configuration)

---

## Prerequisites

All platforms require:
- GitHub account with your repository
- Model file (`model/densenet_model.h5`) > 100MB should be handled carefully
- Application code pushed to GitHub

### Large File Handling

Since the DenseNet model is typically >100MB:

**Option 1: Git LFS (Large File Storage)**
```bash
# Install Git LFS
git lfs install

# Track model file
git lfs track "*.h5"
git add .gitattributes
git commit -m "Add LFS tracking for model files"
git add model/densenet_model.h5
git commit -m "Add model with LFS"
```

**Option 2: Download from Storage Service**
Modify your deployment to download the model from AWS S3, Google Cloud Storage, etc.

**Option 3: Use Build Cache**
Docker caches layers, so the model can be in the repository with the understanding that it will be cached.

---

## Deployment to Render

### Step 1: Prepare Repository

```bash
# Ensure all files are committed
git status
git add .
git commit -m "Ready for Render deployment"
git push origin main
```

### Step 2: Create Render Service

1. Go to https://render.com
2. Sign up/Login with GitHub
3. Click "New +" → "Web Service"
4. Select "Connect a repository"
5. Search and select your x-ray-classification repo
6. Configure the service:
   - **Name**: x-ray-classifier (or your choice)
   - **Environment**: Docker
   - **Build Command**: Default (Render auto-detects)
   - **Start Command**: Default

### Step 3: Add Environment Variables

1. Scroll to "Environment" section
2. Add the following variables:
   ```
   FLASK_ENV=production
   SECRET_KEY=[Generate a random key - minimum 32 characters]
   PYTHONUNBUFFERED=1
   ```
3. To generate SECRET_KEY:
   ```bash
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

### Step 4: Configure Resource Limits

1. Go to "Advanced"
2. Set appropriate resource:
   - Starter: Free tier (limited)
   - Standard: Recommended for this application
   - Pro: For high traffic

### Step 5: Deploy

1. Click "Create Web Service"
2. Render will automatically:
   - Build the Docker image
   - Deploy the container
   - Assign a public URL
3. Monitor progress in the logs

### Step 6: Verify Deployment

```bash
# Test health endpoint
curl https://your-app.onrender.com/health

# Test API info
curl https://your-app.onrender.com/api/info
```

### Important Notes for Render

- **Cold starts**: Free tier instances may have cold starts (5-10 sec)
- **File storage**: Use Render Disks for persistent storage if needed
- **Model persistence**: Model file should be in repository or downloaded on startup
- **GPU**: Available on paid plans only

---

## Deployment to Railway

### Step 1: GitHub Setup

```bash
git push origin main
```

### Step 2: Create Railway Project

1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your x-ray-classification repository
5. Click "Deploy"

### Step 3: Configure Variables

1. In Railway dashboard, go to "Variables"
2. Add:
   ```
   FLASK_ENV=production
   SECRET_KEY=[Generate as in Render instructions]
   PYTHONUNBUFFERED=1
   PYTHON_VERSION=3.11
   ```

### Step 4: Domain Configuration

1. Go to "Settings"
2. Click "Generate Domain"
3. Your app will be available at `https://your-project-name.up.railway.app`

### Step 5: View Logs

1. Click "View Logs" tab
2. Monitor deployment and runtime logs

### Scaling on Railway

Railway has a pay-as-you-go model:
- 1 vCPU & 512MB RAM: Free starter (limited hours)
- Scale up as needed for production traffic

---

## Deployment to Heroku

### Step 1: Install Heroku CLI

**macOS/Linux:**
```bash
brew tap heroku/brew && brew install heroku
```

**Windows:**
```bash
choco install heroku-cli
```

### Step 2: Login to Heroku

```bash
heroku login
# Opens browser for authentication
```

### Step 3: Create Application

```bash
heroku create x-ray-classifier
```

Or use existing app:
```bash
heroku git:remote -a x-ray-classifier
```

### Step 4: Set Configuration

```bash
# Set environment variables
heroku config:set FLASK_ENV=production
heroku config:set SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
heroku config:set PYTHONUNBUFFERED=1

# Verify
heroku config
```

### Step 5: Add Buildpack

```bash
heroku buildpacks:add heroku/python
```

### Step 6: Deploy

```bash
git push heroku main
```

### Step 7: Check Logs

```bash
heroku logs --tail
```

### Step 8: Open Application

```bash
heroku open
```

---

## Local Docker Deployment

For testing deployment locally before pushing to cloud:

### Step 1: Build Image

```bash
docker build -t x-ray-classifier:latest .
```

### Step 2: Run Container

```bash
docker run -p 5000:5000 \
  -e FLASK_ENV=production \
  -e SECRET_KEY=your-secret-key \
  -v $(pwd)/logs:/app/logs \
  x-ray-classifier:latest
```

### Step 3: Test Application

```bash
curl http://localhost:5000
curl http://localhost:5000/health
```

### Using Docker Compose

```bash
# Development mode
docker-compose up

# Production mode
docker-compose -f docker-compose.prod.yml up -d
```

---

## Post-Deployment Configuration

### 1. SSL/HTTPS Setup

All mentioned platforms provide automatic HTTPS:
- **Render**: Automatic with Let's Encrypt
- **Railway**: Automatic with Let's Encrypt
- **Heroku**: Automatic SSL

Enable Security in config:
```python
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
```

### 2. Custom Domain

**Render:**
- Settings → Custom Domain
- Add your domain
- Update DNS records

**Railway:**
- Add custom domain in project settings

**Heroku:**
```bash
heroku domains:add example.com
```

### 3. Monitoring and Logging

#### Render
- View logs in dashboard
- Export logs to external service
- Set up error alerts

#### Railway
- Real-time logs in dashboard
- Integration with external services
- Performance metrics

#### Heroku
```bash
heroku logs --tail --dyno=web
heroku metrics
```

### 4. Performance Optimization

1. **Enable Caching**
   - Set cache headers in Flask
   - Use CDN for static files

2. **Database** (if needed)
   - Render: PostgreSQL add-on
   - Railway: PostgreSQL service
   - Heroku: Heroku Postgres

3. **Worker Processes**
   Currently configured for 4 workers in Gunicorn.
   Adjust in Procfile/settings based on testing.

### 5. Backup Strategy

1. **Model Files**
   - Keep model in Git LFS or remote storage
   - Document exact version

2. **Application Data**
   - Implement database backups if using persistent storage
   - Regular code repository backups (automatic with GitHub)

---

## Troubleshooting

### Deployment Failures

**Build fails - missing dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Model file not found:**
- Ensure model file is tracked in Git LFS
- Or implement download from cloud storage on startup

**Port already in use:**
- Cloud platforms automatically handle this
- Ensure Procfile uses $PORT variable

### Runtime Issues

**Memory limit exceeded:**
- Reduce worker count in Procfile
- Use smaller batch size for predictions
- Upgrade to larger instance type

**Timeout errors:**
- Increase timeout in Procfile: `--timeout 180`
- Optimize image processing
- Use async processing for large files

**Model loading issues:**
```bash
# Test locally first
python -c "import tensorflow; print(tensorflow.__version__)"
python -c "import tensorflow as tf; model = tf.keras.models.load_model('model/densenet_model.h5')"
```

### Configuration Issues

**Environment variables not applied:**
```bash
# Render/Railway: Redeploy after changing variables
# Heroku:
heroku config:get VARIABLE_NAME
heroku dyno:restart
```

**SSL certificate issues:**
- Should auto-resolve on all platforms
- Force HTTPS redirect in app settings

---

## Cost Comparison

| Platform | Free Tier | Starter | Production |
|----------|-----------|---------|------------|
| **Render** | Limited | $7/month | $25+/month |
| **Railway** | 5 Project hours | Pay-as-you-go | Pay-as-you-go |
| **Heroku** | None | $7/month | $25+/month |

---

## Recommended Setup

**For Starting Out:** Railway or Render free tier
**For Production:** Railway or Render with paid plan
**For Enterprise:** Add load balancer, additional services

---

## Summary Checklist

- [ ] Code pushed to GitHub
- [ ] Model file included (Git LFS or downloadable)
- [ ] requirements.txt updated
- [ ] Dockerfile builds successfully locally
- [ ] Environment variables defined
- [ ] Health check passes
- [ ] API endpoints tested
- [ ] Logs accessible
- [ ] Domain configured (if custom)
- [ ] SSL working
- [ ] Backups configured

---

For specific questions about each platform, refer to their official documentation:
- [Render Docs](https://render.com/docs)
- [Railway Docs](https://docs.railway.app)
- [Heroku Docs](https://devcenter.heroku.com)