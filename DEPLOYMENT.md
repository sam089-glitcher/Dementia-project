# Deployment Guide

This guide covers multiple options for hosting your Dementia Prevention Advisor application online.

## Option 1: Streamlit Cloud (Easiest - Free)

**Best for:** Quick deployment of the Streamlit app

### Steps:

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/dementia-project.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file path: `src/api/app.py`
   - Click "Deploy"

3. **Set Environment Variables**
   - In Streamlit Cloud settings, add:
     - `API_URL`: Your API URL (if hosting API separately)

**Note:** For full functionality, you'll need to host the API separately (see Option 2 or 3).

---

## Option 2: Railway (Recommended - Easy & Free Tier)

**Best for:** Deploying both API and Streamlit together

### Steps:

1. **Install Railway CLI** (optional)
   ```bash
   npm i -g @railway/cli
   railway login
   ```

2. **Deploy via Railway Dashboard**
   - Go to https://railway.app
   - Sign in with GitHub
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Configure Services**
   - Railway will auto-detect `railway.json`
   - For Streamlit, add a second service:
     - New Service → GitHub Repo
     - Start Command: `streamlit run src/api/app.py --server.port $PORT --server.address 0.0.0.0`
     - Add environment variable: `API_URL` = your API service URL

4. **Get Your URLs**
   - Railway provides public URLs for each service
   - Update Streamlit's `API_URL` environment variable

**Cost:** Free tier includes $5/month credit

---

## Option 3: Render (Free Tier Available)

**Best for:** Simple deployment with `render.yaml` configuration

### Steps:

1. **Push code to GitHub** (if not already)

2. **Deploy on Render**
   - Go to https://render.com
   - Sign in with GitHub
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Render will detect `render.yaml` automatically

3. **Manual Setup (Alternative)**
   - New → Web Service
   - Connect GitHub repo
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
   - Add environment variables as needed

**Cost:** Free tier available (spins down after inactivity)

---

## Option 4: Fly.io (Docker-based)

**Best for:** Docker deployments with good performance

### Steps:

1. **Install Fly CLI**
   ```bash
   powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
   ```

2. **Create fly.toml** (already created)
   ```bash
   fly launch
   ```

3. **Deploy**
   ```bash
   fly deploy
   ```

**Cost:** Free tier includes 3 shared VMs

---

## Option 5: Heroku (Paid)

**Best for:** Traditional PaaS deployment

### Steps:

1. **Install Heroku CLI**
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login and Create App**
   ```bash
   heroku login
   heroku create dementia-api
   heroku create dementia-streamlit
   ```

3. **Deploy**
   ```bash
   git push heroku main
   ```

4. **Set Environment Variables**
   ```bash
   heroku config:set API_URL=https://your-api-url.herokuapp.com -a dementia-streamlit
   ```

**Cost:** Paid plans start at $7/month

---

## Environment Variables

Set these in your hosting platform:

- `API_URL`: URL of your API service (for Streamlit app)
- `PORT`: Usually set automatically by the platform
- `PYTHONUNBUFFERED`: Set to `1` for better logging

---

## Quick Start Commands

### Local Testing Before Deployment

```bash
# Test API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Test Streamlit (in another terminal)
streamlit run src/api/app.py --server.port 8501
```

### Docker Testing

```bash
# Build and run
docker-compose up --build
```

---

## Recommended Setup

For production, I recommend:

1. **API**: Deploy on Railway or Render
2. **Streamlit**: Deploy on Streamlit Cloud (free, easy)
3. **Connect**: Set `API_URL` in Streamlit Cloud to your API URL

This gives you:
- ✅ Free hosting
- ✅ Easy updates
- ✅ Good performance
- ✅ Separate scaling

---

## Troubleshooting

### API not connecting
- Check `API_URL` environment variable
- Verify API is publicly accessible (not localhost)
- Check CORS settings in `main.py`

### Build failures
- Ensure `requirements.txt` is up to date
- Check Python version compatibility
- Review build logs in your platform

### Port issues
- Most platforms set `$PORT` automatically
- Don't hardcode ports in production

---

## Security Notes

- ✅ CORS is configured in `main.py`
- ⚠️ Add authentication for production use
- ⚠️ Consider rate limiting for API
- ⚠️ Use HTTPS (most platforms provide this)

---

## Next Steps

1. Choose a hosting platform
2. Push code to GitHub
3. Follow platform-specific steps above
4. Update `API_URL` environment variable
5. Test your deployed app!

For questions, check platform documentation or open an issue.

