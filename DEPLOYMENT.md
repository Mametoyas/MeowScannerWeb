# Deployment Guide

## 1. Backend Deployment (Render.com)

### Step 1: Prepare Repository
```bash
cd Backend/database
git init
git add .
git commit -m "Initial backend commit"
git remote add origin https://github.com/yourusername/meowscanner-backend.git
git push -u origin main
```

### Step 2: Deploy on Render
1. Go to [render.com](https://render.com)
2. Sign up/Login
3. Click "New" → "Web Service"
4. Connect GitHub repository
5. Configure:
   - **Name**: meowscanner-database-api
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Environment Variables**: Add all from .env file

### Step 3: Environment Variables on Render
Add these in Render dashboard:
- GOOGLE_PROJECT_ID
- GOOGLE_PRIVATE_KEY_ID
- GOOGLE_PRIVATE_KEY
- GOOGLE_CLIENT_EMAIL
- GOOGLE_CLIENT_ID
- GOOGLE_CLIENT_X509_CERT_URL
- GOOGLE_SHEET_ID

### Step 4: Get Backend URL
After deployment, you'll get URL like:
`https://meowscanner-database-api.onrender.com`

## 2. Frontend Deployment (Vercel)

### Step 1: Update API Config
Edit `config/api.ts`:
```typescript
DATABASE_API_PROD: 'https://your-actual-render-url.onrender.com'
```

### Step 2: Prepare Repository
```bash
cd TESTmicroServices
git init
git add .
git commit -m "Initial frontend commit"
git remote add origin https://github.com/yourusername/meowscanner-frontend.git
git push -u origin main
```

### Step 3: Deploy on Vercel
1. Go to [vercel.com](https://vercel.com)
2. Sign up/Login with GitHub
3. Click "New Project"
4. Import GitHub repository
5. Configure:
   - **Framework Preset**: Next.js
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`
6. Deploy

## 3. Final Steps

### Update API URLs
1. Copy your Render backend URL
2. Update `config/api.ts` with real URL
3. Commit and push changes
4. Vercel will auto-redeploy

### Test Deployment
1. Visit your Vercel frontend URL
2. Test login/register
3. Test cat prediction (if model API is running locally)
4. Test search functionality

## 4. URLs Structure

- **Frontend**: `https://your-app.vercel.app`
- **Backend**: `https://your-app.onrender.com`
- **Model API**: `http://localhost:5000` (local only)

## 5. Troubleshooting

### Backend Issues
- Check Render logs
- Verify environment variables
- Test API endpoints directly

### Frontend Issues
- Check Vercel deployment logs
- Verify API URLs in config
- Check browser console for CORS errors

### CORS Issues
Backend already has CORS enabled, but if issues persist:
```python
CORS(app, origins=["https://your-vercel-app.vercel.app"])
```