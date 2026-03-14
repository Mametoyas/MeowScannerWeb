# Complete Deployment Guide

## 📋 Overview
- **Frontend**: Next.js on Vercel
- **Backend**: Flask on Render.com  
- **Database**: Google Sheets
- **Model API**: Local development

## 🔧 Environment Variables Setup

### Backend (.env)
```env
# Google Sheets API Credentials
GOOGLE_PROJECT_ID=your-project-id
GOOGLE_PRIVATE_KEY_ID=your-private-key-id
GOOGLE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
GOOGLE_CLIENT_EMAIL=service-account@project.iam.gserviceaccount.com
GOOGLE_CLIENT_ID=123456789
GOOGLE_CLIENT_X509_CERT_URL=https://www.googleapis.com/robot/v1/metadata/x509/...
GOOGLE_SHEET_ID=your-sheet-id
PORT=5001
```

### Frontend (.env.local)
```env
# API URLs
NEXT_PUBLIC_DATABASE_API_URL=https://your-backend.onrender.com
NEXT_PUBLIC_MODEL_API_URL=http://localhost:5000
```

## 🚀 Deployment Steps

### 1. Backend Deployment (Render.com)

#### Step 1: Prepare Repository
```bash
cd Backend/database
git init
git add .
git commit -m "Backend deployment ready"
git remote add origin https://github.com/yourusername/meowscanner-backend.git
git push -u origin main
```

#### Step 2: Deploy on Render
1. Go to [render.com](https://render.com) → Sign up
2. New → Web Service
3. Connect GitHub repository
4. Configure:
   - **Name**: `meowscanner-backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Python Version**: `3.9.16`

#### Step 3: Add Environment Variables
In Render dashboard, add all variables from your `.env` file:
- GOOGLE_PROJECT_ID
- GOOGLE_PRIVATE_KEY_ID
- GOOGLE_PRIVATE_KEY (include \\n characters)
- GOOGLE_CLIENT_EMAIL
- GOOGLE_CLIENT_ID
- GOOGLE_CLIENT_X509_CERT_URL
- GOOGLE_SHEET_ID

#### Step 4: Get Backend URL
After deployment: `https://meowscanner-backend.onrender.com`

### 2. Frontend Deployment (Vercel)

#### Step 1: Update Environment Variables
Create `.env.local`:
```env
NEXT_PUBLIC_DATABASE_API_URL=https://meowscanner-backend.onrender.com
NEXT_PUBLIC_MODEL_API_URL=http://localhost:5000
```

#### Step 2: Prepare Repository
```bash
cd TESTmicroServices
git init
git add .
git commit -m "Frontend deployment ready"
git remote add origin https://github.com/yourusername/meowscanner-frontend.git
git push -u origin main
```

#### Step 3: Deploy on Vercel
1. Go to [vercel.com](https://vercel.com) → Sign up with GitHub
2. New Project → Import GitHub repository
3. Configure:
   - **Framework**: Next.js (auto-detected)
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`

#### Step 4: Add Environment Variables
In Vercel dashboard → Settings → Environment Variables:
- `NEXT_PUBLIC_DATABASE_API_URL` = `https://your-backend.onrender.com`
- `NEXT_PUBLIC_MODEL_API_URL` = `http://localhost:5000`

## 🧪 Testing Deployment

### Backend API Test
```bash
curl https://your-backend.onrender.com/health
# Should return: {"code": 200}
```

### Frontend Test
1. Visit your Vercel URL
2. Test login/register
3. Test search functionality
4. Test cat map (if you have data)

## 🔄 Development vs Production

### Local Development
```env
NEXT_PUBLIC_DATABASE_API_URL=http://localhost:5001
NEXT_PUBLIC_MODEL_API_URL=http://localhost:5000
```

### Production
```env
NEXT_PUBLIC_DATABASE_API_URL=https://your-backend.onrender.com
NEXT_PUBLIC_MODEL_API_URL=http://localhost:5000
```

## 🐛 Troubleshooting

### Backend Issues
- Check Render logs: Dashboard → Service → Logs
- Verify environment variables are set
- Test API endpoints directly

### Frontend Issues
- Check Vercel deployment logs
- Verify environment variables in Vercel dashboard
- Check browser console for API errors

### CORS Issues
Backend already configured for CORS, but if needed:
```python
CORS(app, origins=["https://your-frontend.vercel.app"])
```

## 📝 Final Checklist

- [ ] Backend deployed on Render
- [ ] All environment variables set in Render
- [ ] Backend API responding to health check
- [ ] Frontend environment variables updated
- [ ] Frontend deployed on Vercel
- [ ] Environment variables set in Vercel
- [ ] Frontend can connect to backend API
- [ ] All features working in production

## 🔗 URLs Structure

- **Frontend**: `https://your-app.vercel.app`
- **Backend API**: `https://your-backend.onrender.com`
- **Health Check**: `https://your-backend.onrender.com/health`
- **Model API**: `http://localhost:5000` (local only)