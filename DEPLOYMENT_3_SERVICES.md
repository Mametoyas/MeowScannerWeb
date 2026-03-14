# 🚀 Complete 3-Service Deployment Guide

## 📋 Architecture Overview
- **Database API**: Flask + Google Sheets (Render.com)
- **Model API**: Flask + PyTorch (Render.com)  
- **Frontend**: Next.js (Vercel)

## 🔧 Environment Variables

### Database API (.env)
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

### Model API (.env)
```env
# No environment variables needed
PORT=5000
```

### Frontend (.env.local)
```env
# API URLs
NEXT_PUBLIC_DATABASE_API_URL=https://your-database-api.onrender.com
NEXT_PUBLIC_MODEL_API_URL=https://your-model-api.onrender.com
```

## 🚀 Deployment Steps

### 1. Database API Deployment (Render.com)

#### Step 1: Prepare Repository
```bash
cd Backend/database
git init
git add .
git commit -m "Database API deployment"
git remote add origin https://github.com/yourusername/meowscanner-database-api.git
git push -u origin main
```

#### Step 2: Deploy on Render
1. Go to [render.com](https://render.com)
2. New → Web Service
3. Connect GitHub repository
4. Configure:
   - **Name**: `meowscanner-database-api`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Python Version**: `3.9.16`

#### Step 3: Add Environment Variables
Add all Google Sheets credentials in Render dashboard

#### Step 4: Get URL
Result: `https://meowscanner-database-api.onrender.com`

### 2. Model API Deployment (Render.com)

#### Step 1: Upload Model File
**⚠️ Important**: Copy `DenseNet121_CatV3_withCBAMv8.pth` to `Backend/model/` directory

#### Step 2: Prepare Repository
```bash
cd Backend/model
# Copy model file first
cp ../DenseNet121_CatV3_withCBAMv8.pth .
git init
git add .
git commit -m "Model API deployment"
git remote add origin https://github.com/yourusername/meowscanner-model-api.git
git push -u origin main
```

#### Step 3: Deploy on Render
1. New → Web Service
2. Connect GitHub repository
3. Configure:
   - **Name**: `meowscanner-model-api`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --timeout 120`
   - **Python Version**: `3.9.16`
   - **Instance Type**: Standard (for better performance)

#### Step 4: Get URL
Result: `https://meowscanner-model-api.onrender.com`

### 3. Frontend Deployment (Vercel)

#### Step 1: Update Environment Variables
Create `.env.local`:
```env
NEXT_PUBLIC_DATABASE_API_URL=https://meowscanner-database-api.onrender.com
NEXT_PUBLIC_MODEL_API_URL=https://meowscanner-model-api.onrender.com
```

#### Step 2: Prepare Repository
```bash
cd TESTmicroServices
git init
git add .
git commit -m "Frontend deployment"
git remote add origin https://github.com/yourusername/meowscanner-frontend.git
git push -u origin main
```

#### Step 3: Deploy on Vercel
1. Go to [vercel.com](https://vercel.com)
2. New Project → Import GitHub repository
3. Add Environment Variables:
   - `NEXT_PUBLIC_DATABASE_API_URL`
   - `NEXT_PUBLIC_MODEL_API_URL`

#### Step 4: Get URL
Result: `https://your-app.vercel.app`

## 🧪 Testing Deployment

### Test Database API
```bash
curl https://meowscanner-database-api.onrender.com/health
# Expected: {"code": 200}
```

### Test Model API
```bash
curl https://meowscanner-model-api.onrender.com/health
# Expected: {"code": 200}
```

### Test Frontend
1. Visit Vercel URL
2. Check browser console for API URLs
3. Test login/register
4. Test cat prediction
5. Test search functionality

## 📊 Service URLs Structure

| Service | Development | Production |
|---------|-------------|------------|
| Database API | http://localhost:5001 | https://meowscanner-database-api.onrender.com |
| Model API | http://localhost:5000 | https://meowscanner-model-api.onrender.com |
| Frontend | http://localhost:3000 | https://your-app.vercel.app |

## ⚠️ Important Notes

### Model API Considerations
- **File Size**: Model file (~100MB) must be included in repository
- **Memory**: Use Standard instance type on Render for better performance
- **Timeout**: Increased to 120 seconds for model loading
- **Cold Start**: First request may take 30-60 seconds

### Database API
- **Free Tier**: Render free tier sleeps after 15 minutes of inactivity
- **Wake Up**: First request after sleep takes ~30 seconds

### Frontend
- **Environment Variables**: Must start with `NEXT_PUBLIC_`
- **Build Time**: Variables are embedded at build time

## 🐛 Troubleshooting

### Model API Issues
- **Memory Error**: Upgrade to Standard instance
- **Model Not Found**: Ensure `.pth` file is in repository
- **Timeout**: Check if model is loading correctly

### Database API Issues
- **Google Sheets**: Verify service account permissions
- **Environment Variables**: Check all credentials are set

### Frontend Issues
- **API Calls Failing**: Check environment variables
- **CORS Errors**: Both APIs have CORS enabled

## 🔄 Development vs Production

### Development
```env
NEXT_PUBLIC_DATABASE_API_URL=http://localhost:5001
NEXT_PUBLIC_MODEL_API_URL=http://localhost:5000
```

### Production
```env
NEXT_PUBLIC_DATABASE_API_URL=https://meowscanner-database-api.onrender.com
NEXT_PUBLIC_MODEL_API_URL=https://meowscanner-model-api.onrender.com
```

## 📝 Final Checklist

- [ ] Database API deployed and responding
- [ ] Model API deployed with model file
- [ ] Frontend environment variables updated
- [ ] All services responding to health checks
- [ ] Frontend can connect to both APIs
- [ ] Login/Register working
- [ ] Cat prediction working
- [ ] Search functionality working
- [ ] Map functionality working

## 💰 Cost Estimation (Render Free Tier)

- **Database API**: Free (sleeps after 15 min)
- **Model API**: Free (sleeps after 15 min) 
- **Frontend**: Free (Vercel)
- **Total**: $0/month

For production use, consider upgrading to paid plans for better performance and uptime.