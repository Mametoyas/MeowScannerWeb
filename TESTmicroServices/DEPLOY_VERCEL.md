# Deploy Frontend บน Vercel

## 🚀 ขั้นตอน Deploy

### 1. เตรียม Repository

ตรวจสอบว่า push โค้ดขึ้น GitHub แล้ว:
```bash
git add .
git commit -m "Ready for Vercel deployment"
git push
```

### 2. ไปที่ Vercel

1. ไปที่ https://vercel.com
2. Login ด้วย GitHub
3. คลิก **"Add New..." → "Project"**
4. เลือก repository: `TestDeploySE`

### 3. ตั้งค่า Project

```
Framework Preset: Next.js (ตรวจจับอัตโนมัติ)
Root Directory: TESTmicroServices
Build Command: npm run build (default)
Output Directory: .next (default)
Install Command: npm install (default)
```

### 4. ตั้งค่า Environment Variables

คลิก **"Environment Variables"** แล้วเพิ่ม:

```
NEXT_PUBLIC_DB_API_URL=https://meowscanner-database.onrender.com
NEXT_PUBLIC_MODEL_API_URL=https://meowscanner-model.onrender.com
```

⚠️ **สำคัญ**: ใส่ URL จริงจาก Render ที่ deploy ไว้แล้ว

### 5. Deploy

1. คลิก **"Deploy"**
2. รอ 2-3 นาที
3. เสร็จแล้ว! 🎉

---

## 📝 URL ที่ได้

```
https://your-project-name.vercel.app
```

หรือตั้ง Custom Domain ได้ที่ Settings → Domains

---

## ⚙️ ตั้งค่าเพิ่มเติม (ถ้าต้องการ)

### เพิ่ม Custom Domain

1. Settings → Domains
2. เพิ่ม domain ของคุณ
3. ตั้งค่า DNS ตามที่ Vercel บอก

### Auto Deploy

Vercel จะ auto-deploy ทุกครั้งที่ push ไป GitHub:
- `main` branch → Production
- branch อื่นๆ → Preview

### Environment Variables แยกตาม Environment

```
Production: ใช้ URL จริง
Preview: ใช้ URL ทดสอบ
Development: ใช้ localhost
```

---

## 🔧 แก้ปัญหาที่อาจเจอ

### ปัญหา 1: Build Failed

ตรวจสอบ:
```bash
# ทดสอบ build ใน local ก่อน
cd TESTmicroServices
npm install
npm run build
```

### ปัญหา 2: API ไม่ทำงาน (CORS Error)

ต้องเพิ่ม CORS ใน Backend:
```python
# ใน app.py (Database & Model)
from flask_cors import CORS
CORS(app)  # ✅ มีอยู่แล้ว
```

### ปัญหา 3: Environment Variables ไม่ทำงาน

- ต้องขึ้นต้นด้วย `NEXT_PUBLIC_` เท่านั้น
- Redeploy หลังเปลี่ยน env variables

### ปัญหา 4: Leaflet Map ไม่แสดง

เพิ่มใน `next.config.ts`:
```typescript
const nextConfig: NextConfig = {
  transpilePackages: ['leaflet', 'react-leaflet'],
};
```

---

## 📊 สรุป URLs ทั้งหมด

```
Frontend:  https://your-project.vercel.app
Database:  https://meowscanner-database.onrender.com
Model:     https://meowscanner-model.onrender.com
```

---

## 💡 Tips

1. **Free tier** ของ Vercel:
   - Unlimited deployments
   - 100GB bandwidth/เดือน
   - Auto SSL
   - Global CDN

2. **ทดสอบก่อน deploy:**
   ```bash
   npm run build
   npm start
   ```

3. **ดู logs:**
   - Vercel Dashboard → Deployments → View Function Logs

4. **Rollback:**
   - Deployments → เลือก version เก่า → Promote to Production
