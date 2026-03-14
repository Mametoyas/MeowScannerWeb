# Deploy Model API ด้วย Cloudflare Tunnel (Local Machine)

## ขั้นตอน Setup

### 1. ติดตั้ง cloudflared

**Windows:**
```bash
winget install --id Cloudflare.cloudflared
```

หรือดาวน์โหลดจาก: https://github.com/cloudflare/cloudflared/releases

### 2. Login Cloudflare

```bash
cloudflared tunnel login
```

จะเปิด browser ให้ login (สมัครฟรีได้ที่ cloudflare.com)

### 3. สร้าง Tunnel

```bash
1. cloudflared tunnel create meowscanner-model
2. cloudflared tunnel --url http://localhost:5000
```

จะได้:
+--------------------------------------------------------------------------------------------+
|  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
|  https://fraction-embedded-comprehensive-gives.trycloudflare.com   <liNK>                  |
+--------------------------------------------------------------------------------------------+