const DATABASE_API_URL = process.env.NODE_ENV === 'production' 
  ? process.env.NEXT_PUBLIC_DATABASE_API_URL || 'https://your-database-api.vercel.app'
  : 'http://localhost:5001';

const MODEL_API_URL = process.env.NEXT_PUBLIC_MODEL_API_URL || 'http://localhost:5000';

export const authService = {
  async login(username: string, password: string) {
    const response = await fetch(`${DATABASE_API_URL}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    return response.json();
  },

  async register(name: string, username: string, password: string) {
    const response = await fetch(`${DATABASE_API_URL}/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, username, password })
    });
    return response.json();
  }
};

export const modelService = {
  async predict(imageFile: File) {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    const response = await fetch(`${MODEL_API_URL}/predict`, {
      method: 'POST',
      body: formData
    });
    return response.json();
  }
};

export const dataService = {
  async addMapLocation(uid: string, longitude: number, latitude: number, catId: string) {
    const response = await fetch(`${DATABASE_API_URL}/add-location`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ uid, longitude, latitude, cat_id: catId })
    });
    return response.json();
  },

  async getCatRecommendation(housing: string, lifestyle: string, personality: string, uid?: string) {
    const response = await fetch(`${DATABASE_API_URL}/cat-recommendation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ location: housing, time: lifestyle, personality, uid })
    });
    return response.json();
  },

  async addHistory(userId: string, catId: string) {
    const response = await fetch(`${DATABASE_API_URL}/add-history`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, cat_id: catId })
    });
    return response.json();
  }
};