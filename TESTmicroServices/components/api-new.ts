import { getAPIUrls } from '../config/api';

const { DATABASE_API, MODEL_API } = getAPIUrls();

export const authService = {
  async login(username: string, password: string) {
    const response = await fetch(`${DATABASE_API}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    return response.json();
  },

  async register(name: string, username: string, password: string) {
    const response = await fetch(`${DATABASE_API}/register`, {
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
    
    const response = await fetch(`${MODEL_API}/predict`, {
      method: 'POST',
      body: formData
    });
    return response.json();
  }
};

export const meowdexService = {
  async getCatInfo(catName: string) {
    const response = await fetch(`/api/meowdex?catName=${encodeURIComponent(catName)}`);
    return response.json();
  },

  async getCatInfoById(catId: string) {
    const response = await fetch(`/api/meowdex?catId=${encodeURIComponent(catId)}`);
    return response.json();
  }
};

export const dataService = {
  async addMapLocation(uid: string, longitude: number, latitude: number, catId: string) {
    const response = await fetch(`${DATABASE_API}/add-location`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ uid, longitude, latitude, cat_id: catId })
    });
    return response.json();
  },

  async getCatRecommendation(housing: string, lifestyle: string, personality: string, uid?: string) {
    const response = await fetch(`${DATABASE_API}/cat-recommendation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ location: housing, time: lifestyle, personality, uid })
    });
    return response.json();
  },

  async addHistory(userId: string, catId: string) {
    const response = await fetch(`${DATABASE_API}/add-history`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, cat_id: catId })
    });
    return response.json();
  }
};