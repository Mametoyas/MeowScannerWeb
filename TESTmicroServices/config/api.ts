// API Configuration
const API_CONFIG = {
  // Get URLs from environment variables
  DATABASE_API: process.env.NEXT_PUBLIC_DB_API_URL || 'http://localhost:5000',
  MODEL_API: process.env.NEXT_PUBLIC_MODEL_API_URL || 'http://localhost:5001'
}

export const getAPIUrls = () => {
  return {
    DATABASE_API: API_CONFIG.DATABASE_API,
    MODEL_API: API_CONFIG.MODEL_API
  }
}

// For debugging
if (typeof window !== 'undefined') {
  console.log('API URLs:', {
    DATABASE_API: API_CONFIG.DATABASE_API,
    MODEL_API: API_CONFIG.MODEL_API
  })
}

export default API_CONFIG