"use client"
import { useState, useEffect } from 'react'
import { getUserData } from '../../utils/auth'
import ProtectedRoute from '@/components/ProtectedRoute'

export default function SimpleMapPage() {
  const [locations, setLocations] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [user, setUser] = useState<any>(null)

  useEffect(() => {
    const currentUser = getUserData()
    setUser(currentUser)
    if (currentUser) {
      fetchUserMapData(currentUser.user_id)
    }
  }, [])

  const fetchUserMapData = async (userId: string) => {
    try {
      const response = await fetch(`http://localhost:5001/get-user-map-locations?user_id=${userId}`)
      const data = await response.json()
      if (response.ok) {
        setLocations(data.locations || [])
      }
    } catch (error) {
      console.error('Error:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <ProtectedRoute>
      <div style={{ padding: '2rem', minHeight: '100vh', backgroundColor: '#f8f9fa' }}>
        <div style={{ background: 'white', padding: '2rem', borderRadius: '0.5rem', marginBottom: '2rem' }}>
          <h1>🗺️ My Cat Map</h1>
          <button onClick={() => window.location.href = '/main'}>🏠 Home</button>
        </div>

        {loading ? (
          <div>Loading...</div>
        ) : locations.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '4rem' }}>
            <h2>🐱 No Cat Locations Yet</h2>
            <p>Start predicting cats with GPS-enabled photos!</p>
            <button onClick={() => window.location.href = '/predict'}>
              📸 Start Predicting
            </button>
          </div>
        ) : (
          <div style={{ background: 'white', padding: '2rem', borderRadius: '0.5rem' }}>
            <h3>Your Cat Discoveries ({locations.length})</h3>
            {locations.map((location, index) => (
              <div key={location.ID} style={{ 
                padding: '1rem', 
                margin: '0.5rem 0', 
                background: '#f9fafb', 
                borderRadius: '0.375rem',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}>
                <div>
                  <strong>Discovery #{index + 1}</strong>
                  <div>Cat ID: {location.CatID}</div>
                  <div>Location: {location.Latitude}, {location.Longitude}</div>
                </div>
                <button 
                  onClick={() => window.open(`https://www.google.com/maps?q=${location.Latitude},${location.Longitude}`, '_blank')}
                  style={{
                    background: '#1d4ed8',
                    color: 'white',
                    border: 'none',
                    padding: '0.5rem 1rem',
                    borderRadius: '0.25rem',
                    cursor: 'pointer'
                  }}
                >
                  📍 View in Google Maps
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </ProtectedRoute>
  )
}