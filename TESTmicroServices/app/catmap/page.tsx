"use client"
import { useState, useEffect } from 'react'
import { getUserData } from '../../utils/auth'
import ProtectedRoute from '@/components/ProtectedRoute'
import '../../styles/CatMap.css'

interface MapLocation {
  ID: string
  UID: string
  Longitude: number
  Latitude: number
  CatID: string
}

interface CatData {
  CatID: string
  CatName: string
  CatPersonal: string
  Cat: string
}

export default function CatMapPage() {
  const [locations, setLocations] = useState<MapLocation[]>([])
  const [cats, setCats] = useState<CatData[]>([])
  const [loading, setLoading] = useState(true)
  const [message, setMessage] = useState('')
  const [user, setUser] = useState<any>(null)
  const [mapInitialized, setMapInitialized] = useState(false)

  useEffect(() => {
    const currentUser = getUserData()
    setUser(currentUser)
    if (currentUser) {
      fetchUserMapData(currentUser.user_id)
    }
    fetchCatData()
  }, [])

  useEffect(() => {
    if (!loading && locations.length > 0 && cats.length > 0 && !mapInitialized) {
      setTimeout(() => {
        initializeMap()
      }, 100)
    }
  }, [loading, locations, cats, mapInitialized])

  const fetchUserMapData = async (userId: string) => {
    try {
      const { DATABASE_API } = await import('../../config/api').then(m => m.getAPIUrls())
      const response = await fetch(`${DATABASE_API}/get-user-map-locations?user_id=${userId}`)
      const data = await response.json()
      if (response.ok) {
        setLocations(data.locations || [])
      } else {
        setMessage('Failed to fetch map data')
      }
    } catch (error) {
      console.error('Error fetching map data:', error)
      setMessage('Error fetching map data')
    }
  }

  const fetchCatData = async () => {
    try {
      const { DATABASE_API } = await import('../../config/api').then(m => m.getAPIUrls())
      const response = await fetch(`${DATABASE_API}/get-cats`)
      const data = await response.json()
      if (response.ok) {
        setCats(data.cats || [])
      }
    } catch (error) {
      console.error('Error fetching cat data:', error)
    } finally {
      setLoading(false)
    }
  }

  const getCatName = (catId: string) => {
    const cat = cats.find(c => c.CatID === catId)
    return cat ? cat.CatName : catId
  }

  const getCatColor = (catId: string) => {
    const colors = {
      'C0001': '#FF6B6B', // Abyssinian - Red
      'C0002': '#4ECDC4', // Bengal - Teal
      'C0003': '#45B7D1', // Birman - Blue
      'C0004': '#96CEB4', // Oriental - Green
      'C0005': '#FFEAA7', // Other - Yellow
      'C0006': '#DDA0DD', // Siamese - Plum
      'C0007': '#FFB347', // Somali - Orange
      'C0008': '#F8BBD9', // Sphynx - Pink
      'C0009': '#C7CEEA'  // Toyger - Lavender
    }
    return colors[catId as keyof typeof colors] || '#999999'
  }

  const initializeMap = () => {
    if (typeof window === 'undefined' || mapInitialized) return

    // Load Leaflet CSS
    const cssLink = document.createElement('link')
    cssLink.rel = 'stylesheet'
    cssLink.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'
    document.head.appendChild(cssLink)

    // Load Leaflet JS
    const script = document.createElement('script')
    script.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'
    script.onload = () => {
      const L = (window as any).L
      
      // Calculate center point
      const centerLat = locations.reduce((sum, loc) => sum + loc.Latitude, 0) / locations.length
      const centerLng = locations.reduce((sum, loc) => sum + loc.Longitude, 0) / locations.length
      
      // Create map
      const map = L.map('leaflet-map').setView([centerLat, centerLng], 12)
      
      // Add tile layer
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
      }).addTo(map)

      // Add markers for each location
      locations.forEach((location, index) => {
        const catName = getCatName(location.CatID)
        const color = getCatColor(location.CatID)
        
        // Create custom icon
        const customIcon = L.divIcon({
          html: `<div style="
            background-color: ${color};
            width: 25px;
            height: 25px;
            border-radius: 50%;
            border: 2px solid white;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
          ">${index + 1}</div>`,
          className: 'custom-div-icon',
          iconSize: [29, 29],
          iconAnchor: [14, 14]
        })

        // Add marker with popup
        L.marker([location.Latitude, location.Longitude], { icon: customIcon })
          .addTo(map)
          .bindPopup(`
            <div style="text-align: center; padding: 5px;">
              <h4 style="margin: 0 0 5px 0; color: #333;">${catName}</h4>
              <p style="margin: 2px 0; font-size: 12px; color: #666;">Location: ${location.ID}</p>
              <p style="margin: 2px 0; font-size: 12px; color: #666;">Discovery #${index + 1}</p>
              <a href="https://www.google.com/maps?q=${location.Latitude},${location.Longitude}" 
                target="_blank" 
                style="color: #1a73e8; text-decoration: none; font-size: 12px;">
                View in Google Maps
              </a>
            </div>
          `)
      })

      // Fit map to show all markers
      if (locations.length > 1) {
        const group = new L.featureGroup(
          locations.map(loc => L.marker([loc.Latitude, loc.Longitude]))
        )
        map.fitBounds(group.getBounds().pad(0.1))
      }

      setMapInitialized(true)
    }
    
    document.head.appendChild(script)
  }

  return (
    <ProtectedRoute>
      <div className="catmap-container">
        <header className="catmap-header">
          <h1>🗺️ My Cat Map</h1>
          <div className="header-controls">
            <span className="location-count">{locations.length} locations found</span>
            <button 
              onClick={() => window.location.href = '/'} 
              className="home-btn"
            >
              Home
            </button>
          </div>
        </header>

        {message && (
          <div className={`message ${message.includes('Failed') || message.includes('Error') ? 'error' : 'success'}`}>
            {message}
          </div>
        )}

        <div className="map-content">
          {loading ? (
            <div className="loading">Loading your cat map...</div>
          ) : locations.length === 0 ? (
            <div className="no-data">
              <h2>No Cat Locations Yet</h2>
              <p>Start predicting cats with GPS-enabled photos to see them on your map!</p>
              <button 
                onClick={() => window.location.href = '/predict'}
                className="predict-btn"
              >
                Start Predicting Cats
              </button>
            </div>
          ) : (
            <>
              <div className="map-container">
                <div id="leaflet-map" className="leaflet-map"></div>
              </div>

              <div className="legend">
                <h3>🐾 Your Cat Discoveries ({locations.length})</h3>
                <div className="legend-items">
                  {locations.map((location, index) => {
                    const catName = getCatName(location.CatID)
                    const color = getCatColor(location.CatID)
                    
                    return (
                      <div key={`${location.ID}-${index}`} className="legend-item">
                        <div 
                          className="legend-marker" 
                          style={{ backgroundColor: color }}
                        >
                          {index + 1}
                        </div>
                        <div className="legend-info">
                          <strong>{catName}</strong>
                          <button 
                            className="view-btn"
                            onClick={() => window.open(`https://www.google.com/maps?q=${location.Latitude},${location.Longitude}`, '_blank')}
                          >
                            View
                          </button>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </ProtectedRoute>
  )
}