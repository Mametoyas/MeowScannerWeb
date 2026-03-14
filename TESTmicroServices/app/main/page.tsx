"use client"
import { useState, useEffect } from 'react'
import { getUserData, logoutUser } from '../../utils/auth'
import ProtectedRoute from '@/components/ProtectedRoute'
import Navbar from '@/components/main/MainNavbar'
import Link from 'next/link'
import '../../styles/Main.css'

interface UserStats {
  predictions_made: number;
  cats_discovered: number;
  locations_mapped: number;
}

export default function MainPage() {
  const [user, setUser] = useState<any>(null)
  const [stats, setStats] = useState<UserStats>({ predictions_made: 0, cats_discovered: 0, locations_mapped: 0 })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const currentUser = getUserData()
    setUser(currentUser)
    if (currentUser) {
      fetchUserStats(currentUser.user_id)
    } else {
      setLoading(false)
    }
  }, [])

  const fetchUserStats = async (userId: string) => {
    try {
      const { DATABASE_API } = await import('../../config/api').then(m => m.getAPIUrls())
      const response = await fetch(`${DATABASE_API}/get-user-stats?user_id=${userId}`)
      const data = await response.json()
      
      if (response.ok && data.success) {
        setStats(data.stats)
      } else {
        console.error('Failed to fetch user stats:', data.error)
      }
    } catch (error) {
      console.error('Error fetching user stats:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleLogout = () => {
    logoutUser()
  }

  return (
    <ProtectedRoute>
      <Navbar onLogout={handleLogout} userName={user?.name} userRole={user?.role} />
      <div className="main-container">
        {/* Main Content */}
        <main className="main-content">
          

          {/* Feature Cards */}
          <div className="feature-grid">
            <Link href="/predict" className="feature-card">
              <div className="card-icon">🔍</div>
              <h3>What a Meow??</h3>
              <p>Upload a photo to identify cat breeds with AI technology</p>
            </Link>

            <Link href="/search" className="feature-card">
              <div className="card-icon">🔎</div>
              <h3>Search Cats</h3>
              <p>Search and explore different cat breeds and their information</p>
            </Link>

            <Link href="/preference" className="feature-card">
              <div className="card-icon">❤️</div>
              <h3>Cat Recommendation</h3>
              <p>Get personalized cat breed recommendations based on your lifestyle</p>
            </Link>

            <Link href="/catmap" className="feature-card">
              <div className="card-icon">🗺️</div>
              <h3>Cat Map</h3>
              <p>View locations where cats have been spotted and identified</p>
            </Link>
          </div>

          {/* Stats Section */}
          <div className="stats-section">
            <h3>Your Activity</h3>
            {loading ? (
              <div className="stats-loading">
                <p>📊 Loading your activity...</p>
              </div>
            ) : (
              <div className="stats-grid">
                <div className="stat-item">
                  <div className="stat-number">{stats.predictions_made}</div>
                  <div className="stat-label">Predictions Made</div>
                </div>
                <div className="stat-item">
                  <div className="stat-number">{stats.cats_discovered}</div>
                  <div className="stat-label">Cats Discovered</div>
                </div>
                <div className="stat-item">
                  <div className="stat-number">{stats.locations_mapped}</div>
                  <div className="stat-label">Locations Mapped</div>
                </div>
              </div>
            )}
            
            {!loading && stats.predictions_made === 0 && (
              <div className="no-activity">
                <p>🐱 Start your cat discovery journey!</p>
                <Link href="/predict" className="start-btn">
                  📸 Make Your First Prediction
                </Link>
              </div>
            )}
          </div>
        </main>
      </div>
    </ProtectedRoute>
  )
}