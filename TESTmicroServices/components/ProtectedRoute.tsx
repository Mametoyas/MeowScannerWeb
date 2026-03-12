"use client"
import { useEffect, useState } from 'react'
import { isUserLoggedIn, getUserData } from '../utils/auth'

interface ProtectedRouteProps {
  children: React.ReactNode
  requireAdmin?: boolean
}

export default function ProtectedRoute({ children, requireAdmin = false }: ProtectedRouteProps) {
  const [isAuthorized, setIsAuthorized] = useState(false)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const checkAuth = () => {
      if (!isUserLoggedIn()) {
        window.location.href = '/login'
        return
      }

      if (requireAdmin) {
        const user = getUserData()
        if (!user || user.role !== 'admin') {
          window.location.href = '/login'
          return
        }
      }

      setIsAuthorized(true)
      setIsLoading(false)
    }

    checkAuth()
  }, [requireAdmin])

  if (isLoading) {
    return <div>Loading...</div>
  }

  if (!isAuthorized) {
    return null
  }

  return <>{children}</>
}