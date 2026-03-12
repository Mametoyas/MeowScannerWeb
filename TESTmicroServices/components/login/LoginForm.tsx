"use client"
import React, { useState } from "react"
import { authService } from "@/components/api"

const LoginForm: React.FC = () => {
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  })
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')
  const [rememberMe, setRememberMe] = useState(false)

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setMessage('')

    try {
      const result = await authService.login(formData.username, formData.password)
      
      if (result.success) {
        // Store user data
        const userData = {
          user_id: result.user_id,
          name: result.name,
          role: result.role || 'user'
        }
        
        if (rememberMe) {
          localStorage.setItem('user', JSON.stringify(userData))
        } else {
          sessionStorage.setItem('user', JSON.stringify(userData))
        }
        
        setMessage('Login successful!')
        
        // Redirect based on role
        if (result.role === 'admin') {
          window.location.href = '/main'
        } else {
          window.location.href = '/main'
        }
      } else {
        setMessage(result.error || 'Login failed')
      }
    } catch (error) {
      setMessage('Network error. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <div className="form-group">
        <label>Username</label>
        <input 
          type="text" 
          name="username"
          value={formData.username}
          onChange={handleChange}
          required
        />
      </div>

      <div className="form-group">
        <label>Password</label>
        <input 
          type="password" 
          name="password"
          value={formData.password}
          onChange={handleChange}
          required
        />
      </div>

      {message && (
        <div className={`message ${message.includes('successful') ? 'success' : 'error'}`}>
          {message}
        </div>
      )}

      <div className="options">
        <label>
          <input 
            type="checkbox" 
            checked={rememberMe}
            onChange={(e) => setRememberMe(e.target.checked)}
          /> Remember me
        </label>

        <div>
          {/* <a href="/forgot-password">Forgot your password?</a> */}
          <button 
            type="submit" 
            className="login-btn"
            disabled={loading}
          >
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </div>
      </div>
    </form>
  )
}

export default LoginForm