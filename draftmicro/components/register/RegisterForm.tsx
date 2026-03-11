"use client"
import React, { useState } from "react"
import { authService } from "@/components/api"

const RegisterForm: React.FC = () => {
  const [formData, setFormData] = useState({
    name: '',
    username: '',
    password: '',
    confirmPassword: ''
  })
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (formData.password !== formData.confirmPassword) {
      setMessage('Passwords do not match')
      return
    }

    setLoading(true)
    setMessage('')

    try {
      const result = await authService.register(formData.name, formData.username, formData.password)
      
      if (result.success) {
        setMessage('Registration successful!')
        // Redirect to login page
        window.location.href = '/login'
      } else {
        setMessage(result.error || 'Registration failed')
      }
    } catch (error) {
<<<<<<< Updated upstream
      setMessage('Network or API gateway error. Please try again.')
=======
      setMessage('❌ Network error. Please try again.')
>>>>>>> Stashed changes
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <div className="form-group">
        <label>Full Name</label>
        <input 
          type="text" 
          name="name"
          value={formData.name}
          onChange={handleChange}
          required
        />
      </div>

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

      <div className="form-group">
        <label>Confirm Password</label>
        <input 
          type="password" 
          name="confirmPassword"
          value={formData.confirmPassword}
          onChange={handleChange}
          required
        />
      </div>

      {message && (
        <div className={`message ${message.includes('successful') ? 'success' : 'error'}`} style={message.includes('Network error') ? { color: 'red' } : {}}>
          {message}
        </div>
      )}

      <div className="options">
        <div>
          <button 
            type="submit" 
            className="confirmed-btn"
            disabled={loading}
          >
            {loading ? 'Registering...' : 'Register'}
          </button>
        </div>
      </div>
    </form>
  )
}

export default RegisterForm