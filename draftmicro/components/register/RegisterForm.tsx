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
  const [passwordErrors, setPasswordErrors] = useState<string[]>([])

  const validatePassword = (password: string): string[] => {
    const errors: string[] = []
    
    // Check length (minimum 8 characters)
    if (password.length < 8) {
      errors.push('Password must be at least 8 characters long')
    }
    
    // Check for English letters only (a-z, A-Z, 0-9)
    if (!/^[a-zA-Z0-9]+$/.test(password)) {
      errors.push('Password must contain only English letters and numbers')
    }
    
    // Check for both letters and numbers
    if (!/[a-zA-Z]/.test(password)) {
      errors.push('Password must contain at least one letter')
    }
    if (!/[0-9]/.test(password)) {
      errors.push('Password must contain at least one number')
    }
    
    // Check for repetitive patterns (same character repeated)
    if (/^(.)\1{7,}$/.test(password)) {
      errors.push('Password cannot be the same character repeated')
    }
    
    // Check for simple patterns
    const simplePatterns = [
      /^[a]{8,}$/i,  // All 'a's
      /^[0]{8,}$/,   // All '0's
      /^12345678/,   // Sequential numbers
      /^abcdefgh/i,  // Sequential letters
      /^qwertyui/i,  // Keyboard pattern
      /^password/i,  // Common password
      /^username/i   // Username as password
    ]
    
    for (const pattern of simplePatterns) {
      if (pattern.test(password)) {
        errors.push('Password is too simple or common')
        break
      }
    }
    
    return errors
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    
    setFormData({
      ...formData,
      [name]: value
    })
    
    // Validate password in real-time
    if (name === 'password') {
      const errors = validatePassword(value)
      setPasswordErrors(errors)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    // Validate password
    const passwordValidationErrors = validatePassword(formData.password)
    if (passwordValidationErrors.length > 0) {
      setMessage('Please fix password requirements')
      setPasswordErrors(passwordValidationErrors)
      return
    }
    
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
        // Redirect to main page
        window.location.href = '/main'
      } else {
        setMessage(result.error || 'Registration failed')
      }
    } catch (error) {
      setMessage('Network or API gateway error. Please try again.')
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
          className={passwordErrors.length > 0 ? 'error' : ''}
        />
        {passwordErrors.length > 0 && (
          <div className="password-requirements">
            <p><strong>Password Requirements:</strong></p>
            <ul>
              <li className={formData.password.length >= 8 ? 'valid' : 'invalid'}>
                At least 8 characters
              </li>
              <li className={/^[a-zA-Z0-9]+$/.test(formData.password) ? 'valid' : 'invalid'}>
                English letters and numbers only
              </li>
              <li className={/[a-zA-Z]/.test(formData.password) ? 'valid' : 'invalid'}>
                At least one letter
              </li>
              <li className={/[0-9]/.test(formData.password) ? 'valid' : 'invalid'}>
                At least one number
              </li>
              <li className={!/^(.)\1{7,}$/.test(formData.password) && !/^[a]{8,}$/i.test(formData.password) && !/^[0]{8,}$/.test(formData.password) ? 'valid' : 'invalid'}>
                Not too simple (no repeated characters)
              </li>
            </ul>
          </div>
        )}
      </div>

      <div className="form-group">
        <label>Confirm Password</label>
        <input 
          type="password" 
          name="confirmPassword"
          value={formData.confirmPassword}
          onChange={handleChange}
          required
          className={formData.password !== formData.confirmPassword && formData.confirmPassword !== '' ? 'error' : ''}
        />
        {formData.password !== formData.confirmPassword && formData.confirmPassword !== '' && (
          <div className="error-text">Passwords do not match</div>
        )}
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
            disabled={loading || passwordErrors.length > 0}
          >
            {loading ? 'Registering...' : 'Register'}
          </button>
        </div>
      </div>
    </form>
  )
}

export default RegisterForm