// Auth utility functions
export const authUtils = {
  // Get current user from storage
  getCurrentUser() {
    const user = localStorage.getItem('user') || sessionStorage.getItem('user')
    return user ? JSON.parse(user) : null
  },

  // Check if user is logged in
  isLoggedIn() {
    return this.getCurrentUser() !== null
  },

  // Check if user is admin
  isAdmin() {
    const user = this.getCurrentUser()
    return user && user.role === 'admin'
  },

  // Logout user
  logout() {
    localStorage.removeItem('user')
    sessionStorage.removeItem('user')
    window.location.href = '/login'
  },

  // Require login (redirect if not logged in)
  requireLogin() {
    if (!this.isLoggedIn()) {
      window.location.href = '/login'
      return false
    }
    return true
  },

  // Require admin (redirect if not admin)
  requireAdmin() {
    if (!this.isAdmin()) {
      window.location.href = '/login'
      return false
    }
    return true
  }
}