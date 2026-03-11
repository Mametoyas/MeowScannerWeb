import React from "react"

const Navbar: React.FC = () => {
  return (
    <div className="navbar">
      <div className="logo">
        <img src="/images/logo.png" alt="Logo" />
      </div>

      <div className="nav-links">
        <a href="/login" className="login-link">เข้าสู่ระบบ</a>
        <a href="/register" className="register-link">สร้างบัญชี</a>
      </div>
    </div>
  )
}

export default Navbar