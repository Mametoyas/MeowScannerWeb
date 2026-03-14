import React from "react"
import Link from "next/link";

interface MainNavbarProps {
  onLogout?: () => void;
  userName?: string;
  userRole?: string;
}

const MainNavbar: React.FC<MainNavbarProps> = ({ onLogout, userName, userRole }) => {
  return (
    <div className="navbar">
      <div className="logo">
        <Link href="/main">
          <img src="/images/LOGO.png" alt="MeowScanner Logo" />
        </Link>
      </div>

      <div className="nav-links">
        {userRole === 'admin' && (
          <Link href="/admin" className="admin-icon-link">
            <div className="admin-icon">⚙️</div>
          </Link>
        )}
        <span className="welcome-text">
          <b>Welcome: {userName || 'User'}</b>
        </span>
        <button onClick={onLogout} className="logout-nav-btn">
          Logout
        </button>
      </div>
    </div>
  )
}

export default MainNavbar