import React from "react"
import Link from "next/link";
import { getUserData } from '../../utils/auth';

interface NavbarProps {
  searchTerm?: string;
  onSearchChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

const Navbar: React.FC<NavbarProps> = ({ searchTerm, onSearchChange }) => {
  const user = getUserData();
  
  return (
    <div className="navbar">
      <div className="logo">
        <Link href="/main">
          <img src="/images/LOGO.png" alt="Logo" />
        </Link>
      </div>

      <div className="nav-links">
        {user?.role === 'admin' && (
          <Link href="/admin" className="admin-icon-link">
            <div className="admin-icon">⚙️</div>
          </Link>
        )}
        <input 
          type="text" 
          placeholder="ค้นหาประเภทแมว..." 
          className="input-search" 
          value={searchTerm || ""} 
          onChange={onSearchChange} 
        />
        <button className="search-btn">Search</button>
      </div>
    </div>
  )
}

export default Navbar