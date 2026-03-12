import React from "react"
import Link from "next/link";

interface NavbarProps {
  searchTerm?: string;
  onSearchChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

const Navbar: React.FC<NavbarProps> = ({ searchTerm, onSearchChange }) => {
  return (
    <div className="navbar">
      <div className="logo">
        <Link href="/main">
          <img src="/images/logo.png" alt="Logo" />
        </Link>
      </div>

      <div className="nav-links">
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