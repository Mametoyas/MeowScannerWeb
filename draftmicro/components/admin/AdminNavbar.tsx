"use client"
import React from "react";

interface AdminNavbarProps {
  searchTerm: string;
  onSearchChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onAddCat: () => void;
}

export default function AdminNavbar({ searchTerm, onSearchChange, onAddCat }: AdminNavbarProps) {
  return (
    <nav className="navbar">
      <div className="logo">
        <img src="/images/logo.png" alt="MeowScanner Logo" />
      </div>
      
      <div className="nav-links">
        <input
          type="text"
          placeholder="ค้นหาสายพันธุ์แมว..."
          value={searchTerm}
          onChange={onSearchChange}
          className="input-search"
        />
        <button onClick={onAddCat} className="add-btn">
          เพิ่มแมวใหม่
        </button>
      </div>
    </nav>
  );
}