"use client"
import { useState } from "react";
import "../../styles/Search.css";
// Import Navbar ตัวที่คุณใช้อยู่เป็นประจำ
import Navbar from "@/components/search/Navbar"; 
import CatList from "@/components/search/CatList";

export default function SearchPage() {
  const [searchTerm, setSearchTerm] = useState<string>("");

  // ฟังก์ชันจัดการเมื่อมีการพิมพ์ในช่องค้นหา
  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  };

  return (
    <>
      {/* ส่ง Props ลงไปให้ Navbar */}
      <Navbar 
        searchTerm={searchTerm} 
        onSearchChange={handleSearchChange} 
      />

      <div className="search-container">
        {/* ส่งคำค้นหาไปกรองข้อมูลใน CatList */}
        <CatList searchTerm={searchTerm} />
      </div>
    </>
  );
}