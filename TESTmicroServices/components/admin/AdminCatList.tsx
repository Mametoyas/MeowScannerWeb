"use client"
import { useState, useEffect } from "react";
import AdminCatCard from "./AdminCatCard";
import { Cat } from "@/components/search/CatList";

interface AdminCatListProps {
  searchTerm: string;
  onEditCat: (cat: Cat) => void;
}

export default function AdminCatList({ searchTerm, onEditCat }: AdminCatListProps) {
  const [cats, setCats] = useState<Cat[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchCats();
  }, []);

  const fetchCats = async () => {
    try {
      setLoading(true);
      const { DATABASE_API } = await import('../../config/api').then(m => m.getAPIUrls());
      const response = await fetch(`${DATABASE_API}/get-cats`);
      const data = await response.json();
      
      if (response.ok && data.success) {
        setCats(data.cats || []);
      } else {
        setError('Failed to fetch cat data');
      }
    } catch (error) {
      console.error('Error fetching cats:', error);
      setError('Error connecting to database');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteCat = async (id: string) => {
    if (!confirm("คุณแน่ใจหรือไม่ที่จะลบข้อมูลแมวนี้?")) return;
    
    try {
      const { DATABASE_API } = await import('../../config/api').then(m => m.getAPIUrls());
      const response = await fetch(`${DATABASE_API}/delete-cat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cat_id: id })
      });
      
      if (response.ok) {
        setCats(cats.filter(cat => cat.CatID !== id));
        alert('ลบข้อมูลสำเร็จ');
      } else {
        alert('เกิดข้อผิดพลาดในการลบข้อมูล');
      }
    } catch (error) {
      console.error('Error deleting cat:', error);
      alert('เกิดข้อผิดพลาดในการเชื่อมต่อ');
    }
  };

  const filteredCats = cats.filter((cat) => 
    cat.CatName.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (loading) {
    return (
      <div className="cat-list">
        <p style={{ textAlign: "center", color: "white" }}>🐱 กำลังโหลดข้อมูล...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="cat-list">
        <p style={{ textAlign: "center", color: "red" }}>❌ {error}</p>
        <button onClick={fetchCats} style={{ display: "block", margin: "10px auto" }}>
          🔄 ลองอีกครั้ง
        </button>
      </div>
    );
  }

  return (
    <div className="cat-list">
      {filteredCats.length > 0 ? (
        filteredCats.map((cat) => (
          <AdminCatCard 
            key={cat.CatID} 
            cat={cat} 
            onEdit={() => onEditCat(cat)}
            onDelete={() => handleDeleteCat(cat.CatID)}
          />
        ))
      ) : (
        <p style={{ textAlign: "center", color: "white" }}>ไม่พบข้อมูลน้องแมวที่คุณค้นหา 😿</p>
      )}
    </div>
  );
}