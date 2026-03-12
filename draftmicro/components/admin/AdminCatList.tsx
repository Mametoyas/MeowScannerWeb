"use client"
import { useState, useEffect } from "react";
import AdminCatCard from "./AdminCatCard";
import { Cat } from "@/components/search/CatList";

const mockCats: Cat[] = [
  {
    id: 1,
    name: "วิเชียรมาศ (Siamese)",
    history: "ประวัติ: แมววิเชียรมาศเป็นแมวไทยโบราณที่มีต้นกำเนิดในสมัยอยุธยา ปรากฏหลักฐานในสมุดข่อย...",
    source: "The International Cat Association (TICA)\nสมุดข่อยโบราณ (Tamra Maew)",
    image: "/images/logo.png"
  },
  {
    id: 2,
    name: "สก็อตติช โฟลด์ (Scottish Fold)",
    history: "ประวัติ: ต้นกำเนิดของสายพันธุ์นี้เริ่มต้นในปี 1961 ที่สกอตแลนด์ เมื่อมีการค้นพบลูกแมวสีขาวชื่อ 'Susie'...",
    source: "The Cat Fanciers' Association (CFA)\nEncyclopedia of Cat Breeds",
    image: "/images/logo.png"
  }
];

interface AdminCatListProps {
  searchTerm: string;
  onEditCat: (cat: Cat) => void;
}

export default function AdminCatList({ searchTerm, onEditCat }: AdminCatListProps) {
  const [cats, setCats] = useState<Cat[]>([]);

  useEffect(() => {
    setCats(mockCats);
  }, []);

  const handleDeleteCat = (id: number) => {
    if (confirm("คุณแน่ใจหรือไม่ที่จะลบข้อมูลแมวนี้?")) {
      setCats(cats.filter(cat => cat.id !== id));
    }
  };

  const filteredCats = cats.filter((cat) => 
    cat.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="cat-list">
      {filteredCats.length > 0 ? (
        filteredCats.map((cat) => (
          <AdminCatCard 
            key={cat.id} 
            cat={cat} 
            onEdit={() => onEditCat(cat)}
            onDelete={() => handleDeleteCat(cat.id)}
          />
        ))
      ) : (
        <p style={{ textAlign: "center", color: "white" }}>ไม่พบข้อมูลน้องแมวที่คุณค้นหา 😿</p>
      )}
    </div>
  );
}