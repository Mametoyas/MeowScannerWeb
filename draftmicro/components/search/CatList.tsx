"use client"
import { useState, useEffect } from "react";
import CatCard from "./CatCard";

export interface Cat {
  id: number;
  name: string;
  history: string;
  source: string;
  image: string;
}

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

interface CatListProps {
  searchTerm: string;
}

export default function CatList({ searchTerm }: CatListProps) {
  const [cats, setCats] = useState<Cat[]>([]);

  useEffect(() => {
    setCats(mockCats);
  }, []);

  const filteredCats = cats.filter((cat) => 
    cat.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="cat-list">
      {filteredCats.length > 0 ? (
        filteredCats.map((cat) => (
          <CatCard key={cat.id} cat={cat} />
        ))
      ) : (
        <p style={{ textAlign: "center", color: "white" }}>ไม่พบข้อมูลน้องแมวที่คุณค้นหา 😿</p>
      )}
    </div>
  );
}