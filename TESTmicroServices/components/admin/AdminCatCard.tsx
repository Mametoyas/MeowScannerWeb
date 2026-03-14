import React from "react";
import { Cat } from "@/components/search/CatList";

interface AdminCatCardProps {
  cat: Cat;
  onEdit: () => void;
  onDelete: () => void;
}

export default function AdminCatCard({ cat, onEdit, onDelete }: AdminCatCardProps) {
  return (
    <div className="cat-card">
      <div className="cat-info">
        <h3>{cat.CatName}</h3>
        <p>{cat.CatPersonal}</p>
        <hr />
        <small style={{ whiteSpace: "pre-line" }}>{cat.CatDetails}</small>
      </div>
      <div className="cat-image-box">
        <img src={cat.ImgURL || '/images/cat_com.png'} alt={cat.CatName} />
      </div>
      <div className="cat-actions">
        <button onClick={onEdit} className="edit-btn">แก้ไข</button>
        <button onClick={onDelete} className="delete-btn">ลบ</button>
      </div>
    </div>
  );
}