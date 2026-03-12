import React from "react";
import { Cat } from "./CatList";

interface CatCardProps {
  cat: Cat;
}

export default function CatCard({ cat }: CatCardProps) {
  return (
    <div className="cat-card">
      <div className="cat-info">
        <div className="cat-header">
          <h3>{cat.CatName}</h3>
          <span className="cat-id">{cat.CatID}</span>
        </div>
        
        <div className="cat-personality">
          <h4>🐾 Personality & Appearance</h4>
          <p>{cat.CatPersonal}</p>
        </div>
        
        <div className="cat-details">
          <h4>📜 History & Details</h4>
          <p>{cat.CatDetails}</p>
        </div>
      </div>
      
      <div className="cat-image-box">
        {cat.ImgURL ? (
          <img src={cat.ImgURL} alt={cat.CatName} />
        ) : (
          <div className="no-image">
            <span>🐱</span>
            <p>No Image</p>
          </div>
        )}
      </div>
    </div>
  );
}