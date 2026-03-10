import React from "react";
import { Cat } from "./CatList";

interface CatCardProps {
  cat: Cat;
}

export default function CatCard({ cat }: CatCardProps) {
  return (
    <div className="cat-card">
      <div className="cat-info">
        <h3>{cat.name}</h3>
        <p>{cat.history}</p>
        <hr />
        <small style={{ whiteSpace: "pre-line" }}>{cat.source}</small>
      </div>
      <div className="cat-image-box">
        <img src={cat.image} alt={cat.name} />
      </div>
    </div>
  );
}