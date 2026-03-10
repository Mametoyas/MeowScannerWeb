import React from "react";

interface PredictImageProps {
  imageUrl?: string;
}

export default function PredictImage({ imageUrl }: PredictImageProps) {
  return (
    <div className="gray-box">
      {imageUrl ? (
        <img src={imageUrl} alt="Predicted Cat" className="cat-center" />
      ) : (
        <p style={{ textAlign: "center", marginTop: "20%" }}>กำลังประมวลผลรูปภาพ...</p>
      )}
    </div>
  );
}