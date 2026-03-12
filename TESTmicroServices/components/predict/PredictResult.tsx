import React from "react";
import { PredictionData } from "@/components/predict/types/predict";

interface PredictResultProps {
  data: PredictionData | null;
  loading?: boolean;
}

export default function PredictResult({ data, loading }: PredictResultProps) {
  const resultBoxClass = loading ? "result-box loading" : "result-box";

  if (loading) {
    return (
      <div className={resultBoxClass}>
        <p className="result-text">Processing image...</p>
        <p className="result-text">Please wait...</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="result-box">
        <p className="result-text">📸 Upload an image to analyze</p>
        <p className="result-text">Click on the image area to select a cat photo</p>
      </div>
    );
  }

  return (
    <div className="result-box">
      <p className="result-text">จำนวนแมว : {data.catCount} ตัว</p>
      <p className="result-text">พันธุ์แมว : {data.breed}</p>
      
      {/* แสดงจุดเด่นจาก meowdex */}
      {data.catPersonal ? (
        <div className="cat-personal-section">
          <p className="result-text personal-title">จุดเด่น :</p>
          <p className="result-text personal-content">{data.catPersonal}</p>
        </div>
      ) : (
        <p className="result-text">จุดเด่น : {data.features}</p>
      )}
      
      <p className="result-text">ค่าความมั่นใจ : {data.confidence.toFixed(1)}%</p>
    </div>
  );
}