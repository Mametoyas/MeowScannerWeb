import React from "react";
import { PredictionData } from "@/components/predict/types/predict"; // ระวัง Path ตรงนี้นะครับ ให้ชี้ไปที่ไฟล์ types ของคุณ

interface PredictResultProps {
  data: PredictionData | null;
}

export default function PredictResult({ data }: PredictResultProps) {
  if (!data) {
    return (
      <div className="result-box">
        <p className="result-text">กำลังวิเคราะห์ข้อมูลน้องแมว...</p>
      </div>
    );
  }

  return (
    <div className="result-box">
      <p className="result-text">จำนวนแมว : {data.catCount} ตัว</p>
      <p className="result-text">พันธุ์แมว : {data.breed}</p>
      <p className="result-text">จุดเด่น : {data.features}</p>
      <p className="result-text">ค่าความมั่นใจ : {data.confidence}%</p>
    </div>
  );
}