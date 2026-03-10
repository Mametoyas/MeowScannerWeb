"use client"
import { useState, useEffect } from "react";
import "../../styles/Predict.css";
// import Navbar from "@/components/Navbar"; // ใส่ Navbar ของคุณ
import PredictImage from "@/components/predict/PredictImage";
import PredictResult from "@/components/predict/PredictResult";
import { PredictionData } from "@/components/predict/types/predict"; // ระวัง Path ตรงนี้นะครับ ให้ชี้ไปที่ไฟล์ types ของคุณ
import Navbar from "@/components/predict/Navbar";

export default function PredictPage() {
  const [prediction, setPrediction] = useState<PredictionData | null>(null);

  useEffect(() => {
    // ข้อมูลจำลองให้ตรงกับรูปภาพของคุณ
    const mockData: PredictionData = {
      imageUrl: "/images/logo.png", // เปลี่ยนเป็นรูปที่มีกรอบเขียวๆ แดงๆ ของคุณ
      catCount: 17,
      breed: "Cat",
      features: "มี 4 ขา 2 ตา 1 หาง",
      confidence: 100
    };
    
    setPrediction(mockData);
  }, []);

  return (
    <>
      <Navbar />

      <div className="predict-container">
        <div className="main-layout">
          <PredictImage imageUrl={prediction?.imageUrl} />
          <PredictResult data={prediction} />
        </div>
      </div>
    </>
  );
}