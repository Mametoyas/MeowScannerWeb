"use client"
import { useState, useEffect } from "react";
import "../../styles/Predict.css";
// import Navbar from "@/components/Navbar"; // ใส่ Navbar ของคุณ
import PredictImage from "@/components/predict/PredictImage";
import PredictResult from "@/components/predict/PredictResult";
import { PredictionData } from "@/components/predict/types/predict";
import Navbar from "@/components/predict/Navbar";
import { dataService } from "@/components/api";

export default function PredictPage() {
  const [prediction, setPrediction] = useState<PredictionData | null>(null);

  const handleImageUpload = async (imageFile: File) => {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    const apiUrl = process.env.NODE_ENV === 'production' 
      ? process.env.NEXT_PUBLIC_API_URL || 'https://your-ngrok-url.ngrok.io'
      : 'http://localhost:5000';
    
    try {
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        body: formData
      });
      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      console.error('Prediction failed:', error);
    }
  };

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