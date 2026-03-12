"use client"
import { useState } from "react";
import "@/styles/Preference.css"; // ชี้ Path ไปที่ไฟล์ CSS ให้ถูกต้อง
import Navbar from "@/components/predict/Navbar"; // เรียกใช้ Navbar ตัวเดียวกับ PredictPage
import CatImage from "@/components/register/CatImage";

export default function QuizPage() {
  // State เก็บคำตอบ
  const [answers, setAnswers] = useState({
    location: "",
    freeTime: "",
    personality: ""
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setAnswers(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log("คำตอบที่คุณเลือก:", answers);
    alert("ค้นหาแมวที่เหมาะกับคุณสำเร็จ!");
  };

  return (
    <>
      {/* 1. แสดง Navbar ด้านบนสุด แบบเดียวกับ PredictPage */}
      <Navbar />

      {/* 2. Container สำหรับจัดกลาง */}
      <div className="quiz-container">
        <div className="quiz-layout">
          
          {/* ส่วนจัดหน้าใหม่: รูปแมวด้านซ้าย ฟอร์มด้านขวา */}
          <div className="quiz-content-wrapper">
            
            {/* คอลัมน์ด้านซ้ายสำหรับรูปภาพ */}
            <div className="quiz-image-col">
              <div className="cat-image"><CatImage /></div>
            </div>

            {/* คอลัมน์ด้านขวาสำหรับหัวข้อและฟอร์ม */}
            <div className="quiz-form-col">
              <h1>คุณเหมาะกับแมวแบบไหน ?</h1>

              <form onSubmit={handleSubmit}>
                
                <div className="question-group">
                  <p>คุณอาศัยอยู่ที่ไหน</p>
                  <div className="radio-options">
                    <label><input type="radio" name="location" value="A" onChange={handleChange} required /> A. คอนโด</label>
                    <label><input type="radio" name="location" value="B" onChange={handleChange} /> B. บ้าน</label>
                    <label><input type="radio" name="location" value="C" onChange={handleChange} /> C. บ้านสวน</label>
                  </div>
                </div>

                <div className="question-group">
                  <p>โดยปกติคุณมีเวลาว่างหรือไม่</p>
                  <div className="radio-options">
                    <label><input type="radio" name="freeTime" value="A" onChange={handleChange} required /> A. มีเวลา</label>
                    <label><input type="radio" name="freeTime" value="B" onChange={handleChange} /> B. เวลาน้อย</label>
                    <label><input type="radio" name="freeTime" value="C" onChange={handleChange} /> C. WFH</label>
                  </div>
                </div>

                <div className="question-group">
                  <p>คุณอยากเห็นเจ้าเหมียวมีลักษณะนิสัยอย่างไร</p>
                  <div className="radio-options">
                    <label><input type="radio" name="personality" value="A" onChange={handleChange} required /> A. นิ่งๆ</label>
                    <label><input type="radio" name="personality" value="B" onChange={handleChange} /> B. ซน</label>
                    <label><input type="radio" name="personality" value="C" onChange={handleChange} /> C. ติดหนึบ</label>
                  </div>
                </div>

                <button type="submit" className="submit-btn">
                  ค้นหาแมวที่เหมาะกับคุณ
                </button>

              </form>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}