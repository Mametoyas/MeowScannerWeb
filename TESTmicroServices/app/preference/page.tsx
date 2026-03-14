"use client"
import { useState } from "react";
import "@/styles/PreferenceNew.css";
import Navbar from "@/components/predict/Navbar";
import CatImage from "@/components/register/CatImage";
import { dataService } from "@/components/api";
import { authUtils } from "@/components/auth";
import ProtectedRoute from "@/components/ProtectedRoute";
import { getAPIUrls } from "@/config/api";

interface RecommendationResult {
  match_id: string;
  recommended_cat: string;
  why_match: string;
}

interface CatData {
  CatID: string;
  CatName: string;
  CatPersonal: string;
  CatDetails: string;
  Prices?: string;
  ImgURL?: string;
}

export default function QuizPage() {
  const [answers, setAnswers] = useState({
    location: "",
    freeTime: "",
    personality: ""
  });
  const [result, setResult] = useState<RecommendationResult | null>(null);
  const [catData, setCatData] = useState<CatData | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  const { DATABASE_API } = getAPIUrls();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setAnswers(prev => ({ ...prev, [name]: value }));
  };

  const fetchCatData = async (catName: string) => {
    try {
      const response = await fetch(`${DATABASE_API}/get-cat-by-name?cat_name=${encodeURIComponent(catName)}`);
      const data = await response.json();
      if (data.success && data.cat) {
        setCatData(data.cat);
      }
    } catch (error) {
      console.error('Failed to fetch cat data:', error);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setMessage('กำลังค้นหาแมวที่เหมาะกับคุณ...');
    
    try {
      const user = authUtils.getCurrentUser();
      const recommendation = await dataService.getCatRecommendation(
        answers.location,
        answers.freeTime, 
        answers.personality,
        user?.user_id
      );
      
      setResult(recommendation);
      setMessage('พบแมวที่เหมาะกับคุณแล้ว!');
      
      // Fetch cat data for image display
      if (recommendation.recommended_cat) {
        await fetchCatData(recommendation.recommended_cat);
      }
    } catch (error) {
      console.error('Failed to get recommendation:', error);
      setMessage('ไม่สามารถค้นหาแมวได้ กรุณาลองใหม่อีกครั้ง');
    } finally {
      setLoading(false);
    }
  };

  const resetQuiz = () => {
    setAnswers({ location: "", freeTime: "", personality: "" });
    setResult(null);
    setCatData(null);
    setMessage('');
  };

  return (
    <ProtectedRoute>
      <Navbar />

      <div className="quiz-container">
        <div className="quiz-layout">
          
          <div className="quiz-content-wrapper">
            
            {!result && (
              <div className="quiz-image-col">
                <div className="cat-image"><CatImage /></div>
              </div>
            )}

            <div className="quiz-form-col">
              {!result ? (
                <>
                  <h1>คุณเหมาะกับแมวแบบไหน ?</h1>
                  
                  {message && (
                    <div className={`message ${loading ? 'loading' : 'success'}`}>
                      {message}
                    </div>
                  )}

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

                    <button type="submit" className="submit-btn" disabled={loading}>
                      {loading ? 'กำลังค้นหา...' : 'ค้นหาแมวที่เหมาะกับคุณ'}
                    </button>

                  </form>
                </>
              ) : (
                <div className="result-section">
                  <h1>Your Perfect Cat Match!</h1>
                  
                  <div className="result-card">
                    <div className="cat-info">
                      <div className="cat-breed">
                        <h2>{result.recommended_cat}</h2>
                      </div>
                      
                      <div className="match-reason">
                        <h3>เหตุผลที่เหมาะกับคุณ:</h3>
                        <p>{result.why_match}</p>
                      </div>
                      
                      <div className="match-id">
                        <small>Match ID: {result.match_id}</small>
                      </div>
                    </div>
                    
                    <div className="cat-image-section">
                      <div className="cat-image-box">
                        {catData && catData.ImgURL ? (
                          <img 
                            src={catData.ImgURL} 
                            alt={catData.CatName}
                            onError={(e) => {
                              e.currentTarget.src = './images/cat_com.png';
                            }}
                          />
                        ) : (
                          <div className="no-image">
                            <p>No Image</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  <div className="result-actions">
                    <button className="retry-btn" onClick={resetQuiz}>
                      ทำแบบทดสอบใหม่
                    </button>
                    <button className="predict-btn" onClick={() => window.location.href = '/predict'}>
                      ไปสแกนแมว
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </ProtectedRoute>
  );
}