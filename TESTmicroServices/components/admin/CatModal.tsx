"use client"
import React, { useState, useEffect } from "react";
import { Cat } from "@/components/search/CatList";

interface CatModalProps {
  cat: Cat | null;
  onClose: () => void;
}

export default function CatModal({ cat, onClose }: CatModalProps) {
  const [formData, setFormData] = useState({
    CatID: "",
    CatName: "",
    CatPersonal: "",
    CatDetails: "",
    ImgURL: ""
  });

  useEffect(() => {
    if (cat) {
      setFormData({
        CatID: cat.CatID,
        CatName: cat.CatName,
        CatPersonal: cat.CatPersonal,
        CatDetails: cat.CatDetails,
        ImgURL: cat.ImgURL || ""
      });
    }
  }, [cat]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      const { DATABASE_API } = await import('../../config/api').then(m => m.getAPIUrls());
      const endpoint = cat ? '/update-cat' : '/add-cat';
      
      const response = await fetch(`${DATABASE_API}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          CatID: formData.CatID,
          CatName: formData.CatName,
          CatPersonal: formData.CatPersonal,
          Cat: formData.CatDetails
        })
      });
      
      if (response.ok) {
        alert(cat ? 'แก้ไขข้อมูลสำเร็จ' : 'เพิ่มข้อมูลสำเร็จ');
        onClose();
      } else {
        alert('เกิดข้อผิดพลาด');
      }
    } catch (error) {
      console.error('Error saving cat:', error);
      alert('เกิดข้อผิดพลาดในการเชื่อมต่อ');
    }
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <div className="modal-header">
          <h2>{cat ? "แก้ไขข้อมูลแมว" : "เพิ่มแมวใหม่"}</h2>
          <button onClick={onClose} className="close-btn">×</button>
        </div>
        
        <form onSubmit={handleSubmit} className="cat-form">
          <div className="form-group">
            <label>รหัสแมว:</label>
            <input
              type="text"
              name="CatID"
              value={formData.CatID}
              onChange={handleChange}
              placeholder="C0001"
              required
              disabled={!!cat}
            />
          </div>

          <div className="form-group">
            <label>ชื่อสายพันธุ์:</label>
            <input
              type="text"
              name="CatName"
              value={formData.CatName}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label>นิสัย/ลักษณะ:</label>
            <textarea
              name="CatPersonal"
              value={formData.CatPersonal}
              onChange={handleChange}
              rows={4}
              required
            />
          </div>

          <div className="form-group">
            <label>รายละเอียด:</label>
            <textarea
              name="CatDetails"
              value={formData.CatDetails}
              onChange={handleChange}
              rows={3}
              required
            />
          </div>

          <div className="form-group">
            <label>URL รูปภาพ:</label>
            <input
              type="text"
              name="ImgURL"
              value={formData.ImgURL}
              onChange={handleChange}
            />
          </div>

          <div className="form-actions">
            <button type="button" onClick={onClose} className="cancel-btn">
              ยกเลิก
            </button>
            <button type="submit" className="save-btn">
              {cat ? "บันทึกการแก้ไข" : "เพิ่มแมว"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}