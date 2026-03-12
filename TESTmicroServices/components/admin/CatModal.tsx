"use client"
import React, { useState, useEffect } from "react";
import { Cat } from "@/components/search/CatList";

interface CatModalProps {
  cat: Cat | null;
  onClose: () => void;
}

export default function CatModal({ cat, onClose }: CatModalProps) {
  const [formData, setFormData] = useState({
    name: "",
    history: "",
    source: "",
    image: ""
  });

  useEffect(() => {
    if (cat) {
      setFormData({
        name: cat.name,
        history: cat.history,
        source: cat.source,
        image: cat.image
      });
    }
  }, [cat]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Implement save logic
    console.log("Saving cat data:", formData);
    onClose();
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
            <label>ชื่อสายพันธุ์:</label>
            <input
              type="text"
              name="name"
              value={formData.name}
              onChange={handleChange}
              required
            />
          </div>

          <div className="form-group">
            <label>ประวัติ:</label>
            <textarea
              name="history"
              value={formData.history}
              onChange={handleChange}
              rows={4}
              required
            />
          </div>

          <div className="form-group">
            <label>แหล่งอ้างอิง:</label>
            <textarea
              name="source"
              value={formData.source}
              onChange={handleChange}
              rows={3}
              required
            />
          </div>

          <div className="form-group">
            <label>URL รูปภาพ:</label>
            <input
              type="text"
              name="image"
              value={formData.image}
              onChange={handleChange}
              required
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