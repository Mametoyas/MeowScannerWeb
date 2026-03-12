"use client"
import React, { useRef, useState } from "react";

interface PredictImageProps {
  imageUrl?: string;
  onImageUpload?: (file: File) => void;
  onPredict?: () => void;
  loading?: boolean;
}

export default function PredictImage({ imageUrl, onImageUpload, onPredict, loading }: PredictImageProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      if (onImageUpload) {
        onImageUpload(file);
      }
    }
  };

  const handleClick = () => {
    if (!selectedFile) {
      fileInputRef.current?.click();
    }
  };

  const handlePredict = () => {
    if (selectedFile && onPredict) {
      onPredict();
    }
  };

  const handleNewImage = () => {
    setSelectedFile(null);
    if (onImageUpload) {
      onImageUpload(null as any);
    }
    fileInputRef.current?.click();
  };

  return (
    <div className="predict-image-container">
      <div className="gray-box" onClick={handleClick}>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/png,image/jpeg,image/jpg"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
        {imageUrl ? (
          <img 
            src={imageUrl} 
            alt="Selected Cat" 
            className="cat-center"
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          />
        ) : (
          <div className="upload-prompt">
            <p>📷 Click to upload image</p>
            <p>Upload Cat Image</p>
            <p style={{ fontSize: '0.9rem', opacity: 0.8 }}>Supports PNG, JPG, JPEG</p>
          </div>
        )}
      </div>
      
      {selectedFile && (
        <div className="predict-controls">
          <button 
            className="predict-btn"
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? '🔄 Processing...' : '🔍 Predict Cat Breed'}
          </button>
          <button 
            className="change-image-btn"
            onClick={handleNewImage}
            disabled={loading}
          >
            📷 Change Image
          </button>
        </div>
      )}
    </div>
  );
}