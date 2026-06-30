import React, { useCallback, useRef } from 'react';
import { Upload, Image as ImageIcon } from 'lucide-react';

export default function UploadArea({ onFileSelect }) {
  const fileInputRef = useRef(null);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('drag-active');
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-active');
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('drag-active');
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please select a valid image file (PNG, JPG, JPEG)');
      return;
    }
    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('File size exceeds 10MB limit');
      return;
    }
    onFileSelect(file);
  };

  return (
    <div className="glass-card">
      <h2 className="results-header" style={{ marginBottom: '1.5rem' }}>
        <ImageIcon size={24} color="var(--primary-color)" />
        Upload X-Ray Image
      </h2>
      <div 
        className="upload-zone"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <Upload className="upload-icon" />
        <div>
          <p className="upload-text">Drag & drop your X-ray here</p>
          <p className="upload-subtext">or click to browse from your computer</p>
        </div>
        <p className="upload-subtext" style={{ fontSize: '0.75rem', marginTop: '1rem' }}>
          Supports PNG, JPG, JPEG (Max 10MB)
        </p>
        <input 
          type="file" 
          className="hidden-input" 
          ref={fileInputRef}
          onChange={handleFileInput}
          accept="image/png, image/jpeg, image/jpg"
        />
      </div>
    </div>
  );
}
