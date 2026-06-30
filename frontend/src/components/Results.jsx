import React from 'react';
import { AlertTriangle, CheckCircle, Clock, Activity } from 'lucide-react';

export default function Results({ result, previewUrl }) {
  if (!result) {
    return (
      <div className="glass-card">
        <h2 className="results-header" style={{ marginBottom: '1.5rem', color: 'var(--text-secondary)' }}>
          Analysis Results
        </h2>
        <div className="placeholder-image">
          <p>Upload an image to see analysis results</p>
        </div>
      </div>
    );
  }

  const isFracture = result.fracture_detected;

  return (
    <div className="glass-card">
      <h2 className="results-header" style={{ marginBottom: '1.5rem' }}>
        Diagnosis Report
      </h2>
      
      <div className="preview-container">
        {/* We would ideally show the Grad-CAM heatmap here if the backend returned it as an image.
            For now, we show the uploaded image, but in a full production system, 
            the backend should return a base64 encoded heatmap image string. */}
        {previewUrl && (
          <img src={previewUrl} alt="X-Ray Analysis" className="preview-image" />
        )}
        
        <div style={{ width: '100%' }}>
          <div className={`result-badge ${isFracture ? 'badge-danger' : 'badge-success'}`}>
            {isFracture ? <AlertTriangle size={20} /> : <CheckCircle size={20} />}
            {result.message || (isFracture ? 'Fracture Detected' : 'No Fracture Detected')}
          </div>
          
          <div className="result-details">
            <div className="detail-item">
              <span className="detail-label"><Activity size={14} style={{ display: 'inline', marginRight: '4px' }}/> Confidence Level</span>
              <span className="detail-value">{(result.confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="detail-item">
              <span className="detail-label"><Clock size={14} style={{ display: 'inline', marginRight: '4px' }}/> Processing Time</span>
              <span className="detail-value">
                {result.processing_time ? `${result.processing_time.toFixed(2)}s` : 'N/A'}
              </span>
            </div>
          </div>
        </div>

        {isFracture && (
          <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#fef2f2', border: '1px solid #fee2e2', borderRadius: '0.5rem', color: '#b91c1c', fontSize: '0.875rem' }}>
            <strong>Note:</strong> This analysis indicates a potential fracture. Please consult a qualified healthcare professional for a formal diagnosis.
          </div>
        )}
      </div>
    </div>
  );
}
