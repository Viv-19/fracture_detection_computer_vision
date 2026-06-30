import React, { useState } from 'react';
import axios from 'axios';
import { Activity, RefreshCw } from 'lucide-react';
import UploadArea from './components/UploadArea';
import Results from './components/Results';
import './index.css';

const API_URL = 'http://localhost:8001';

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
    } catch (err) {
      console.error('Analysis failed:', err);
      setError(
        err.response?.data?.detail || 'Failed to analyze image. Please ensure the backend is running.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="container">
      <header className="app-header">
        <Activity size={40} color="var(--primary-color)" />
        <h1 className="app-title">Bone Fracture AI Assistant</h1>
      </header>

      <main className="main-layout">
        <div className="left-column" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {!file ? (
            <UploadArea onFileSelect={handleFileSelect} />
          ) : (
            <div className="glass-card">
              <h2 className="results-header" style={{ marginBottom: '1.5rem' }}>Selected Image</h2>
              <div className="preview-container">
                <img src={previewUrl} alt="Selected X-ray" className="preview-image" />
                <div className="action-buttons">
                  <button 
                    className="btn btn-secondary" 
                    onClick={handleReset}
                    disabled={loading}
                  >
                    <RefreshCw size={18} />
                    Change Image
                  </button>
                  <button 
                    className="btn btn-primary" 
                    onClick={handleAnalyze}
                    disabled={loading}
                  >
                    {loading ? (
                      <>
                        <div className="spinner"><RefreshCw size={18} /></div>
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Activity size={18} />
                        Analyze Image
                      </>
                    )}
                  </button>
                </div>
                {error && (
                  <div style={{ marginTop: '1rem', padding: '0.75rem', backgroundColor: '#fef2f2', color: '#dc2626', borderRadius: '0.5rem', width: '100%', textAlign: 'center' }}>
                    {error}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="right-column">
          <Results result={result} previewUrl={result ? previewUrl : null} />
        </div>
      </main>

      <footer style={{ marginTop: '4rem', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
        <p>Medical Disclaimer: This AI tool is designed to assist medical professionals and does not replace clinical judgment.</p>
      </footer>
    </div>
  );
}

export default App;
