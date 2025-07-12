# Bone Fracture Detection - AI Medical Assistant

Industry-grade bone fracture detection system with FastAPI backend and modern frontend.

## 🚀 Quick Start

### 1. Setup Environment
```bash
conda activate ML_DL_venv
```

### 2. Start Backend
```bash
cd backend
python main.py
```
Backend runs on: http://localhost:8000

### 3. Open Frontend
Open `frontend/index.html` in your browser or serve it:
```bash
cd frontend
python -m http.server 8080
```
Frontend available at: http://localhost:8080

## 📁 Project Structure
```
├── backend/                 # FastAPI Backend
│   ├── main.py             # Main application
│   ├── config/             # Settings
│   ├── services/           # Business logic
│   └── utils/              # Utilities
├── frontend/               # HTML/CSS/JS
│   ├── index.html          # Main page
│   ├── styles.css          # Styling
│   └── script.js           # Functionality
└── final_densenet_fracture_model.h5  # AI Model
```

## 🔧 Features
- AI-powered fracture detection
- Modern responsive UI
- Drag & drop file upload
- Real-time processing
- RESTful API with docs

## 📖 API Documentation
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## ⚠️ Medical Disclaimer
This tool assists medical professionals but does not replace clinical judgment. Always consult healthcare providers for diagnosis.