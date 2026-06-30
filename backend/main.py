"""
FastAPI Backend for Bone Fracture Detection
Industry-grade implementation with proper error handling and validation
"""

import os
import logging
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from services.fracture_detection import FractureDetectionService
from utils.logger import setup_logger
from config.settings import Settings

# Setup logging
logger = setup_logger(__name__)

# Load settings
settings = Settings()

# Initialize FastAPI app
app = FastAPI(
    title="Bone Fracture Detection API",
    description="AI-powered bone fracture detection from X-ray images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
fracture_service = FractureDetectionService()

# Pydantic models for request/response
class PredictionResponse(BaseModel):
    """Response model for fracture prediction"""
    success: bool
    fracture_detected: bool
    confidence: float
    message: str
    processing_time: Optional[float] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    version: str
    model_loaded: bool

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        await fracture_service.initialize()
        logger.info("Fracture detection service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize fracture detection service: {e}")
        raise

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=fracture_service.is_model_loaded()
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=fracture_service.is_model_loaded()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_fracture(file: UploadFile = File(...)):
    """
    Predict bone fracture from uploaded X-ray image
    
    Args:
        file: X-ray image file (PNG, JPG, JPEG)
    
    Returns:
        PredictionResponse with fracture detection results
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image (PNG, JPG, JPEG)"
            )
        
        # Validate file size (10MB limit)
        if file.size > settings.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds {settings.max_file_size // (1024*1024)}MB limit"
            )
        
        # Process the image
        result = await fracture_service.predict(file)
        
        return PredictionResponse(
            success=True,
            fracture_detected=result['fracture_detected'],
            confidence=result['confidence'],
            message=result['message'],
            processing_time=result.get('processing_time')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return PredictionResponse(
            success=False,
            fracture_detected=False,
            confidence=0.0,
            message="Failed to process image",
            error=str(e)
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )