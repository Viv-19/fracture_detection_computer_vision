"""
fastapi backend for bone fracture detection
"""

import os 
import logging
from typing import  Optional
from fastapi import FastAPI , File , UploadFile , HTTPException , status
from fastapi.response import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from services.fracture_detection import FractureDetectionService
from utils.logger import setup_logger
from config.settings import Settings

# setup logging
logger = setup_logger(__name__)

# load settings
settings = Settings()

# initialize fastapi app
app = FastAPI(
    title="Bone Fracture Detection API",
    discription = "AI powered bone fracture detection from X-rays",
    version = "1.0.0",
    docs_url = "/docs",
    redoc_url = "/redoc")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

# initialize fracture detection service
fracture_service = FractureDetectionService()

# pydantic models for request and response
class PredictionRequest(BaseModel):
    """ response model for fracture detection"""
    success: bool
    fracture_detected: bool
    confidence: float
    message: str
    processing_time: Optional[float] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """ health check response"""
    status: str
    version: str
    model_loaded: bool

@app.on_event("startup")
async def startup_event():
    """ startup event for the app"""
    
    try:
        await fracture_service.initialize()
        logger.info("Fracture detection service initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize fracture detection service: {e}")
        raise


