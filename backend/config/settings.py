"""
Configuration settings for the fracture detection application
"""

import os
from typing import Tuple
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Model settings
    model_path: str = "../final_densenet_fracture_model.h5"
    target_size: Tuple[int, int] = (224, 224)
    last_conv_layer: str = "conv5_block16_2_conv"
    
    # File upload settings
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_image_types: list = ["image/jpeg", "image/jpg", "image/png"]
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS settings
    cors_origins: list = ["*"]  # In production, specify exact origins
    
    class Config:
        env_file = ".env"
        case_sensitive = False 