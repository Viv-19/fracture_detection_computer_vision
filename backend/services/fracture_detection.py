"""
Fracture Detection Service
Handles image processing and model prediction
"""

import os
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional
from fastapi import UploadFile
from skimage import exposure

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model, Model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from utils.logger import setup_logger
from config.settings import Settings

logger = setup_logger(__name__)

class FractureDetectionService:
    """Service for bone fracture detection"""
    
    def __init__(self):
        self.settings = Settings()
        self.model = None
        self.model_loaded = False
        
    async def initialize(self) -> None:
        """Initialize the model"""
        try:
            if not TENSORFLOW_AVAILABLE:
                logger.warning("TensorFlow not available. Running in demo mode.")
                return
                
            model_path = os.path.join(os.path.dirname(__file__), "..", self.settings.model_path)
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                self.model_loaded = True
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model file not found at {model_path}. Running in demo mode.")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_loaded and self.model is not None
    
    def preprocess_xray(self, img: np.ndarray) -> np.ndarray:
        """Enhanced X-ray image preprocessing pipeline"""
        try:
            # Convert to RGB if grayscale
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Gamma correction
            img = exposure.adjust_gamma(img, gamma=0.7)
            
            # Noise reduction
            img = cv2.GaussianBlur(img, (3, 3), 1)
            
            # CLAHE contrast enhancement
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Edge enhancement
            blurred = cv2.GaussianBlur(img, (5, 5), 2.0)
            img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
            
            # Edge detection
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_map = np.sqrt(sobelx**2 + sobely**2)
            edge_map = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX)
            edge_map = cv2.cvtColor(edge_map.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            img = cv2.addWeighted(img, 0.7, edge_map, 0.3, 0)
            
            return img.astype(np.float32) / 255.0
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise
    
    def generate_gradcam(self, img: np.ndarray, layer_name: str) -> np.ndarray:
        """Generate Grad-CAM heatmap for fracture localization"""
        try:
            if not self.is_model_loaded():
                # Return a demo heatmap when model is not available
                heatmap = np.random.rand(img.shape[0], img.shape[1]) * 255
                return heatmap.astype(np.uint8)
            
            # Create model that maps input to conv layer output + predictions
            grad_model = Model(
                inputs=self.model.inputs,
                outputs=[self.model.get_layer(layer_name).output, self.model.output]
            )
            
            # Compute gradient of top predicted class
            with tf.GradientTape() as tape:
                conv_output, preds = grad_model(np.expand_dims(img, axis=0))
                pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]
            
            # Gradient calculation
            grads = tape.gradient(class_channel, conv_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Generate heatmap
            conv_output = conv_output[0].numpy()
            pooled_grads = pooled_grads.numpy()
            for i in range(pooled_grads.shape[-1]):
                conv_output[:, :, i] *= pooled_grads[i]
            
            heatmap = np.mean(conv_output, axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap) + 1e-8
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            
            return heatmap
            
        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {e}")
            # Return empty heatmap on error
            return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    async def predict(self, file: UploadFile) -> Dict[str, Any]:
        """Process image and make prediction"""
        start_time = time.time()
        
        try:
            # Read image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Invalid image file")
            
            # Resize image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.settings.target_size)
            
            # Preprocess image
            processed_img = self.preprocess_xray(img)
            
            # Make prediction
            if self.is_model_loaded():
                img_array = np.expand_dims(processed_img, axis=0)
                fracture_prob = self.model.predict(img_array, verbose=0)[0][0]
                fracture_detected = fracture_prob > 0.5
                confidence = float(fracture_prob if fracture_detected else 1 - fracture_prob)
                message = "Fracture detected" if fracture_detected else "No fracture detected"
            else:
                # Demo mode
                fracture_prob = 0.3
                fracture_detected = False
                confidence = 0.7
                message = "Demo mode - Model not available"
            
            processing_time = time.time() - start_time
            
            return {
                'fracture_detected': fracture_detected,
                'confidence': confidence,
                'message': message,
                'processing_time': processing_time,
                'original_probability': float(fracture_prob)
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise 