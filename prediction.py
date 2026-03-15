"""
Prediction utilities
"""
import numpy as np
import tensorflow as tf
from logger import get_logger

logger = get_logger(__name__)

class PredictionEngine:
    """Handles model predictions and results"""
    
    # Class names for 12-class classification
    CLASS_NAMES = [
        "Avulsion Fracture",
        "Comminuted Fracture",
        "Fracture Dislocation",
        "Greenstick Fracture",
        "Hairline Fracture",
        "Impacted Fracture",
        "Longitudinal Fracture",
        "Normal",
        "Oblique Fracture",
        "Pathological Fracture",
        "Spiral Fracture",
        "Transverse Fracture"
    ]
    
    def __init__(self, model_path):
        """
        Initialize prediction engine with model
        
        Args:
            model_path: Path to the saved model
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the model from disk"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict(self, image_array):
        """
        Make prediction on image
        
        Args:
            image_array: Preprocessed image array (batch format)
        
        Returns:
            dict: Prediction results with class, confidence, and probabilities
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            # Get predictions
            predictions = self.model.predict(image_array, verbose=0)[0]
            
            # Get top prediction
            pred_class_idx = int(np.argmax(predictions))
            confidence = float(predictions[pred_class_idx])
            pred_class = self.CLASS_NAMES[pred_class_idx]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            top_predictions = [
                {
                    'class': self.CLASS_NAMES[idx],
                    'confidence': float(predictions[idx]) * 100
                }
                for idx in top_3_indices
            ]
            
            result = {
                'class': pred_class,
                'class_index': pred_class_idx,
                'confidence': confidence * 100,
                'confidence_score': confidence,
                'probabilities': {self.CLASS_NAMES[i]: float(predictions[i]) * 100 
                                  for i in range(len(self.CLASS_NAMES))},
                'top_3_predictions': top_predictions
            }
            
            logger.info(f"Prediction successful: {pred_class} ({confidence*100:.2f}%)")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError(f"Prediction error: {str(e)}")
    
    def get_class_names(self):
        """Get list of all class names"""
        return self.CLASS_NAMES
