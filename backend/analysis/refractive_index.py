import cv2
import numpy as np
from scipy import signal
from typing import Dict, Tuple, List

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess the image for refractive index analysis."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    return enhanced

def calculate_reflectance(image: np.ndarray, angle: float) -> float:
    """
    Calculate reflectance at a given angle using Fresnel equations.
    Returns a normalized reflectance value between 0 and 1.
    """
    # Convert angle to radians
    theta = np.radians(angle)
    
    # Calculate average intensity in the region of interest
    intensity = np.mean(image)
    
    # Normalize intensity to 0-1 range
    normalized_intensity = intensity / 255.0
    
    # Simple Fresnel reflectance approximation
    # This is a simplified model - in practice, you'd need more sophisticated calculations
    n_air = 1.0  # refractive index of air
    n_glass = 1.5  # approximate refractive index of glass
    
    # Calculate Fresnel reflectance
    r_s = ((n_air * np.cos(theta) - n_glass * np.sqrt(1 - (n_air/n_glass * np.sin(theta))**2)) /
           (n_air * np.cos(theta) + n_glass * np.sqrt(1 - (n_air/n_glass * np.sin(theta))**2)))**2
    
    r_p = ((n_air * np.sqrt(1 - (n_air/n_glass * np.sin(theta))**2) - n_glass * np.cos(theta)) /
           (n_air * np.sqrt(1 - (n_air/n_glass * np.sin(theta))**2) + n_glass * np.cos(theta)))**2
    
    theoretical_reflectance = (r_s + r_p) / 2
    
    # Compare measured intensity with theoretical reflectance
    reflectance_score = 1.0 - abs(normalized_intensity - theoretical_reflectance)
    
    return max(0.0, min(1.0, reflectance_score))

def analyze_refractive_index(image: np.ndarray, angle: float) -> Dict[str, float]:
    """
    Analyze the refractive index of a lens from an image.
    Returns a dictionary containing analysis results and confidence score.
    """
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Calculate reflectance
    reflectance_score = calculate_reflectance(processed_image, angle)
    
    # Calculate confidence score based on image quality
    # This is a simplified confidence calculation
    confidence = np.std(processed_image) / 255.0  # Higher variance = lower confidence
    
    return {
        "reflectance_score": reflectance_score,
        "confidence": confidence,
        "estimated_refractive_index": 1.5,  # This would be calculated in a real implementation
        "quality_score": reflectance_score * (1.0 - confidence)
    }
