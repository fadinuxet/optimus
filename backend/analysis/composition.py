import cv2
import numpy as np
from scipy import signal
from typing import Dict, List, Tuple

def analyze_spectral_properties(image: np.ndarray) -> Dict[str, float]:
    """Analyze spectral properties of the lens material."""
    # Convert to LAB color space for better material analysis
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Extract L, a, b channels
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    
    # Calculate spectral properties
    l_mean = np.mean(l_channel)
    a_mean = np.mean(a_channel)
    b_mean = np.mean(b_channel)
    
    # Calculate spectral variance
    l_var = np.var(l_channel)
    a_var = np.var(a_channel)
    b_var = np.var(b_channel)
    
    return {
        'l_mean': l_mean,
        'a_mean': a_mean,
        'b_mean': b_mean,
        'l_variance': l_var,
        'a_variance': a_var,
        'b_variance': b_var
    }

def detect_material_type(spectral_props: Dict[str, float]) -> str:
    """Detect the type of lens material based on spectral properties."""
    l_mean = spectral_props['l_mean']
    a_mean = spectral_props['a_mean']
    b_mean = spectral_props['b_mean']
    
    # Simple material classification based on LAB values
    if l_mean > 200 and abs(a_mean) < 5 and abs(b_mean) < 5:
        return "CR-39"  # Standard plastic
    elif l_mean > 180 and a_mean < -5:
        return "Polycarbonate"
    elif l_mean > 190 and b_mean > 5:
        return "Trivex"
    elif l_mean > 195 and abs(a_mean) < 3 and abs(b_mean) < 3:
        return "High-Index 1.67"
    else:
        return "Unknown Material"

def analyze_transparency(image: np.ndarray) -> float:
    """Analyze the transparency of the lens."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate transparency score
    transparency = np.mean(gray) / 255.0
    
    return transparency

def analyze_homogeneity(image: np.ndarray) -> float:
    """Analyze the homogeneity of the lens material."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate local variance
    local_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Convert to homogeneity score (lower variance = higher homogeneity)
    homogeneity = 1.0 - min(1.0, local_var / 1000.0)
    
    return homogeneity

def analyze_composition(image: np.ndarray) -> Dict[str, float]:
    """
    Analyze the composition of a lens from an image.
    Returns a dictionary containing analysis results and confidence score.
    """
    # Analyze spectral properties
    spectral_props = analyze_spectral_properties(image)
    
    # Detect material type
    material_type = detect_material_type(spectral_props)
    
    # Analyze transparency
    transparency = analyze_transparency(image)
    
    # Analyze homogeneity
    homogeneity = analyze_homogeneity(image)
    
    # Calculate material quality score
    material_quality = (transparency + homogeneity) / 2
    
    # Calculate confidence based on spectral properties
    spectral_confidence = 1.0 - (
        spectral_props['l_variance'] +
        spectral_props['a_variance'] +
        spectral_props['b_variance']
    ) / (3 * 255.0)
    
    return {
        "material_type": material_type,
        "transparency": transparency,
        "homogeneity": homogeneity,
        "material_quality": material_quality,
        "spectral_confidence": spectral_confidence,
        "confidence": (material_quality + spectral_confidence) / 2
    } 