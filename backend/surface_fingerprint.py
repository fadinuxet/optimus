import sys
print("PYTHONPATH:", sys.path)
import cv2
import numpy as np
from typing import Dict, List, Tuple
import hashlib
from .gabor_analysis import analyze_texture
from .specular_flow import analyze_specular_flow

def generate_surface_fingerprint(image: np.ndarray, frames: List[np.ndarray] = None) -> Dict:
    """
    Generate a comprehensive surface fingerprint combining texture and flow features.
    
    Args:
        image: Main image for texture analysis
        frames: Optional list of frames for flow analysis
        
    Returns:
        Dictionary containing combined surface analysis results
    """
    # Analyze texture
    texture_results = analyze_texture(image)
    
    # Analyze flow if frames are provided
    flow_results = None
    if frames is not None:
        flow_results = analyze_specular_flow(frames)
    
    # Combine features
    combined_features = []
    combined_features.extend([
        texture_results["metrics"]["mean_features"],
        texture_results["metrics"]["std_features"]
    ])
    
    if flow_results and "metrics" in flow_results:
        combined_features.extend([
            flow_results["metrics"]["mean_magnitude"],
            flow_results["metrics"]["std_magnitude"],
            flow_results["metrics"]["mean_direction"]
        ])
    
    # Generate combined fingerprint
    feature_bytes = np.array(combined_features).tobytes()
    fingerprint = hashlib.sha256(feature_bytes).hexdigest()
    
    # Calculate combined score
    if flow_results:
        combined_score = (texture_results["score"] + flow_results["score"]) / 2
    else:
        combined_score = texture_results["score"]
    
    # Determine overall quality
    if combined_score > 0.8:
        quality = "High"
    elif combined_score > 0.6:
        quality = "Medium"
    else:
        quality = "Low"
    
    return {
        "fingerprint": fingerprint,
        "quality": quality,
        "score": combined_score,
        "texture_analysis": texture_results,
        "flow_analysis": flow_results,
        "description": f"Surface analysis shows {quality.lower()} quality with distinct patterns and characteristics."
    }

def compare_surface_fingerprints(fingerprint1: Dict, fingerprint2: Dict) -> float:
    """
    Compare two surface fingerprints and return a similarity score.
    
    Args:
        fingerprint1: First surface fingerprint
        fingerprint2: Second surface fingerprint
        
    Returns:
        Similarity score between 0 and 1
    """
    # Compare texture features
    texture_similarity = 0.0
    if "texture_analysis" in fingerprint1 and "texture_analysis" in fingerprint2:
        texture1 = fingerprint1["texture_analysis"]
        texture2 = fingerprint2["texture_analysis"]
        
        metric1 = np.array([
            texture1["metrics"]["mean_features"],
            texture1["metrics"]["std_features"]
        ])
        
        metric2 = np.array([
            texture2["metrics"]["mean_features"],
            texture2["metrics"]["std_features"]
        ])
        
        # Normalize and compute similarity
        metric1_norm = metric1 / np.linalg.norm(metric1)
        metric2_norm = metric2 / np.linalg.norm(metric2)
        texture_similarity = float(np.dot(metric1_norm, metric2_norm))
    
    # Compare flow features if available
    flow_similarity = 0.0
    if ("flow_analysis" in fingerprint1 and "flow_analysis" in fingerprint2 and
        "metrics" in fingerprint1["flow_analysis"] and "metrics" in fingerprint2["flow_analysis"]):
        flow1 = fingerprint1["flow_analysis"]
        flow2 = fingerprint2["flow_analysis"]
        
        metric1 = np.array([
            flow1["metrics"]["mean_magnitude"],
            flow1["metrics"]["std_magnitude"],
            flow1["metrics"]["mean_direction"]
        ])
        
        metric2 = np.array([
            flow2["metrics"]["mean_magnitude"],
            flow2["metrics"]["std_magnitude"],
            flow2["metrics"]["mean_direction"]
        ])
        
        # Normalize and compute similarity
        metric1_norm = metric1 / np.linalg.norm(metric1)
        metric2_norm = metric2 / np.linalg.norm(metric2)
        flow_similarity = float(np.dot(metric1_norm, metric2_norm))
    
    # Compute combined similarity
    if flow_similarity > 0:
        return (texture_similarity + flow_similarity) / 2
    else:
        return texture_similarity

def analyze_surface_quality(fingerprint: Dict) -> Dict:
    """
    Analyze the quality of a surface based on its fingerprint.
    
    Args:
        fingerprint: Surface fingerprint dictionary
        
    Returns:
        Dictionary containing quality analysis results
    """
    texture_quality = fingerprint["texture_analysis"]["quality"]
    flow_quality = None
    if fingerprint["flow_analysis"] and "quality" in fingerprint["flow_analysis"]:
        flow_quality = fingerprint["flow_analysis"]["quality"]
    
    # Determine overall quality
    if flow_quality:
        if texture_quality == "High" and flow_quality == "High":
            overall_quality = "Excellent"
            score = 0.95
        elif texture_quality == "High" or flow_quality == "High":
            overall_quality = "Good"
            score = 0.8
        elif texture_quality == "Medium" and flow_quality == "Medium":
            overall_quality = "Average"
            score = 0.6
        else:
            overall_quality = "Poor"
            score = 0.4
    else:
        if texture_quality == "High":
            overall_quality = "Good"
            score = 0.8
        elif texture_quality == "Medium":
            overall_quality = "Average"
            score = 0.6
        else:
            overall_quality = "Poor"
            score = 0.4
    
    return {
        "overall_quality": overall_quality,
        "score": score,
        "texture_quality": texture_quality,
        "flow_quality": flow_quality,
        "description": f"Surface quality analysis shows {overall_quality.lower()} quality with {texture_quality.lower()} texture quality" +
                      (f" and {flow_quality.lower()} flow quality" if flow_quality else "")
    }

def check_capture_quality(image: np.ndarray):
    """
    Checks if the image has acceptable brightness and sharpness.
    - Brightness: mean pixel value in [80, 200]
    - Sharpness: Laplacian variance > 30
    Returns (True, None) if pass, (False, reason) otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    if not (80 <= brightness <= 200):
        return False, f"Image brightness ({brightness:.1f}) is out of range [80, 200]."
    if sharpness < 30:
        return False, f"Image sharpness (Laplacian variance {sharpness:.1f}) is too low."
    return True, None 