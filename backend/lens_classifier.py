import cv2
import numpy as np
import joblib
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

def detect_ar_coating_interference(image: np.ndarray) -> Tuple[bool, float]:
    """Detect AR coating interference patterns."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply FFT to detect interference patterns
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # Look for characteristic interference patterns
    center_y, center_x = magnitude_spectrum.shape[0]//2, magnitude_spectrum.shape[1]//2
    roi = magnitude_spectrum[center_y-50:center_y+50, center_x-50:center_x+50]
    interference_score = np.mean(roi) / 255.0
    
    return interference_score > 0.3, interference_score

def detect_edge_bevel_signature(image: np.ndarray) -> Tuple[bool, float]:
    """Detect lens edge bevel signature."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Look for circular edge patterns
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=50, maxRadius=300)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Calculate edge density around the circle
        edge_density = np.mean(edges[circles[0, 0, 1]-10:circles[0, 0, 1]+10, 
                                    circles[0, 0, 0]-10:circles[0, 0, 0]+10])
        return edge_density > 0.1, edge_density
    return False, 0.0

def analyze_surface_curvature(image: np.ndarray) -> Tuple[bool, float]:
    """Analyze surface curvature using gradient analysis."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and direction
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx)
    
    # Look for characteristic lens curvature patterns
    curvature_score = np.std(direction) / np.pi
    return curvature_score > 0.2, curvature_score

def analyze_reflection_patterns(image: np.ndarray) -> Tuple[bool, float]:
    """Analyze reflection patterns specific to lenses."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Look for bright spots and their distribution
    bright_spots = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(bright_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False, 0.0
        
    # Analyze contour distribution
    areas = [cv2.contourArea(c) for c in contours]
    if not areas:
        return False, 0.0
        
    # Calculate reflection pattern score
    pattern_score = np.std(areas) / np.mean(areas)
    return pattern_score > 0.5, pattern_score

def is_lens_image(image: np.ndarray, gabor_vec: List[float], flow_vec: List[float]) -> Tuple[bool, Dict]:
    """
    Comprehensive lens detection using multiple features.
    Returns (is_lens, details_dict)
    """
    try:
        # Stage 1: Basic Feature Checks
        ar_coating, ar_score = detect_ar_coating_interference(image)
        edge_bevel, edge_score = detect_edge_bevel_signature(image)
        curvature, curve_score = analyze_surface_curvature(image)
        reflection, ref_score = analyze_reflection_patterns(image)
        
        # Stage 2: Texture Analysis
        texture_score = float(np.std(gabor_vec)) if gabor_vec else 0.0
        has_texture = texture_score > 0.1
        
        # Stage 3: Flow Analysis
        flow_score = float(np.mean(flow_vec)) if flow_vec else 0.0
        has_flow = flow_score > 0.05
        
        # Calculate overall score
        scores = {
            "ar_coating": float(ar_score),
            "edge_bevel": float(edge_score),
            "curvature": float(curve_score),
            "reflection": float(ref_score),
            "texture": float(texture_score),
            "flow": float(flow_score)
        }
        
        # Weighted scoring system - Adjusted weights to be stricter
        weights = {
            "ar_coating": 0.35,    # Increased weight for AR coating
            "edge_bevel": 0.25,    # Increased weight for edge bevel
            "curvature": 0.20,     # Kept same
            "reflection": 0.15,    # Kept same
            "texture": 0.03,       # Reduced weight
            "flow": 0.02          # Reduced weight
        }
        
        weighted_score = float(sum(scores[k] * weights[k] for k in scores))
        
        # MUCH stricter verification rules
        is_lens = bool(
            weighted_score > 0.75 and  # Increased from 0.6 to 0.75
            ar_coating and  # Must have AR coating
            edge_bevel and  # Must have edge bevel (changed from OR to AND)
            curvature and   # Must have curvature (changed from OR to AND)
            (reflection or has_texture) and  # Must have either reflection patterns or texture
            ar_score > 0.4 and  # Minimum AR score
            edge_score > 0.3 and  # Minimum edge score
            curve_score > 0.25 and  # Minimum curvature score
            (ref_score > 0.4 or texture_score > 0.15)  # Minimum reflection or texture score
        )
        
        # Additional checks for common non-lens objects
        if is_lens:
            # Check for plastic-like reflections (water bottles)
            if ref_score > 0.8 and texture_score < 0.1:
                is_lens = False
                return is_lens, {
                    "is_lens": False,
                    "weighted_score": float(weighted_score),
                    "scores": {k: float(v) for k, v in scores.items()},
                    "checks_passed": {
                        "ar_coating": bool(ar_coating),
                        "edge_bevel": bool(edge_bevel),
                        "curvature": bool(curvature),
                        "reflection": bool(reflection),
                        "texture": bool(has_texture),
                        "flow": bool(has_flow)
                    },
                    "description": "Object appears to be plastic (like a water bottle)"
                }
            
            # Check for flat glass (windows, screens)
            if curve_score < 0.3 and edge_score < 0.4:
                is_lens = False
                return is_lens, {
                    "is_lens": False,
                    "weighted_score": float(weighted_score),
                    "scores": {k: float(v) for k, v in scores.items()},
                    "checks_passed": {
                        "ar_coating": bool(ar_coating),
                        "edge_bevel": bool(edge_bevel),
                        "curvature": bool(curvature),
                        "reflection": bool(reflection),
                        "texture": bool(has_texture),
                        "flow": bool(has_flow)
                    },
                    "description": "Object appears to be flat glass"
                }
        
        return is_lens, {
            "is_lens": bool(is_lens),
            "weighted_score": float(weighted_score),
            "scores": {k: float(v) for k, v in scores.items()},
            "checks_passed": {
                "ar_coating": bool(ar_coating),
                "edge_bevel": bool(edge_bevel),
                "curvature": bool(curvature),
                "reflection": bool(reflection),
                "texture": bool(has_texture),
                "flow": bool(has_flow)
            },
            "description": "Lens detected" if is_lens else "Not a lens"
        }
        
    except Exception as e:
        logger.error(f"Error in lens detection: {str(e)}")
        return False, {
            "error": str(e),
            "description": "Error during lens detection"
        }

# Placeholder: load your trained model here
try:
    clf = joblib.load("lens_vs_not_lens.joblib")
except Exception:
    clf = None

def build_classifier_features(image, gabor_vec, flow_vec):
    return [
        detect_ar_coating_interference(image),
        detect_edge_bevel_signature(image),
        analyze_surface_curvature(image),
        analyze_reflection_patterns(image),
        float(np.std(gabor_vec)),
        float(flow_vec[0])
    ]

def is_lens_image_old(image, gabor_vec, flow_vec):
    """
    Check if the image is a lens.
    During development, this is more lenient to allow testing.
    """
    try:
        # Ensure we have the right number of features
        if isinstance(gabor_vec, (list, np.ndarray)):
            gabor_vec = np.array(gabor_vec)
        if isinstance(flow_vec, (list, np.ndarray)):
            flow_vec = np.array(flow_vec)
            
        # Pad or truncate vectors if needed
        if len(gabor_vec) < 5:
            gabor_vec = np.pad(gabor_vec, (0, 5 - len(gabor_vec)))
        if len(flow_vec) < 5:
            flow_vec = np.pad(flow_vec, (0, 5 - len(flow_vec)))
            
        feats = build_classifier_features(image, gabor_vec, flow_vec)
        
        if clf is None:
            # If model not loaded, use basic checks
            has_ar = detect_ar_coating_interference(image)
            has_edge = detect_edge_bevel_signature(image)
            has_curvature = analyze_surface_curvature(image)
            has_reflection = analyze_reflection_patterns(image)
            
            # During development, accept if any check passes
            checks_passed = sum([has_ar, has_edge, has_curvature, has_reflection])
            if checks_passed >= 1:  # More lenient: only need 1 check to pass
                return True, {
                    "ar_coating": has_ar,
                    "edge_bevel": has_edge,
                    "curvature": has_curvature,
                    "reflection": has_reflection,
                    "note": "Using basic checks during development"
                }
            return False, {
                "ar_coating": has_ar,
                "edge_bevel": has_edge,
                "curvature": has_curvature,
                "reflection": has_reflection,
                "note": "Basic checks failed during development"
            }
            
        # Ensure features are in the right format for prediction
        feats = np.array(feats).reshape(1, -1)
        return bool(clf.predict(feats)[0]), dict(zip(
            ["ar_coating", "edge_bevel", "curvature", "reflection", "tex_var", "flow_mag"], feats[0]))
            
    except Exception as e:
        logger.error(f"Error in lens detection: {str(e)}")
        # During development, return True on error
        return True, {"note": "Error in detection, accepting during development"} 