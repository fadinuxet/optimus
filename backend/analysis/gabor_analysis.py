import cv2
import numpy as np
from typing import Tuple, List, Dict
import hashlib
from fastapi import HTTPException
from fastapi.responses import JSONResponse
# from deep_metric import train_triplet

def create_gabor_filters(orientations=8, scales=5, ksize=31):
    filters = []
    for scale in range(scales):
        for orientation in range(orientations):
            theta = orientation * np.pi / orientations
            sigma = 4.0 * (2 ** scale)
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma,
                theta,
                10.0,
                0.5,
                0,
                ktype=cv2.CV_32F
            )
            filters.append(kernel)
    return filters

def extract_gabor_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    filters = create_gabor_filters(orientations=8, scales=5)
    stats = []
    for kern in filters:
        resp = cv2.filter2D(gray, cv2.CV_32F, kern)
        stats.append(resp.mean())
        stats.append(resp.std())
    return np.array(stats, dtype=np.float32)  # (80,)

def generate_texture_fingerprint(image: np.ndarray) -> Tuple[str, np.ndarray]:
    """
    Generate a unique texture fingerprint for an image using Gabor features.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (fingerprint hash, feature vector)
    """
    # Create Gabor filters
    filters = create_gabor_filters()
    
    # Extract features
    features = extract_gabor_features(image)
    
    # Generate hash
    feature_bytes = features.tobytes()
    fingerprint = hashlib.sha256(feature_bytes).hexdigest()
    
    return fingerprint, features

def compare_texture_fingerprints(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    Compare two texture fingerprints and return a similarity score.
    
    Args:
        features1: Feature vector of first image
        features2: Feature vector of second image
        
    Returns:
        Similarity score between 0 and 1
    """
    # Normalize features
    features1_norm = features1 / np.linalg.norm(features1)
    features2_norm = features2 / np.linalg.norm(features2)
    
    # Compute cosine similarity
    similarity = np.dot(features1_norm, features2_norm)
    
    return float(similarity)

def analyze_texture(image: np.ndarray) -> Dict:
    """
    Analyze the texture of an image and return detailed metrics.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary containing texture analysis results
    """
    # Create Gabor filters with more orientations and scales
    filters = create_gabor_filters(ksize=31, num_orientations=12, num_scales=5)
    
    # Extract features
    features = extract_gabor_features(image)
    
    # Calculate texture metrics
    mean_features = np.mean(features)
    std_features = np.std(features)
    
    # Generate fingerprint
    feature_bytes = features.tobytes()
    fingerprint = hashlib.sha256(feature_bytes).hexdigest()
    
    # Determine texture quality based on feature statistics
    if std_features > 0.8 and np.max(features) > 0.5:
        quality = "High"
        score = 0.9
    elif std_features > 0.5 and np.max(features) > 0.3:
        quality = "Medium"
        score = 0.7
    else:
        quality = "Low"
        score = 0.4
    
    return {
        "fingerprint": fingerprint,
        "quality": quality,
        "score": score,
        "metrics": {
            "mean_features": float(mean_features),
            "std_features": float(std_features),
            "max_features": float(np.max(features)),
            "min_features": float(np.min(features))
        },
        "description": f"Texture analysis shows {quality.lower()} quality with distinct patterns."
    }

def guess_glass_type(discr):
    if discr['ar_coating'] and discr['edge_bevel']:
        return "Likely an optical lens"
    elif discr['ar_coating']:
        return "Coated glass (e.g., camera filter, phone screen)"
    elif discr['edge_bevel']:
        return "Cut glass (e.g., bottle or window edge)"
    elif discr['polarization']:
        return "Polarizing filter or special glass"
    else:
        return "Generic glass (bottle, window, etc.)"

# Test shape (uncomment for dev)
# test_img = np.zeros((128,128), dtype=np.uint8)
# assert extract_gabor_features(test_img).shape == (80,) 

def analyze_image(image: np.ndarray):
    # Placeholder for image analysis logic
    pass

def analyze_image_handler(image: np.ndarray):
    try:
        analysis_result = analyze_texture(image)
        return analysis_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

def analyze_image_handler_with_glass_type(image: np.ndarray):
    try:
        analysis_result = analyze_texture(image)
        glass_type = guess_glass_type(discr)
        return {
            "analysis_result": analysis_result,
            "glass_type": glass_type
        }
    except Exception as e:
        glass_type = guess_glass_type(discr)
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid capture: not a full lens or missing lens features",
                "discriminators": discr,
                "glass_type": glass_type,
                "suggestion": "Include entire lens, ensure AR coating or use polarized frames."
            }
        )

# Remove or comment out this line to prevent automatic training on import
# train_triplet("data/enroll", "model_triplet.pth", epochs=10) 