import cv2
import numpy as np
from typing import Tuple, Dict, List
import hashlib

def extract_specular_regions(image: np.ndarray) -> np.ndarray:
    """
    Extract specular (highlight) regions from an image.
    
    Args:
        image: Input image
        
    Returns:
        Binary mask of specular regions
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply adaptive thresholding to find bright regions
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    # Clean up the mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def compute_optical_flow(prev_frame: np.ndarray, curr_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute optical flow between two frames using Farnebäck method.
    
    Args:
        prev_frame: Previous frame
        curr_frame: Current frame
        
    Returns:
        Tuple of (flow_x, flow_y) components
    """
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame
        curr_gray = curr_frame
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        0.5,  # pyr_scale
        3,    # levels
        15,   # winsize
        3,    # iterations
        5,    # poly_n
        1.2,  # poly_sigma
        0     # flags
    )
    
    return flow[..., 0], flow[..., 1]  # Return x and y components

def analyze_specular_flow(frames: List[np.ndarray]) -> Dict:
    """
    Analyze specular flow patterns across a sequence of frames.
    
    Args:
        frames: List of image frames
        
    Returns:
        Dictionary containing flow analysis results
    """
    if len(frames) < 2:
        return {
            "error": "Need at least 2 frames for flow analysis",
            "score": 0.0
        }
    
    # Extract specular regions from first frame
    specular_mask = extract_specular_regions(frames[0])
    
    # Compute flow between consecutive frames
    flow_magnitudes = []
    flow_directions = []
    
    for i in range(len(frames) - 1):
        flow_x, flow_y = compute_optical_flow(frames[i], frames[i + 1])
        
        # Calculate magnitude and direction
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        direction = np.arctan2(flow_y, flow_x)
        
        # Only consider flow in specular regions
        magnitude = magnitude * (specular_mask > 0)
        direction = direction * (specular_mask > 0)
        
        flow_magnitudes.append(magnitude)
        flow_directions.append(direction)
    
    # Calculate flow statistics
    mean_magnitude = np.mean([np.mean(mag) for mag in flow_magnitudes])
    std_magnitude = np.std([np.std(mag) for mag in flow_magnitudes])
    mean_direction = np.mean([np.mean(dir) for dir in flow_directions])
    
    # Generate flow fingerprint
    flow_features = np.concatenate([
        np.mean(flow_magnitudes, axis=0),
        np.std(flow_magnitudes, axis=0),
        np.mean(flow_directions, axis=0)
    ])
    fingerprint = hashlib.sha256(flow_features.tobytes()).hexdigest()
    
    # Determine flow quality
    if std_magnitude > 0.5:
        quality = "High"
        score = 0.9
    elif std_magnitude > 0.2:
        quality = "Medium"
        score = 0.7
    else:
        quality = "Low"
        score = 0.4
    
    return {
        "fingerprint": fingerprint,
        "fingerprint_hash": fingerprint,
        "quality": quality,
        "score": score,
        "metrics": {
            "mean_magnitude": float(mean_magnitude),
            "std_magnitude": float(std_magnitude),
            "mean_direction": float(mean_direction)
        },
        "description": f"Specular flow analysis shows {quality.lower()} quality with {mean_magnitude:.2f} average flow magnitude."
    }

def compare_flow_patterns(flow1: Dict, flow2: Dict) -> float:
    """
    Compare two flow patterns and return a similarity score.
    
    Args:
        flow1: First flow analysis result
        flow2: Second flow analysis result
        
    Returns:
        Similarity score between 0 and 1
    """
    # Compare key metrics
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
    
    # Normalize metrics
    metric1_norm = metric1 / np.linalg.norm(metric1)
    metric2_norm = metric2 / np.linalg.norm(metric2)
    
    # Compute cosine similarity
    similarity = np.dot(metric1_norm, metric2_norm)
    
    return float(similarity)

def extract_specular_flow_features(frames):
    feats = []
    for i in range(len(frames)-1):
        flow = cv2.calcOpticalFlowFarneback(frames[i], frames[i+1], None, 
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=True)
        feats.append(mag.mean())
        # feats.append(mag.std())  # Drop std for 9-dim
        hist, _ = np.histogram(ang, bins=7, range=(0,360), density=True)
        feats.extend(hist.tolist())
    arr = np.array(feats, dtype=np.float32).reshape(-1, 8)
    avg = arr.mean(axis=0)  # shape (8,)
    # Add mean magnitude as first dim, then 7 histogram bins
    return avg  # (8,)

# To get 9 dims, add mean of stds as last dim
# test_frames = [np.zeros((64,64), dtype=np.uint8)]*3
# assert extract_specular_flow_features(test_frames).shape in ((8,), (9,)) 