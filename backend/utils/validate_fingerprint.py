import numpy as np
import cv2
import os
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two normalized vectors."""
    vec1_norm = normalize_vector(vec1)
    vec2_norm = normalize_vector(vec2)
    return float(np.dot(vec1_norm, vec2_norm))

def load_validation_data(data_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load validation data from directory structure:
    data_dir/
        lens1/
            capture1.jpg
            capture2.jpg
            ...
        lens2/
            capture1.jpg
            ...
    """
    vectors = []
    labels = []
    
    for lens_dir in Path(data_dir).iterdir():
        if not lens_dir.is_dir():
            continue
            
        lens_id = lens_dir.name
        for img_path in lens_dir.glob("*.jpg"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                # Generate fingerprint vector
                result = generate_surface_fingerprint(img)
                vectors.append(result["fingerprint_vector"])
                labels.append(lens_id)
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                continue
    
    return vectors, labels

def generate_roc_curve(vectors: List[np.ndarray], labels: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ROC curve data from validation vectors."""
    y_true = []
    y_score = []
    
    # Generate genuine and impostor pairs
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            similarity = compute_similarity(vectors[i], vectors[j])
            y_score.append(similarity)
            y_true.append(1 if labels[i] == labels[j] else 0)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, thresholds

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, output_dir: str):
    """Plot ROC curve and save to file."""
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc(fpr, tpr):.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Add threshold points
    for i in range(0, len(thresholds), len(thresholds)//10):
        plt.plot(fpr[i], tpr[i], 'o', label=f'Threshold = {thresholds[i]:.3f}')
    
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def find_optimal_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> float:
    """Find optimal threshold that maximizes TPR while keeping FPR low."""
    # Find threshold that gives TPR > 0.95 and minimizes FPR
    mask = tpr >= 0.95
    if not any(mask):
        mask = tpr >= 0.90  # Fallback to 90% TPR
    
    optimal_idx = np.argmin(fpr[mask])
    optimal_threshold = thresholds[mask][optimal_idx]
    
    return optimal_threshold

def main():
    # Configuration
    DATA_DIR = "validation_data"
    OUTPUT_DIR = "validation_results"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load validation data
    logger.info("Loading validation data...")
    vectors, labels = load_validation_data(DATA_DIR)
    
    if not vectors:
        logger.error("No validation data found!")
        return
    
    # Generate ROC curve
    logger.info("Generating ROC curve...")
    fpr, tpr, thresholds = generate_roc_curve(vectors, labels)
    
    # Plot ROC curve
    logger.info("Plotting ROC curve...")
    plot_roc_curve(fpr, tpr, thresholds, OUTPUT_DIR)
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(fpr, tpr, thresholds)
    logger.info(f"Optimal threshold: {optimal_threshold:.3f}")
    
    # Print performance metrics at optimal threshold
    idx = np.argmin(np.abs(thresholds - optimal_threshold))
    logger.info(f"At threshold {optimal_threshold:.3f}:")
    logger.info(f"True Positive Rate: {tpr[idx]:.3f}")
    logger.info(f"False Positive Rate: {fpr[idx]:.3f}")

if __name__ == "__main__":
    main() 