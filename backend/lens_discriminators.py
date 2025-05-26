import cv2
import numpy as np

def detect_ar_coating_interference(image: np.ndarray, roi_size=100) -> bool:
    """
    Sample a central ROI and analyze for thin-film interference colors.
    Returns True if hue variance exceeds threshold.
    """
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    # Ensure ROI is within bounds
    if h < roi_size or w < roi_size:
        print("[AR] ROI too small for analysis.")
        return False
    roi = image[cy-roi_size//2:cy+roi_size//2, cx-roi_size//2:cx+roi_size//2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue_std = float(np.std(hsv[:,:,0]) / 180.0)
    print(f"[AR] hue_std: {hue_std:.4f}")
    # Raise threshold for stricter check
    return bool(hue_std > 0.05)  # tune this based on real lens AR patterns

def analyze_polarization_sequence(frames: list[np.ndarray]) -> bool:
    """
    Given 4 frames through a rotating polarizer at 0°,45°,90°,135°,
    compute mean intensities and check for extinction dip.
    Returns True if I_min/I_max < 0.3 (strong polarization).
    """
    ints = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        ints.append(float(np.mean(gray)))
    Imax, Imin = max(ints), min(ints)
    ratio = (Imin / Imax) if Imax > 0 else 1.0
    print(f"[POL] Imax: {Imax:.2f}, Imin: {Imin:.2f}, ratio: {ratio:.3f}")
    return bool(ratio < 0.25)  # stricter threshold

def detect_edge_bevel_signature(image: np.ndarray) -> bool:
    """
    Detect a clear bevel or polished edge around the lens.
    Returns True if a consistent bright ring of width 5–15 pixels is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Dilate to connect edge segments
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    ring = cv2.dilate(edges, kernel) - edges
    h, w = gray.shape
    if h < 20 or w < 20:
        print("[EDGE] Image too small for edge analysis.")
        return False
    border = ring[5:-5, 5:-5]  # avoid corners
    ring_density = float(np.mean(border) / 255.0)
    print(f"[EDGE] ring_density: {ring_density:.4f}")
    # Raise threshold for stricter check
    return bool(ring_density > 0.05)  # tune per your lens edge profile

def compute_lens_discriminators(
    macro_image: np.ndarray,
    pol_frames: list[np.ndarray] = None
) -> (bool, dict):
    """
    Run all discriminators and return (passed: bool, details: dict).
    """
    details = {}
    details['ar_coating'] = bool(detect_ar_coating_interference(macro_image))
    details['edge_bevel'] = bool(detect_edge_bevel_signature(macro_image))
    if pol_frames:
        details['polarization'] = bool(analyze_polarization_sequence(pol_frames))
    else:
        details['polarization'] = False
    print(f"[DISCRIMINATORS] Results: {details}")
    passed = sum(details.values()) >= 2
    print(f"[DISCRIMINATORS] Passed: {passed}")
    return passed, details

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