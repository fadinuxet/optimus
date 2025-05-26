from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from analysis.refractive_index import analyze_refractive_index
from analysis.coating import analyze_coating
from analysis.composition import analyze_composition
from typing import Optional, Dict, List, Tuple
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from analysis.surface_fingerprint import generate_surface_fingerprint, check_capture_quality, compare_surface_fingerprints
import tempfile
import os
import io
import logging
from analysis.gabor_analysis import analyze_texture
from analysis.specular_flow import analyze_specular_flow
from lens_discriminators import compute_lens_discriminators, guess_glass_type
from lens_classifier import is_lens_image
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure data directories exist
os.makedirs('data/enroll', exist_ok=True)
os.makedirs('data/train/lens', exist_ok=True)
os.makedirs('data/train/not_lens', exist_ok=True)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory lens database
lens_database = {}

def read_image(file: UploadFile) -> np.ndarray:
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return img

@app.post("/analyze/refractive-index")
async def analyze_image(
    file: UploadFile = File(...),
    angle: float = 45.0
):
    # Read image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Analyze image
    results = analyze_refractive_index(img, angle)
    
    return results

@app.post("/analyze/coating")
async def analyze_coating_endpoint(
    file: UploadFile = File(...)
):
    # Read image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Analyze coating
    results = analyze_coating(img)
    
    return results

@app.post("/analyze/composition")
async def analyze_composition_endpoint(
    file: UploadFile = File(...)
):
    # Read image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Analyze composition
    results = analyze_composition(img)
    
    return results

def save_training_image(image_np, lens_id):
    """Save training image with proper error handling."""
    try:
        # Create base directory if it doesn't exist
        base_dir = 'data/enroll'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
            logger.info(f"Created base directory: {base_dir}")
        
        # Create lens-specific directory
        out_dir = os.path.join(base_dir, lens_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            logger.info(f"Created lens directory: {out_dir}")
        
        # Generate filename and save
        filename = f'{uuid.uuid4().hex}.jpg'
        filepath = os.path.join(out_dir, filename)
        
        # Save image
        success = cv2.imwrite(filepath, image_np)
        if not success:
            raise Exception("Failed to write image file")
            
        logger.info(f"Successfully saved training image to: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving training image: {str(e)}")
        # Don't raise the error, just log it and continue
        return False

def validate_input(macro_image: Optional[UploadFile], video: Optional[UploadFile]) -> Tuple[bool, str]:
    """Validate input files before processing."""
    try:
        if not macro_image and not video:
            return False, "Either image or video must be provided"
        
        if macro_image:
            # Check if file exists and has content
            if not macro_image.file:
                return False, "Image file is empty"
            
            # Log the content type for debugging
            logger.info(f"Image content type: {macro_image.content_type}")
            
            # Accept common image formats and octet-stream
            valid_types = ['image/jpeg', 'image/png', 'image/jpg', 'application/octet-stream']
            if macro_image.content_type.lower() not in valid_types:
                return False, f"Invalid image format. Accepted formats: {', '.join(valid_types)}"
            
            # Check file size
            if macro_image.size > 10 * 1024 * 1024:  # 10MB limit
                return False, "Image file too large (max 10MB)"
            
            # For octet-stream, try to validate the file content
            if macro_image.content_type == 'application/octet-stream':
                try:
                    # Read first few bytes to check file signature
                    header = macro_image.file.read(4)
                    macro_image.file.seek(0)  # Reset file pointer
                    
                    # Check for common image file signatures
                    if header.startswith(b'\xFF\xD8\xFF'):  # JPEG
                        logger.info("Detected JPEG format from file signature")
                    elif header.startswith(b'\x89PNG'):  # PNG
                        logger.info("Detected PNG format from file signature")
                    else:
                        logger.warning("Unknown file signature for octet-stream")
                except Exception as e:
                    logger.error(f"Error checking file signature: {str(e)}")
        
        if video:
            if not video.content_type.startswith('video/'):
                return False, "Invalid video format"
            if video.size > 50 * 1024 * 1024:  # 50MB limit
                return False, "Video file too large (max 50MB)"
        
        return True, ""
        
    except Exception as e:
        logger.error(f"Error in input validation: {str(e)}")
        return False, f"Error validating input: {str(e)}"

def extract_frames(video_file: UploadFile, max_frames: int = 10):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            video_bytes = video_file.file.read()
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise HTTPException(status_code=400, detail="Video file contains no frames")

        # Calculate frame interval to get evenly spaced frames
        if total_frames > max_frames:
            interval = total_frames // max_frames
        else:
            interval = 1

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only process frames at the calculated interval
            if frame_count % interval == 0:
                # Convert to grayscale and check quality
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ok, _ = check_capture_quality(frame)
                if ok:
                    frames.append(gray)
            
            frame_count += 1

        cap.release()
        os.unlink(temp_path)
        
        if not frames:
            raise HTTPException(status_code=400, detail="No valid frames found in video")
            
        return frames

    except Exception as e:
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/api/fingerprint")
async def fingerprint_lens(
    macro_image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    lens_id: Optional[str] = None
):
    try:
        # Validate inputs
        is_valid, error_msg = validate_input(macro_image, video)
        if not is_valid:
            return JSONResponse(status_code=400, content={"error": error_msg})
        
        main_image = None
        if macro_image:
            main_image = read_image(macro_image)
        
        frames = None
        if video:
            frames = extract_frames(video)
            if not main_image and frames:
                main_image = cv2.cvtColor(frames[0], cv2.COLOR_GRAY2BGR)
        
        if main_image is None:
            return JSONResponse(status_code=400, content={"error": "No valid image data available"})
        
        # Check image quality
        ok, reason = check_capture_quality(main_image)
        if not ok:
            return JSONResponse(status_code=400, content={"error": "Capture quality too low", "reason": reason})
        
        try:
            # First, check if it's actually a lens
            from analysis.gabor_analysis import extract_gabor_features
            from analysis.specular_flow import extract_specular_flow_features
            gabor_vec = extract_gabor_features(main_image)
            flow_vec = extract_specular_flow_features(frames) if frames is not None else np.zeros(8, dtype=np.float32)
            
            # Convert numpy arrays to lists for comparison
            gabor_list = gabor_vec.tolist() if isinstance(gabor_vec, np.ndarray) else gabor_vec
            flow_list = flow_vec.tolist() if isinstance(flow_vec, np.ndarray) else flow_vec
            
            ok, details = is_lens_image(main_image, gabor_list, flow_list)
            if not ok:
                return JSONResponse(status_code=400, content={
                    "error": "Not a lens",
                    "details": details,
                    "description": "This object does not appear to be a lens. Please capture a clear image of a lens."
                })
            
            # Only proceed with fingerprinting if it's confirmed to be a lens
            result = generate_surface_fingerprint(main_image, frames)
            if not isinstance(result, dict):
                raise ValueError("Fingerprint generation did not return a dictionary")
            
            # Additional validation of the fingerprint
            if result.get("quality", "Low") == "Low":
                return JSONResponse(status_code=400, content={
                    "error": "Low quality fingerprint",
                    "description": "The lens image quality is too low for reliable fingerprinting."
                })
            
            if result.get("score", 0.0) < 0.7:
                return JSONResponse(status_code=400, content={
                    "error": "Low confidence fingerprint",
                    "description": "The lens image does not provide enough confidence for fingerprinting."
                })
            
            # If we have a lens_id, save the fingerprint
            if lens_id:
                # Check if this lens_id already exists
                if lens_id in lens_database:
                    return JSONResponse(status_code=400, content={
                        "error": "Lens ID already exists",
                        "description": f"Lens ID {lens_id} is already registered. Please use a different ID."
                    })
                
                # Save the fingerprint
                lens_database[lens_id] = {
                    "quality": result.get("quality", "N/A"),
                    "score": result.get("score", 0.0),
                    "fingerprint_vector": result.get("fingerprint_vector", []),
                    "fingerprint_hash": result.get("fingerprint_hash", ""),
                }
                
                return {
                    "message": "Lens registered successfully",
                    "lens_id": lens_id,
                    "quality": result.get("quality", "N/A"),
                    "score": result.get("score", 0.0),
                    "fingerprint_hash": result.get("fingerprint_hash", ""),
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in fingerprint processing: {str(e)}")
            return JSONResponse(status_code=500, content={"error": f"Error processing fingerprint: {str(e)}"})
            
    except Exception as e:
        logger.error(f"Fingerprint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

from analysis.surface_fingerprint import compare_surface_fingerprints

@app.post("/api/authenticate")
async def authenticate_lens(
    lens_id: str = Form(...),
    macro_image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None)
):
    try:
        # Validate inputs
        is_valid, error_msg = validate_input(macro_image, video)
        if not is_valid:
            return JSONResponse(status_code=400, content={"error": error_msg})
        
        if lens_id not in lens_database:
            return JSONResponse(status_code=404, content={"error": "Lens ID not found"})
        
        main_image = None
        if macro_image:
            main_image = read_image(macro_image)
        
        frames = None
        if video:
            frames = extract_frames(video)
            if not main_image and frames:
                main_image = cv2.cvtColor(frames[0], cv2.COLOR_GRAY2BGR)
        
        if not main_image:
            return JSONResponse(status_code=400, content={"error": "No valid image data available"})
        
        # Check image quality
        ok, reason = check_capture_quality(main_image)
        if not ok:
            return JSONResponse(status_code=400, content={"error": "Capture quality too low", "reason": reason})
        
        try:
            result = generate_surface_fingerprint(main_image, frames)
            new_fingerprint = result
            stored_fingerprint = lens_database[lens_id]
            
            # Get fingerprint vectors
            v1 = np.array(new_fingerprint["fingerprint_vector"])
            v2 = np.array(stored_fingerprint["fingerprint_vector"])
            
            # Check if vectors are valid
            if v1.size == 0 or v2.size == 0:
                return JSONResponse(status_code=400, content={"error": "Invalid fingerprint vectors"})
            
            # Normalize vectors
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            
            # Calculate cosine similarity
            similarity = float(np.dot(v1_norm, v2_norm))
            
            # Use a stricter threshold
            threshold = 0.95
            authenticated = similarity >= threshold
            
            return {
                "authenticated": authenticated,
                "similarity_score": similarity,
                "threshold": threshold,
                "quality": new_fingerprint.get("quality", "N/A"),
                "score": new_fingerprint.get("score", 0.0),
                "description": new_fingerprint.get("description", "")
            }
        except Exception as e:
            logger.error(f"Error in authentication processing: {str(e)}")
            return JSONResponse(status_code=500, content={"error": "Error processing authentication"})
            
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/authenticate_any")
async def authenticate_any_lens(
    macro_image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None)
):
    try:
        # Validate inputs
        is_valid, error_msg = validate_input(macro_image, video)
        if not is_valid:
            return JSONResponse(status_code=400, content={"error": error_msg})
        
        main_image = None
        if macro_image:
            main_image = read_image(macro_image)
        
        frames = None
        if video:
            frames = extract_frames(video)
            if not main_image and frames:
                main_image = cv2.cvtColor(frames[0], cv2.COLOR_GRAY2BGR)
        
        if main_image is None:
            return JSONResponse(status_code=400, content={"error": "No valid image data available"})
        
        # Check image quality
        ok, reason = check_capture_quality(main_image)
        if not ok:
            return JSONResponse(status_code=400, content={"error": "Capture quality too low", "reason": reason})
        
        try:
            # First, check if it's actually a lens
            from analysis.gabor_analysis import extract_gabor_features
            from analysis.specular_flow import extract_specular_flow_features
            gabor_vec = extract_gabor_features(main_image)
            flow_vec = extract_specular_flow_features(frames) if frames is not None else np.zeros(8, dtype=np.float32)
            
            # Convert numpy arrays to lists for comparison
            gabor_list = gabor_vec.tolist() if isinstance(gabor_vec, np.ndarray) else gabor_vec
            flow_list = flow_vec.tolist() if isinstance(flow_vec, np.ndarray) else flow_vec
            
            ok, details = is_lens_image(main_image, gabor_list, flow_list)
            if not ok:
                return JSONResponse(status_code=400, content={
                    "authenticated": False,
                    "error": "Not a lens",
                    "details": details,
                    "description": "This object does not appear to be a lens. Please capture a clear image of a lens."
                })
            
            # Generate fingerprint for the new image
            result = generate_surface_fingerprint(main_image, frames)
            if not isinstance(result, dict):
                raise ValueError("Fingerprint generation did not return a dictionary")
                
            new_fingerprint = result
            v1 = np.array(new_fingerprint["fingerprint_vector"])
            
            if v1.size == 0:
                return JSONResponse(status_code=400, content={"error": "Invalid fingerprint vector"})
            
            best_similarity = 0
            best_id = None
            
            # Compare with all stored fingerprints
            for lens_id, stored_fingerprint in lens_database.items():
                try:
                    # Get stored vector
                    v2 = np.array(stored_fingerprint["fingerprint_vector"])
                    
                    if v2.size == 0:
                        logger.warning(f"Empty stored vector for lens {lens_id}")
                        continue
                    
                    # Ensure vectors are the same length
                    if v1.shape != v2.shape:
                        logger.warning(f"Vector shape mismatch for lens {lens_id}: {v1.shape} vs {v2.shape}")
                        continue
                    
                    # Normalize vectors
                    v1_norm = v1 / np.linalg.norm(v1)
                    v2_norm = v2 / np.linalg.norm(v2)
                    
                    # Calculate cosine similarity
                    similarity = float(np.dot(v1_norm, v2_norm))
                    
                    # Additional quality checks
                    quality_score = new_fingerprint.get("score", 0.0)
                    if quality_score < 0.7:  # Require high quality
                        logger.warning(f"Low quality score: {quality_score}")
                        continue
                        
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_id = lens_id
                        
                except Exception as e:
                    logger.error(f"Error comparing with lens {lens_id}: {str(e)}")
                    continue
            
            # Use a much stricter threshold
            threshold = 0.98  # Increased from 0.95
            authenticated = best_similarity >= threshold
            
            # Additional validation
            if authenticated:
                # Check if the quality is high enough
                if new_fingerprint.get("quality", "Low") == "Low":
                    authenticated = False
                    logger.warning("Authentication rejected due to low quality")
                
                # Check if the score is high enough
                if new_fingerprint.get("score", 0.0) < 0.8:
                    authenticated = False
                    logger.warning("Authentication rejected due to low score")
            
            return {
                "authenticated": authenticated,
                "similarity_score": best_similarity,
                "matched_lens_id": best_id if authenticated else None,
                "threshold": threshold,
                "quality": new_fingerprint.get("quality", "N/A"),
                "score": new_fingerprint.get("score", 0.0),
                "description": new_fingerprint.get("description", "")
            }
            
        except Exception as e:
            logger.error(f"Error in authentication processing: {str(e)}")
            return JSONResponse(status_code=500, content={"error": f"Error processing authentication: {str(e)}"})
            
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/lenses")
def list_lenses():
    # Return a list of registered lens IDs and their hashes
    return {"lenses": [{"lens_id": lid, "fingerprint_hash": lens_database[lid].get("fingerprint_hash", "")} for lid in lens_database]}

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two normalized vectors."""
    vec1_norm = normalize_vector(vec1)
    vec2_norm = normalize_vector(vec2)
    return float(np.dot(vec1_norm, vec2_norm))

def check_capture_quality(image: np.ndarray):
    """
    Checks if the image has acceptable brightness and sharpness.
    - Brightness: mean pixel value in [50, 250] (more lenient range)
    - Sharpness: Laplacian variance > 20 (lower threshold)
    Returns (True, None) if pass, (False, reason) otherwise.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if not (50 <= brightness <= 250):
            return False, f"Image brightness ({brightness:.1f}) is out of range [50, 250]. Please adjust lighting."
        if sharpness < 20:
            return False, f"Image is not sharp enough (score: {sharpness:.1f}). Please hold the camera steady and ensure focus."
        return True, None
    except Exception as e:
        logger.error(f"Error in quality check: {str(e)}")
        return False, "Error checking image quality. Please try again."

@app.post("/api/register")
async def register_lens(
    image: UploadFile = File(...),
    lens_id: str = Form(None)
) -> Dict:
    """Register a new lens with multiple fingerprint captures."""
    try:
        # Read and decode image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Check image quality
        ok, reason = check_capture_quality(img)
        if not ok:
            raise HTTPException(status_code=400, detail=reason)
        
        # Generate fingerprint
        try:
            result = generate_surface_fingerprint(img)
        except Exception as e:
            logger.error(f"Error generating fingerprint: {str(e)}")
            raise HTTPException(status_code=400, detail="Failed to generate fingerprint. Please try again with a clearer image.")
        
        # Store the full feature vector and hash
        if lens_id not in lens_database:
            lens_database[lens_id] = []
        
        # Store up to 5 fingerprints per lens
        if len(lens_database[lens_id]) >= 5:
            lens_database[lens_id].pop(0)  # Remove oldest fingerprint
        
        lens_database[lens_id].append(result)
        
        # Save the image for training
        try:
            save_training_image(img, lens_id)
        except Exception as e:
            logger.error(f"Error saving training image: {str(e)}")
            # Continue even if saving fails
        
        logger.info(f"Registered lens {lens_id} with {len(lens_database[lens_id])} fingerprints")
        
        return {
            "status": "success",
            "message": f"Lens {lens_id} registered successfully",
            "fingerprint_hash": result["fingerprint_hash"],
            "quality": result["quality"],
            "score": result["score"]
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in registration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/lenses")
async def list_lenses() -> Dict:
    """List all registered lenses."""
    return {
        "lenses": [
            {
                "lens_id": lens_id,
                "num_fingerprints": len(fingerprints)
            }
            for lens_id, fingerprints in lens_database.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 