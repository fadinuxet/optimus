# Optical Auth Backend Server

This is the backend server for the Optical Auth app. Follow these steps to run it:

## Prerequisites

1. Install Python 3.8 or higher
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

1. Open a terminal/command prompt
2. Navigate to the backend folder:
   ```bash
   cd path/to/optical_auth_app/backend
   ```
3. Run the server:
   ```bash
   python start_server.py
   ```

The server will start and show you:
- The local URL (http://localhost:8000)
- The network URL (http://YOUR_IP:8000)

## Using with the App

1. Make sure your phone is on the same WiFi network as the computer running the server
2. In the app, update the API URL to the network URL shown by the server
3. The app should now be able to connect to the server

## Troubleshooting

1. If you see "Port 8000 is already in use":
   - Close any other applications that might be using port 8000
   - Or change the port in start_server.py

2. If the app can't connect:
   - Make sure your phone is on the same WiFi network
   - Check if your computer's firewall is blocking the connection
   - Try using the local IP address shown by the server

3. If you get Python errors:
   - Make sure you have all required packages installed
   - Try running: `pip install -r requirements.txt` again

# Optical Lens Authentication Backend

## Data Organization

### For RandomForest Classifier
```
data/train/lens/      # authentic lens images (for classifier)
data/train/not_lens/  # fakes, bottles, windows, screens, etc.
```

### For Deep Metric Model (Triplet)
```
data/enroll/LensA/    # multiple images of LensA
data/enroll/LensB/    # multiple images of LensB
# ...
```NFO:main:Image content type: application/octet-stream
ERROR:main:Input validation failed: Invalid image format. Accepted formats: image/jpeg, image/png, image/jpg
INFO:     192.168.5.61:50708 - "POST /api/fingerprint HTTP/1.1" 400 Bad Request

## Training the RandomForest Classifier

1. Place your labeled images in `data/train/lens` and `data/train/not_lens`.
2. Run:
   ```bash
   cd optical_auth_app/backend
   python train_classifier.py
   ```
3. This will save `lens_vs_not_lens.joblib` in the backend directory.

## Training the Deep Metric Model

1. Place authentic lens images in `data/enroll/<LensID>/` folders.
2. In Python, run:
   ```python
   from deep_metric import train_triplet
   # train_triplet("data/enroll", "model_triplet.pth", epochs=10)
   ```
3. This will save `model_triplet.pth` in the backend directory.

## Deploy
- Place both `lens_vs_not_lens.joblib` and `model_triplet.pth` in the backend directory.
- Restart your backend server.

## Test
- Register and authenticate real lenses (should succeed).
- Try fakes or non-lens glass (should be rejected). 

def is_lens_image(image, gabor_vec, flow_vec):
    # For now, always accept
    return True, {"note": "Classifier bypassed for now"} 

import os
import uuid
import cv2
from fastapi import UploadFile
from fastapi.responses import JSONResponse

def save_training_image(image_np, lens_id):
    out_dir = f'data/enroll/{lens_id}'
    os.makedirs(out_dir, exist_ok=True)
    filename = f'{uuid.uuid4().hex}.jpg'
    cv2.imwrite(os.path.join(out_dir, filename), image_np) 

def extract_frames(video_file: UploadFile, max_frames: int = 10):
    # ... video processing logic ...
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray) 

def check_capture_quality(image):
    # Implementation of check_capture_quality function
    # This is a placeholder and should be replaced with the actual implementation
    return True, "Image quality is acceptable"

async def authenticate_image(main_image: UploadFile):
    # ... existing code ...
    ok, reason = check_capture_quality(main_image)
    if not ok:
        return JSONResponse(status_code=400, content={"error": "Capture quality too low", "reason": reason})
    # ... rest of the function ... 

is_lens = (
    weighted_score > 0.6 and  # Overall score threshold
    ar_coating and  # Must have AR coating
    (edge_bevel or curvature) and  # Must have either edge bevel or curvature
    (reflection or has_texture)  # Must have either reflection patterns or texture
) 