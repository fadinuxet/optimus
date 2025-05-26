# Optical Authentication App

A mobile app for authenticating optical lenses using computer vision and machine learning.

## Features

- Lens detection and authentication
- Surface fingerprinting
- AR coating analysis
- Refractive index measurement
- Composition analysis

## Project Structure

- `lib/` - Flutter app code
- `backend/` - Python FastAPI server
- `android/` - Android app configuration
- `ios/` - iOS app configuration

## Setup

### Backend Server

1. Install Python 3.9 or higher
2. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   python start_server.py
   ```

### Mobile App

1. Install Flutter
2. Install dependencies:
   ```bash
   flutter pub get
   ```
3. Run the app:
   ```bash
   flutter run
   ```

## API Endpoints

- `POST /api/fingerprint` - Generate lens fingerprint
- `POST /api/authenticate` - Authenticate a lens
- `GET /api/lenses` - List registered lenses
- `GET /health` - Health check

## License

MIT License
