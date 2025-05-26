import uvicorn
import socket
import sys
import os
import logging
from fastapi.middleware.cors import CORSMiddleware
from main import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_public_ip():
    """Get the public IP address of the machine."""
    try:
        import requests
        response = requests.get('https://api.ipify.org?format=json')
        return response.json()['ip']
    except Exception as e:
        logger.error(f"Error getting public IP: {e}")
        return None

def check_port(port):
    """Check if the port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def main():
    port = 8000

    # Check if port is available
    if not check_port(port):
        logger.error(f"Error: Port {port} is already in use!")
        logger.error("Please close any other applications using this port.")
        sys.exit(1)

    # Get IP addresses
    local_ip = socket.gethostbyname(socket.gethostname())
    public_ip = get_public_ip()

    # Configure CORS for the app
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    print("\n=== Optical Auth Backend Server ===")
    print(f"Starting server at:")
    print(f"Local:   http://localhost:{port}")
    print(f"Network: http://{local_ip}:{port}")
    if public_ip:
        print(f"Public:  http://{public_ip}:{port}")
    print("\nTo use the app:")
    print("1. Make sure port 8000 is forwarded in your router")
    print("2. Update the API URL in the app to use the public IP")
    print("\nPress Ctrl+C to stop the server")
    print("================================\n")

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",  # Allow external connections
            port=port,
            reload=True,
            workers=1
        )
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 