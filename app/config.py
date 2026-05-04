"""
Application configuration — override values via environment variables.
"""
import os

# Model paths
MODEL_PATH = os.getenv("MODEL_PATH", "models/incident_classifier.pth")
LABEL_MAP_PATH = os.getenv("LABEL_MAP_PATH", "models/label_map.json")

# Upload directory
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

# Device
DEVICE = os.getenv("DEVICE", "")  # leave empty for auto-detect (cuda/cpu)

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
