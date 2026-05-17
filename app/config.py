# INCIDENT-AI-SERVICE/app/config.py
"""
Application configuration — override values via environment variables.
"""
import os
from pathlib import Path

# Base directory (PROJECT ROOT)
BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths (relative → absolute)
MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "models/incident_classifier (1).pth")
LABEL_MAP_PATH = BASE_DIR / os.getenv("LABEL_MAP_PATH", "models/label_map (1).json")

# Upload directory
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")

# Device
DEVICE = os.getenv("DEVICE", "")  # auto-detect if empty

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))