# INCIDENT-AI-SERVICE/app/utils.py
"""
Shared utility helpers.
"""
import os
import uuid
from pathlib import Path
import shutil

def save_upload(file, upload_dir: str = "uploads") -> str:
    os.makedirs(upload_dir, exist_ok=True)

    ext = Path(file.filename).suffix
    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest = os.path.join(upload_dir, unique_name)

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return dest

def allowed_extension(filename: str, allowed: set = None) -> bool:
    if allowed is None:
        allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return Path(filename).suffix.lower() in allowed