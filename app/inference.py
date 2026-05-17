# INCIDENT-AI-SERVICE/app/inference.py

import os
import json
import time
import urllib.request
import torch
import timm
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from app.config import MODEL_PATH, LABEL_MAP_PATH, DEVICE as _CFG_DEVICE

# -------------------------
# DEVICE
# -------------------------
DEVICE = _CFG_DEVICE if _CFG_DEVICE else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -------------------------
# MODEL DOWNLOAD URL
# -------------------------
MODEL_URL = (
    "https://huggingface.co/ABENEZERSELESHI/incident_report_model_2.0/resolve/main/incident_classifier%20(1).pth"
)

# -------------------------
# DOWNLOAD HELPER
# -------------------------
def download_with_retry(url, path, retries=3):

    for i in range(retries):

        try:
            print(f"⬇️ Attempt {i+1} downloading model...")

            urllib.request.urlretrieve(url, path)

            # Validate file is not HTML
            with open(path, "rb") as f:
                first_bytes = f.read(20)

            if first_bytes.startswith(b"<"):
                raise RuntimeError(
                    "Downloaded file is HTML, not model binary."
                )

            print("✅ Model downloaded successfully")
            return

        except Exception as e:

            print(f"❌ Retry {i+1} failed:", e)

            # Remove corrupted file
            if os.path.exists(path):
                os.remove(path)

            time.sleep(2)

    raise RuntimeError("Download failed after retries")

# -------------------------
# ENSURE MODEL DIRECTORY
# -------------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# -------------------------
# FORCE CLEAN OLD MODEL
# -------------------------
if os.path.exists(MODEL_PATH):
    print("🗑 Removing old cached model...")
    os.remove(MODEL_PATH)

# -------------------------
# DOWNLOAD MODEL
# -------------------------
print("⬇️ Downloading fresh model...")
download_with_retry(MODEL_URL, MODEL_PATH)

# -------------------------
# VALIDATE LABEL MAP
# -------------------------
if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError(
        f"Label map not found at: {LABEL_MAP_PATH}"
    )

print(f"✅ Model path: {MODEL_PATH}")
print(f"✅ Label map path: {LABEL_MAP_PATH}")

# -------------------------
# LOAD LABEL MAP
# -------------------------
with open(LABEL_MAP_PATH, "r") as f:
    ID2LABEL = json.load(f)

# -------------------------
# CREATE MODEL
# -------------------------
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=4
)

# -------------------------
# LOAD CHECKPOINT
# -------------------------
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

# -------------------------
# HANDLE DIFFERENT SAVE FORMATS
# -------------------------
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

# -------------------------
# FINALIZE MODEL
# -------------------------
model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully")

# -------------------------
# IMAGE TRANSFORMS
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# PREDICT
# -------------------------
def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probs = F.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs, dim=1)

    return {
        "priority": ID2LABEL[str(pred.item())],
        "confidence": round(confidence.item(), 4)
    }