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
DEVICE = _CFG_DEVICE if _CFG_DEVICE else ("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# DOWNLOAD CONFIG
# -------------------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1CgQQmWDp3uYxrlE8AfnuYIqCQSbQdLsm"

# -------------------------
# HELPERS
# -------------------------
def download_with_retry(url, path, retries=3):
    for i in range(retries):
        try:
            print(f"⬇️ Attempt {i+1} downloading model...")
            urllib.request.urlretrieve(url, path)
            print("✅ Model downloaded successfully")
            return
        except Exception as e:
            print(f"Retry {i+1} failed:", e)
            time.sleep(2)
    raise RuntimeError("Download failed after retries")

# -------------------------
# ENSURE MODEL EXISTS
# -------------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("⬇️ Model not found. Downloading...")
    download_with_retry(MODEL_URL, MODEL_PATH)

# -------------------------
# VALIDATE LABEL MAP
# -------------------------
if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError(f"Label map not found at: {LABEL_MAP_PATH}")

print(f"✅ Model path: {MODEL_PATH}")
print(f"✅ Label map path: {LABEL_MAP_PATH}")

# -------------------------
# LOAD LABEL MAP
# -------------------------
with open(LABEL_MAP_PATH) as f:
    ID2LABEL = json.load(f)

# -------------------------
# LOAD MODEL
# -------------------------
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=4
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False   # 🔥 CRITICAL FIX
    )
)

model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully")

# -------------------------
# TRANSFORMS
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