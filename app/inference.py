import json
import torch
import timm
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Resolve paths relative to project root (one level up from app/)
BASE_DIR = Path(__file__).resolve().parent.parent

# -------------------------
# LOAD LABEL MAP
# -------------------------

with open(BASE_DIR / "models" / "label_map.json") as f:
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
        BASE_DIR / "models" / "incident_classifier.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

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
    image = transform(image)
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    return {
        "priority": ID2LABEL[str(pred.item())],
        "confidence": round(confidence.item(), 4)
    }
