# incident-ai-service

FastAPI service that classifies incident images into priority levels using EfficientNet-B0.

## Project Structure

```
incident-ai-service/
├── app/
│   ├── main.py        # FastAPI app & routes
│   ├── inference.py   # Model loading & prediction
│   ├── explain.py     # Grad-CAM explanations
│   ├── utils.py       # Shared helpers
│   └── config.py      # Configuration / env vars
├── models/
│   ├── incident_classifier.pth   # Trained weights (add manually)
│   └── label_map.json            # {id: label} mapping
├── uploads/           # Uploaded images (auto-created)
├── requirements.txt
└── README.md
```

## Setup
https://github.com/ABENEZER-SELESHI/incident-ai-service-model.git
```bash
# 1. Create & activate virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Add Your Model

Place your trained weights at `models/incident_classifier.pth`.  
The default `label_map.json` maps 0→low, 1→medium, 2→high, 3→critical.

## Run

```bash
cd incident-ai-service
uvicorn app.main:app --reload
```

## API

| Method | Path       | Description              |
|--------|------------|--------------------------|
| GET    | `/`        | Health check             |
| POST   | `/predict` | Upload image, get priority |

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/image.jpg"
```

Response:
```json
{"priority": "high", "confidence": 0.9231}
```
