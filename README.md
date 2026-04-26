# Intel Image Classification Backend

FastAPI backend for classifying images into 6 classes (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`) using a PyTorch model (`ImprovedIntelModel`). Supports feature extraction and dropout transformations.

## Installation

```bash
git clone (https://github.com/Kratos-024/CNN-Explainer-backend.git)
cd CNN-Explainer-backend
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Run Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

* `GET /get` → Health check
* `POST /classify` → Classify image (file form data: `Img`)
* `POST /getImageData` → Extract features (optional `layers` parameter)
* `POST /applyDropout` → Apply dropout to image (JSON: `Img` as base64)

## Notes

* Model file: `intel_complete_model.pt`
* Uses GPU if available
* API is CORS-enabled for frontend integration

