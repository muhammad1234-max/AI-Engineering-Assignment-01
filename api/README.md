# Shakespearean LLM REST API & Inference Service

This directory documents the REST API service exposing Zaid's from-scratch trained Pre-LN causal Transformer (`zaid-lm`, 809k parameters) on Tiny Shakespeare.

---

## 1. Architecture & Endpoints

The API is built using **FastAPI** (`src/zaid_lm/api.py`) with asynchronous lifespan management, loading model weights into memory **once** at startup.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status, root welcome message, and endpoints directory |
| `GET` | `/health` | Health check verifying model residency in memory and device state |
| `GET` | `/model-info` | Metadata reporting parameter count (809,984), layers, heads, d_model, vocab size |
| `POST` | `/generate` | Autoregressive text generation conditioned on a text prompt |
| `GET` | `/docs` | Interactive Swagger OpenAPI documentation and testing UI |

---

## 2. Request & Response Specification

### `POST /generate`

**Request Body (`application/json`):**
```json
{
  "prompt": "ROMEO:\n",
  "max_new_tokens": 80,
  "temperature": 0.8,
  "top_k": 40,
  "top_p": 0.9,
  "repetition_penalty": 1.05,
  "seed": 42
}
```

**Response Body (`HTTP 200 OK`):**
```json
{
  "prompt": "ROMEO:\n",
  "completion": "As hor sow, lild sheng bat allee...",
  "full_text": "ROMEO:\nAs hor sow, lild sheng bat allee...",
  "tokens_generated": 80,
  "elapsed_seconds": 0.654,
  "tokens_per_second": 122.3,
  "model_step": 1500,
  "val_loss": 2.3384
}
```

---

## 3. Running Locally

### Direct Uvicorn Launch:
```bash
# Launch with 1 worker on port 8000
python -m uvicorn zaid_lm.api:app --host 0.0.0.0 --port 8000 --reload
```

### Testing with cURL:
```bash
# Health check
curl -X GET http://localhost:8000/health

# Model info
curl -X GET http://localhost:8000/model-info

# Text generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "HAMLET:\n", "max_new_tokens": 50, "temperature": 0.8}'
```

### Testing with PowerShell (Windows):
```powershell
# Text generation via Invoke-RestMethod
$Body = @{
    prompt = "KING CLAUDIUS:`n"
    max_new_tokens = 60
    temperature = 0.8
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/generate" -Method Post -Body $Body -ContentType "application/json"
```

---

## 4. Containerized & Cloud Deployment

### Local Docker Build & Run:
```bash
docker build -t zaid-lm-api .
docker run -p 8000:8000 zaid-lm-api
```

### Free-Tier Cloud Deployment (Render):
1. Connect repository `zaidbinnaveed/AI-Engineering-Assignment-01` to Render.
2. Select **Web Service** with Docker runtime (`render.yaml` blueprint included).
3. Checkpoint `checkpoints/best_model.pt` is loaded automatically.
4. Render exposes a public HTTPS endpoint (e.g. `https://zaid-lm-api.onrender.com`).
