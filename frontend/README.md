# Shakespearean LLM — Web Studio & Playground

A modern, high-performance web interface for interacting with the from-scratch trained Shakespearean LLM (809k-parameter Pre-LN Causal Transformer).

---

## Features

- **Autoregressive Streaming Verse Generation**: Real-time token-by-token character reveal with stable OpenType tabular numerals (`tok/s`, elapsed time, latency HUD).
- **Theatrical First Folio Interface**: Dark velvet / imperial gold aesthetics with dialogue starter presets (Hamlet, Romeo & Juliet, King Claudius, Richard III).
- **Tuning Deck**: Full client-side control over sampling hyperparameters (temperature, top-k, top-p nucleus sampling, repetition penalty, RNG seeds, and backend API base URL).
- **Graceful Error Resilience**: Direct integration with FastAPI backend (`POST /generate` and `GET /health`). Unreachable backend states display clear connection error cards with instant retry actions.
- **Model Architecture Inspector**: Inspect model specifications (4 layers, 4 attention heads, 128 context length, 809,984 parameters).

---

## Getting Started

### Prerequisites

- Node.js 18+ and npm

### Local Development

1. Ensure the Python FastAPI backend is running on `http://localhost:8000`:
   ```bash
   python -m zaid_lm.api
   ```

2. Install dependencies and start the dev server:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. Open [http://localhost:8080](http://localhost:8080) in your browser.

### Production Build

```bash
npm run build
npm run preview
```
