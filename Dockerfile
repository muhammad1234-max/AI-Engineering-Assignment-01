# Shakespearean LLM — Educational GPT Model Deployment Container
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# Install core API dependencies from standard PyPI
RUN pip install --no-cache-dir \
    numpy==2.4.6 \
    fastapi==0.138.0 \
    uvicorn==0.49.0 \
    pydantic==2.13.4

# Copy repository package and metadata
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e . --no-deps

# Copy checkpoints directory
COPY checkpoints/ ./checkpoints/

EXPOSE 8000

CMD ["sh", "-c", "if [ ! -f checkpoints/best_model.pt ] && [ -n \"$ZAID_LM_CHECKPOINT_URL\" ]; then echo 'Fetching model checkpoint...' && curl -fsSL \"$ZAID_LM_CHECKPOINT_URL\" -o checkpoints/best_model.pt; fi && uvicorn zaid_lm.api:app --host 0.0.0.0 --port ${PORT}"]
