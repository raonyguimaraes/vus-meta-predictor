# VUS Meta-Predictor
# Multi-stage build: download data + build app
FROM python:3.12-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps (pinned for reproducibility)
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    numpy pandas scikit-learn xgboost shap loguru tqdm \
    torch --index-url https://download.pytorch.org/whl/cpu \
    fair-esm biopython requests click pydantic \
    fastapi uvicorn jinja2 pyBigWig matplotlib seaborn && \
    pip install --no-cache-dir -e .

# Copy source
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY tests/ tests/

# Download data at build time (optional — can also mount)
RUN mkdir -p data/raw data/processed data/models

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default: run web server
CMD ["python", "-m", "uvicorn", "vus_meta.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
