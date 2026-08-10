# EMAFRS / AMIF backend image -- FastAPI + the full research pipeline (CPU-only).
FROM python:3.12-slim AS backend

WORKDIR /app

# System deps for scientific Python wheels that occasionally need build tools on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
# CPU-only torch wheel first (avoids the much larger default CUDA build).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Non-root user for the running container.
RUN useradd --create-home --shell /bin/bash emafrs \
    && mkdir -p /app/data /app/models/checkpoints \
    && chown -R emafrs:emafrs /app
USER emafrs

ENV DATABASE_URL=sqlite:////app/data/emafrs.db
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "emafrs.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
