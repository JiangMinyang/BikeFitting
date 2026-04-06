# ─────────────────────────────────────────────────────────────────────────────
# Bike Fit Analyzer — Docker Image (Web Mode)
#
# Serves the web UI on port 8080. Upload side and/or front videos via the
# browser and get annotated videos + an HTML report.
#
# Build:  make build          (run `make download-model` first for RTMPose-l)
# Run:    make run
# ─────────────────────────────────────────────────────────────────────────────

# --- Stage 1: Base with system dependencies ---
FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive

# System packages for OpenCV + video processing (no GUI/X11 needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- Stage 2: Python dependencies ---
FROM base AS deps

WORKDIR /app

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# --- Stage 3: Application ---
FROM deps AS app

WORKDIR /app

# Copy application code
COPY core/       ./core/
COPY reports/    ./reports/
COPY web/        ./web/
COPY server.py   .
COPY main.py     .

# Copy pre-downloaded RTMPose-l ONNX model (if present locally).
# Run `make download-model` before `make build` to include it.
# If absent at build time, the backend auto-downloads on first inference.
COPY models/     ./models/

# Create output directory
RUN mkdir -p /app/output

# Expose the web server port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/status')" || exit 1

# Start in web mode
ENTRYPOINT ["python", "main.py", "--web", "--port", "8080"]
