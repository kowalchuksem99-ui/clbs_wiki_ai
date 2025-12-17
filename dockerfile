FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    CHROMA_DB_PATH=/data/chroma_db

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl ca-certificates \
    build-essential \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- зависимости Python ---
COPY requirements.txt /app/requirements.txt

# CPU - PyTorch
#RUN pip install --no-cache-dir torch --index-url https://pypi.org/simple/
#RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# reqs
RUN pip install --no-cache-dir -r /app/requirements.txt

# catalogues
RUN mkdir -p /data/chroma_db /models

# code
COPY . /app

EXPOSE 5010

# /healthz
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:5000/dev/healthz || exit 1

# uvicorn
CMD ["uvicorn", "main_fast_api:app", "--host", "0.0.0.0", "--port", "5010"]
