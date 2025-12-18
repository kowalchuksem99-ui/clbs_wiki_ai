FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Единый кэш для HF/transformers/sentence-transformers
    HF_HOME=/models \
    SENTENCE_TRANSFORMERS_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    HF_HUB_ENABLE_TELEMETRY=0 \
    CHROMA_DB_PATH=/data/chroma_db

# EMBED_MODEL можно переопределить при сборке --build-arg EMBED_MODEL=...
ARG EMBED_MODEL=paraphrase-multilingual-mpnet-base-v2
ENV EMBED_MODEL=${EMBED_MODEL}

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl ca-certificates \
    build-essential gcc g++ cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- зависимости Python ---
COPY requirements.txt /app/requirements.txt

# PyTorch CPU
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Остальные зависимости
RUN pip install --no-cache-dir -r /app/requirements.txt

# Каталоги данных/кэша
RUN mkdir -p /data/chroma_db /models

# 🔥 Прогреваем эмбеддер на этапе сборки (кладём модель в /models)
RUN python - <<'PY'
import os
from sentence_transformers import SentenceTransformer as S

model = os.getenv("EMBED_MODEL", "paraphrase-multilingual-mpnet-base-v2")
cache = os.getenv("SENTENCE_TRANSFORMERS_HOME", "/models")
print(f"Downloading '{model}' to cache: {cache}")
S(model, cache_folder=cache)
print("Model cached.")
PY

# Код приложения
COPY . /app

EXPOSE 5010

# Healthcheck: порт должен совпадать с uvicorn (5010), путь — ваш эндпоинт
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:5010/healthz || exit 1

# Uvicorn
CMD ["uvicorn", "main_fast_api:app", "--host", "0.0.0.0", "--port", "5010"]
