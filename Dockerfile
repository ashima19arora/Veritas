# Pinned exact version — avoids the "latest tag moved under us" problem
FROM python:3.11.9-slim-bookworm

WORKDIR /app

# System deps needed transiently by some packages to build cleanly.
# Purged again at the end of this same layer so they don't bloat the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && pip install --no-cache-dir --break-system-packages --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/* /root/.cache

# App code, Streamlit config, and local embedding model weights — all baked into the image
COPY streamlit_app.py .
COPY .streamlit/ ./.streamlit/
COPY models/ ./models/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8501

# Ollama runs OUTSIDE this container, on the host machine — this app connects to it
# over the network. Do not try to run Ollama inside this same container.
ENV OLLAMA_HOST=http://host.docker.internal:11434

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]