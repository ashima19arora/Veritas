#!/bin/bash
# entrypoint.sh — runs every time the container starts, AFTER volumes are mounted.
# Ensures vectorstore_index/ and docs/ are writable regardless of how Docker
# created them on the host (Windows volume mounts often create read-only folders).
chmod -R 777 /app/vectorstore_index 2>/dev/null || true
chmod -R 777 /app/docs 2>/dev/null || true
exec python -m streamlit run streamlit_app.py --server.address=0.0.0.0
