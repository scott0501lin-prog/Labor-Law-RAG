# Hugging Face Spaces (Docker SDK) image for the Streamlit app.
# Deploy with scripts/deploy_hf.sh — it pushes only the runtime files.
FROM python:3.12-slim

# HF runs the container as uid 1000; give that user a writable home + app dir
# (Chroma writes backend/chroma_db, the app writes rag_timing.log).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only torch first, so sentence-transformers doesn't pull the CUDA build
# (several GB larger; the Space has no GPU).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake bge-m3 into the image so a restart doesn't re-download ~2 GB.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY --chown=user . .

EXPOSE 8501
CMD ["streamlit", "run", "Fronted/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
