# --- Stage 1: Frontend Builder ---
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

# Install frontend dependencies
COPY frontend/package.json ./
RUN npm install

# Copy source and build static bundle
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Production Server ---
FROM python:3.12-slim-bookworm
WORKDIR /app/backend

# Install minimal OS packages required by OpenCV & Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./

# Copy production static assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist/ ./static/
# Copy the built index.html from dist to backend directory so it serves at root /
COPY --from=frontend-builder /app/frontend/dist/index.html ./index.html

EXPOSE 8000

ENV PORT=8000
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
