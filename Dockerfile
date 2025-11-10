FROM python:3.13-slim

# Install uv (fast dependency manager)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Install system dependencies (fitz, pypdf etc. may need these)
RUN apt-get update && apt-get install -y build-essential libgl1-mesa-dev && rm -rf /var/lib/apt/lists/*

# Copy project metadata and code
COPY pyproject.toml uv.lock main.py ./

# Install dependencies from lockfile
RUN uv sync --frozen --no-dev

# Set runtime port (you can use 8080 for local and Code Engine)
ENV PORT=8080
EXPOSE $PORT

# Start FastAPI app through uv
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
