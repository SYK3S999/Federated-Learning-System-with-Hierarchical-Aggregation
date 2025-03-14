FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    iproute2 \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install base requirements
RUN python3 -m pip install --upgrade pip
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt --timeout=120

# Default command (overridden by specific Dockerfiles)
CMD ["python3"]