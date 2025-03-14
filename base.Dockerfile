FROM nvidia/cuda:12.4.1-runtime-ubuntu20.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    iproute2 \
    procps \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install base requirements
RUN python3.9 -m pip install --upgrade pip
COPY requirements.txt .
RUN python3.9 -m pip install --no-cache-dir -r requirements.txt --timeout=120

# Default command (overridden by specific Dockerfiles)
CMD ["python3.9"]