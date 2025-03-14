FROM nvidia/cuda:12.4.1-runtime-ubuntu20.04
WORKDIR /app
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y pytPhon3.9 python3.9-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN python3.9 -m pip install --upgrade pip
COPY requirements.txt .
RUN python3.9 -m pip install --no-cache-dir -r requirements.txt --timeout=120