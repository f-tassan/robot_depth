FROM python:3.11-slim

# Fix apt and install system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        git \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p weights && \
    curl -L -o weights/dpt_hybrid-midas-501f0c75.pt \
        https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt

COPY . .

CMD ["python3", "run_monodepth_webcam.py"]
