# docker build -t speedrun_plm .
# docker run --gpus all -v "${PWD}:/workspace" speedrun_plm python train.py --config experiment.json
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.12.7 \
    PATH=/usr/local/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates ninja-build \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libncursesw5-dev xz-utils tk-dev \
        libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLO https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd .. && rm -rf Python-${PYTHON_VERSION}* && \
    ln -s /usr/local/bin/python3.12 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.12    /usr/local/bin/pip

WORKDIR /app

# Cache dependency installation independently of source changes.
COPY requirements.txt .

RUN pip install --upgrade pip setuptools && \
    pip install torch --index-url https://download.pytorch.org/whl/cu128 -U && \
    pip install -r requirements.txt

COPY . .

RUN pip install -e ".[test,evaluation]"

WORKDIR /workspace

# Prefer the bind-mounted candidate over the image's installed source.
ENV PROJECT_ROOT=/workspace \
    PYTHONPATH=/workspace/src \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    XDG_CACHE_HOME=/workspace/.cache \
    TQDM_CACHE=/workspace/.cache/tqdm

RUN mkdir -p \
      /workspace/.cache/huggingface \
      /workspace/.cache/torch \
      /workspace/.cache/tqdm \
      /workspace/logs \
      /workspace/data \
      /workspace/results

VOLUME ["/workspace"]

CMD ["bash"]
