# The speedrun image: CUDA 12.8 (the box's driver is 570), Python 3.12 in a venv, torch 2.11 cu128 (the newest
# aarch64 cu128 wheel), this project installed editable. Built and run through run_gh200.sh, which bind-mounts the shipped tree and the foundry import root at
# their host paths.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PATH=/opt/venv/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.12 python3.12-dev python3.12-venv git ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    python3.12 -m venv /opt/venv

WORKDIR /opt/speedrun

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128 && \
    pip install -r requirements.txt

COPY . .
RUN pip install -e ".[test]"

ENV HF_HUB_ENABLE_HF_TRANSFER=1
CMD ["bash"]
