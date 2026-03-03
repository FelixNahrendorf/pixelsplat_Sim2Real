# PixelSplat Container Setup Guide

This guide walks through building and configuring a Podman container for running PixelSplat (Sim2Real) with GPU support, proxy access, and all required dependencies.

---

## Table of Contents

1. [Pull and Tag Base Image](#1-pull-and-tag-base-image)
2. [Start the Container](#2-start-the-container)
3. [Configure Proxy and SSL](#3-configure-proxy-and-ssl)
4. [Enter the Container](#4-enter-the-container)
5. [Fix APT Sources and Update Certificates](#5-fix-apt-sources-and-update-certificates)
6. [Install Dependencies](#6-install-dependencies)
7. [Download Pretrained Weights](#7-download-pretrained-weights)
8. [Alternative: Build from Dockerfile](#8-alternative-build-from-dockerfile)
9. [Run Training](#9-run-training)

---

## 1. Pull and Tag Base Image

```bash
podman pull pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
podman tag pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel pixelsplat-base:latest
```

---

## 2. Start the Container

```bash
podman run --name pixelsplat \
  --userns=keep-id \
  --pids-limit=8192 \
  --security-opt label=disable \
  --device nvidia.com/gpu=all \
  --ipc=host \
  --network=host \
  --init \
  -d \
  -v /rbcn002x/share/<username>:/app/workspace \
  -v /rbcn002x/share/datasets:/app/datasets \
  pixelsplat-base:latest \
  sleep infinity
```

> **Note:** Replace `<username>` and the volume paths with your actual user/data directories.

---

## 3. Configure Proxy and SSL

### Set APT Proxy

```bash
podman exec -it --user=0:0 pixelsplat bash -c 'cat > /etc/apt/apt.conf.d/proxy.conf <<EOF
Acquire::http::Proxy "http://127.0.0.1:3128";
Acquire::https::Proxy "http://127.0.0.1:3128";
EOF'
```

### Set SSL Certificate Environment Variables

```bash
podman exec -it --user=0:0 pixelsplat sh -c '
  export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
  export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
'
```

---

## 4. Enter the Container

```bash
podman exec -it --user=0:0 pixelsplat bash
```

All commands below are run **inside** the container as root.

---

## 5. Fix APT Sources and Update Certificates

Some environments require HTTPS for APT sources. Use **one** of the following:

```bash
# Option A (Ubuntu 24+ sources format)
sed -i 's|URIs: http://|URIs: https://|g' /etc/apt/sources.list.d/ubuntu.sources

# Option B (legacy sources format)
sed -i 's|http://|https://|g' /etc/apt/sources.list
```

Then refresh the CA certificates:

```bash
update-ca-certificates --fresh
```

---

## 6. Install Dependencies

### Navigate to project directory and set proxy

```bash
cd /app/workspace/code/pixelsplat_Sim2Real

export http_proxy=http://127.0.0.1:3128
export https_proxy=http://127.0.0.1:3128
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

### Install system packages

```bash
apt-get update
apt-get install -y git wget \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libgomp1 \
  libxrender1 \
  libsm6
```

### Configure Git proxy

```bash
git config --global http.proxy http://127.0.0.1:3128
git config --global https.proxy http://127.0.0.1:3128
```

### Install Python requirements

```bash
pip install -r requirements.txt
```

### Install additional Python packages

```bash
pip install numpy==1.24
pip install open3d==0.18.0
pip install moviepy==1.0.3
pip install scikit-video==1.1.11
pip install nuscenes-devkit
```

### Install OpenCV (with contrib)

OpenCV must be installed in a specific order to avoid conflicts:

```bash
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
pip install opencv-contrib-python==4.7.0.72
```

### Install Gaussian Rasterization

```bash
pip install --no-build-isolation git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
```

---

## 7. Download Pretrained Weights

### DINO ViT-Base (patch 8)

```bash
wget --no-check-certificate \
  https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth

mkdir -p ~/.cache/torch/hub/checkpoints
mv dino_vitbase8_pretrain.pth ~/.cache/torch/hub/checkpoints/
```

### DINO ResNet-50

```bash
mkdir -p /root/.cache/torch/hub/checkpoints
wget --no-check-certificate \
  https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth \
  -O /root/.cache/torch/hub/checkpoints/dino_resnet50_pretrain.pth
```

### VGG-16

```bash
wget --no-check-certificate \
  https://download.pytorch.org/models/vgg16-397923af.pth

mkdir -p ~/.cache/torch/hub/checkpoints
mv vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/
```

### Alternative: Download via Python (if wget SSL fails)

```bash
mkdir -p /tmp/torch_cache/hub/checkpoints
cd /tmp/torch_cache/hub/checkpoints

python3 -c "
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
torch.hub.download_url_to_file(
    'https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth',
    'dino_vitbase8_pretrain.pth'
)
"

export TORCH_HOME=/tmp/torch_cache
export MPLCONFIGDIR=/tmp/matplotlib_config
```

---

## 8. Alternative: Build from Dockerfile

If the `Dockerfile` and `requirements.txt` are up to date, you can build directly from the project directory:

```bash
podman build -t pixelsplat-custom:latest .
```

---

## 9. Run Training

Set environment variables and launch training with W&B online logging:

```bash
export HOME=/tmp
export WANDB_INSECURE_DISABLE_SSL=true
export HTTP_PROXY=http://127.0.0.1:3128
export HTTPS_PROXY=http://127.0.0.1:3128

TORCH_HOME=/tmp/torch_cache \
MPLCONFIGDIR=/tmp/matplotlib_config \
python3 -m src.main +experiment=seed4d.yaml mode=train
```