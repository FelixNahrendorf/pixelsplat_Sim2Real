FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Copy CA certificates and update them
COPY ca-certificates ca-certificates
RUN apt-get update \
    && apt-get install --yes --no-install-recommends software-properties-common ca-certificates \
    && update-ca-certificates

# Set timezone to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    wget \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-iostreams-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libglew-dev \
    libglfw3-dev \
    libgmp-dev \
    libjpeg-dev \
    libmpfr-dev \
    libpng-dev \
    libqhull-dev \
    libusb-1.0-0-dev \
    libvtk7-dev \
    libxrandr-dev \
    libxxf86vm-dev \
    libxi-dev \
    libxinerama-dev \
    libxcursor-dev \
    pkg-config \
    qtbase5-dev \
    libtbb-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libcairo2-dev \
    libgirepository1.0-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libxrender1 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Reset DEBIAN_FRONTEND to default
ENV DEBIAN_FRONTEND=

# Set Python alias
RUN alias python=python3.10

# Install glog from source (specific version)
RUN git clone --branch v0.6.0 https://github.com/google/glog.git --single-branch && \
    cd glog && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j `nproc` && \
    make install && \
    cd ../.. && \
    rm -rf glog

# Copy requirements and wheels
COPY requirements.txt /app/requirements.txt
COPY relevant_libraries/torch_scatter-2.1.2+pt21cu121-cp310-cp310-linux_x86_64.whl /app/torch_scatter-2.1.2+pt21cu121-cp310-cp310-linux_x86_64.whl
COPY relevant_libraries/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl /app/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl

# Upgrade pip and install basic packages
RUN python -m pip install --no-cache-dir --upgrade pip setuptools pathtools promise pybind11

# Install pyparsing first (dependency fix)
RUN python -m pip install --no-cache-dir --upgrade "pyparsing>=3.1.0"

# Install main requirements with NVIDIA index for RAPIDS packages
RUN python -m pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com -r /app/requirements.txt || \
    (echo "Installation failed, trying with --force-reinstall" && \
     python -m pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com --force-reinstall --break-system-packages -r /app/requirements.txt) || \
    (echo "Trying individual package installation excluding blinker" && \
     sed '/^blinker==/d' /app/requirements.txt > /tmp/requirements_no_blinker.txt && \
     python -m pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com -r /tmp/requirements_no_blinker.txt) || \
    (echo "Final attempt: install without problematic packages and handle manually" && \
     sed -e '/^blinker==/d' -e '/^dbus-python==/d' /app/requirements.txt > /tmp/requirements_final.txt && \
     python -m pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com --ignore-installed -r /tmp/requirements_final.txt && \
     python -m pip install --no-cache-dir --force-reinstall --no-deps blinker || echo "Blinker installation skipped")

# Install custom wheels that were skipped from requirements.txt
RUN python -m pip install --no-cache-dir /app/torch_scatter-2.1.2+pt21cu121-cp310-cp310-linux_x86_64.whl
RUN python -m pip install --no-cache-dir /app/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl

# Download pretrained model weights
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    wget --no-check-certificate -O /root/.cache/torch/hub/checkpoints/dino_vitbase8_pretrain.pth \
         https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth && \
    wget --no-check-certificate -O /root/.cache/torch/hub/checkpoints/dino_resnet50_pretrain.pth \
         https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth && \
    wget --no-check-certificate -O /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth \
         https://download.pytorch.org/models/vgg16-397923af.pth

# Set working directory
WORKDIR /app

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Create a non-root user (optional, for security)
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default command
CMD ["/bin/bash"]