FROM nvcr.io/nvidia/l4t-tensorrt:r10.3.0-devel

# Set working directory
WORKDIR /team15

# Copy the necessary files
COPY . .
RUN chmod -R 777 /team15

# Setup TensorRT
COPY ./jetson/cudnn9/lib /usr/local/cudnn/lib
COPY ./jetson/cudnn9/include /usr/local/cudnn/include

ENV LD_LIBRARY_PATH=/usr/local/cudnn/lib:${LD_LIBRARY_PATH}
ENV CPATH=/usr/local/cudnn/include:${CPATH}

# Copy tensorrt python
COPY ./jetson/tensorrtlib/python/tensorrt* /usr/lib/python3.10/dist-packages/

# Install runtime
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libopenblas-base \
    libpng-dev \
    libjpeg8 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get install -f -y ./jetson/debs/libnvinfer10_10.3.0.30-1+cuda12.5_arm64.deb
RUN apt-get install -f -y ./jetson/debs/libnvinfer-headers-dev_10.3.0.30-1+cuda12.5_arm64.deb
RUN apt-get install -f -y ./jetson/debs/libnvinfer-dev_10.3.0.30-1+cuda12.5_arm64.deb

RUN apt-get install -f -y ./jetson/debs/libnvonnxparsers10_10.3.0.30-1+cuda12.5_arm64.deb
RUN apt-get install -f -y ./jetson/debs/libnvonnxparsers-dev_10.3.0.30-1+cuda12.5_arm64.deb

RUN apt-get install -f -y ./jetson/debs/libnvinfer-lean10_10.3.0.30-1+cuda12.5_arm64.deb
RUN apt-get install -f -y ./jetson/debs/libnvinfer-plugin10_10.3.0.30-1+cuda12.5_arm64.deb
RUN apt-get install -f -y ./jetson/debs/libnvinfer-vc-plugin10_10.3.0.30-1+cuda12.5_arm64.deb
RUN apt-get install -f -y ./jetson/debs/libnvinfer-dispatch10_10.3.0.30-1+cuda12.5_arm64.deb
RUN apt-get install -f -y ./jetson/debs/libnvinfer-bin_10.3.0.30-1+cuda12.5_arm64.deb
RUN apt-get install -f -y ./jetson/debs/libnvinfer-dispatch-dev_10.3.0.30-1+cuda12.5_arm64.deb
RUN apt-get install -f -y ./jetson/debs/libnvinfer-headers-plugin-dev_10.3.0.30-1+cuda12.5_arm64.deb
RUN apt-get install -f -y ./jetson/debs/libnvinfer-lean-dev_10.3.0.30-1+cuda12.5_arm64.deb
RUN apt-get install -f -y ./jetson/debs/libnvinfer-plugin-dev_10.3.0.30-1+cuda12.5_arm64.deb
RUN apt-get install -f -y ./jetson/debs/libnvinfer-vc-plugin-dev_10.3.0.30-1+cuda12.5_arm64.deb

RUN apt-get install -f -y ./jetson/debs/libnvinfer-samples_10.3.0.30-1+cuda12.5_all.deb

RUN apt-get install -f -y ./jetson/debs/python3-libnvinfer_10.3.0.30-1+cuda12.5_arm64.deb
RUN apt-get install -f -y ./jetson/debs/python3-libnvinfer-lean_10.3.0.30-1+cuda12.5_arm64.deb
RUN apt-get install -f -y ./jetson/debs/python3-libnvinfer-dispatch_10.3.0.30-1+cuda12.5_arm64.deb
RUN apt-get install -f -y ./jetson/debs/python3-libnvinfer-dev_10.3.0.30-1+cuda12.5_arm64.deb

RUN apt-get install -f -y ./jetson/debs/tensorrt_10.3.0.30-1+cuda12.5_arm64.deb
RUN apt-get install -f -y ./jetson/debs/tensorrt-dev_10.3.0.30-1+cuda12.5_arm64.deb
RUN apt-get install -f -y ./jetson/debs/tensorrt-libs_10.3.0.30-1+cuda12.5_arm64.deb

# Install pip and wheel, upgrade pip
RUN pip install --no-cache-dir --upgrade pip wheel

# Install all wheels from the wheels directory
RUN pip install --no-cache-dir ./jetson/wheels/*.whl

# Install Python dependencies if requirements.txt exists
RUN pip install --no-cache-dir --upgrade pip && \
    if [ -f ./jetson/requirements.txt ]; then pip install --no-cache-dir -r ./jetson/requirements.txt; fi
