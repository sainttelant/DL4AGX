# SPDX-License-Identifier: Apache-2.0
ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=20.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}

LABEL maintainer="NVIDIA CORPORATION"

ENV TRT_VERSION 8.6.1.6
SHELL ["/bin/bash", "-c"]

# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# Install required libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    git \
    pkg-config \
    sudo \
    ssh \
    libssl-dev \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    devscripts \
    lintian \
    fakeroot \
    dh-make \
    build-essential

# Add PPA for gcc-5 and install it
RUN echo "deb http://archive.ubuntu.com/ubuntu trusty main restricted" >> /etc/apt/sources.list
RUN apt-get update && apt-get install -y gcc

# Install python3
RUN apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-wheel &&\
    cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip;

# Install PyPI packages
RUN pip3 install --upgrade pip
RUN apt update && apt install -y libturbojpeg libsm6 libxext6 -y
RUN pip3 install setuptools==69.5.1

# Install Cmake
RUN cd /tmp && \
    wget https://cmake.org/files/v3.14/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

# Install PyTorch

ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;6.1;8.0;8.6"
ENV FORCE_CUDA="1"
RUN pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

RUN apt install libgl1-mesa-glx libsm6 libxext6  -y
RUN MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv_full==1.7.0 mmdet==2.28.2 mmdet3d==1.0.0rc6 mmsegmentation==0.30.0
RUN pip install numpy==1.23.5 nuscenes-devkit==1.1.10 yapf==0.33.0 tensorboard==2.14.0 motmetrics==1.1.3 pandas==1.1.5
RUN pip install "opencv-python-headless<4.3" "opencv-python<=4.5" --force-reinstall
RUN pip install numba==0.53.0 numpy==1.23
RUN pip install similaritymeasures
# onnx related packages
RUN pip install onnx onnxsim onnxruntime
RUN pip install onnx onnxruntime onnx_graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
# for benchmark with tensorrt
RUN pip install pycuda numpy==1.23



WORKDIR /workspace
#COPY ./nuscenes /usr/local/lib/python3.8/dist-packages/nuscenes
RUN pip install torchmetrics==0.11.4
RUN pip install pandas==1.4.4
RUN pip install onnx==1.16.2 onnx_graphsurgeon==0.5.2


RUN pip install shapely==1.8.0


# Install TensorRT
RUN if [ "${CUDA_VERSION}" = "10.2" ] ; then \
    v="${TRT_VERSION}-1+cuda${CUDA_VERSION}" &&\
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub &&\
    apt-get update &&\
    sudo apt-get install libnvinfer8=${v} libnvonnxparsers8=${v} libnvparsers8=${v} libnvinfer-plugin8=${v} \
    libnvinfer-dev=${v} libnvonnxparsers-dev=${v} libnvparsers-dev=${v} libnvinfer-plugin-dev=${v} \
    python3-libnvinfer=${v} libnvinfer-dispatch8=${v} libnvinfer-dispatch-dev=${v} libnvinfer-lean8=${v} \
    libnvinfer-lean-dev=${v} libnvinfer-vc-plugin8=${v} libnvinfer-vc-plugin-dev=${v} \
    libnvinfer-headers-dev=${v} libnvinfer-headers-plugin-dev=${v}; \
    else \
    ver="${CUDA_VERSION%.*}" &&\
    if [ "${ver%.*}" = "12" ] ; then \
    ver="12.0"; \
    fi &&\
    v="${TRT_VERSION}-1+cuda${ver}" &&\
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub &&\
    apt-get update &&\
    sudo apt-get -y install libnvinfer8=${v} libnvonnxparsers8=${v} libnvparsers8=${v} libnvinfer-plugin8=${v} \
    libnvinfer-dev=${v} libnvonnxparsers-dev=${v} libnvparsers-dev=${v} libnvinfer-plugin-dev=${v} \
    python3-libnvinfer=${v} libnvinfer-dispatch8=${v} libnvinfer-dispatch-dev=${v} libnvinfer-lean8=${v} \
    libnvinfer-lean-dev=${v} libnvinfer-vc-plugin8=${v} libnvinfer-vc-plugin-dev=${v} \
    libnvinfer-headers-dev=${v} libnvinfer-headers-plugin-dev=${v}; \
    fi

# Set environment and working directory
ENV TRT_LIBPATH /usr/lib/x86_64-linux-gnu
ENV TRT_OSSPATH /workspace/TensorRT
ENV PATH="/workspace/TensorRT/build/out:${PATH}:/usr/local/bin/ngc-cli"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"

WORKDIR /workspace

