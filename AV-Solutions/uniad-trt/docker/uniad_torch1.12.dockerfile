FROM nvcr.io/nvidia/l4t-tensorrt:r8.5.2.2-devel
ENV DEBIAN_FRONTEND=noninteractive

# 安装必要的库
RUN apt-get update && apt-get install -y software-properties-common
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
    build-essential \
    libopenblas-dev  

# 安装 python3
RUN apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      python3-wheel &&\
    cd /usr/local/bin &&\
    rm -f python pip &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip;

# 安装 PyPI 包
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0

# 安装 Cmake
RUN apt-get update && apt-get install -y wget
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.2/cmake-3.31.2.tar.gz
RUN tar -zxvf cmake-3.31.2.tar.gz
WORKDIR cmake-3.31.2
RUN ./bootstrap && make -j16 && make install
WORKDIR /

RUN apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev
RUN pip3 install Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn

# 安装 BEVFormer_tensorrt 所需的库
# RUN apt-get remove -y python3.10 python3.10-dev python3.10-venv

# 更新包列表并安装必要的工具
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    curl \
    git

# 添加 deadsnakes PPA 并安装 Python 3.8
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 python3.8-dev python3.8-venv

RUN wget https://bootstrap.pypa.io/get-pip.py && python3.8 get-pip.py && \
    rm -f /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python3 && \
    python3 -m pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# 增加 Git 缓冲区大小
RUN git config --global http.postBuffer 1048576000

# 安装 MMLab
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;6.1;8.0;8.6;8.7"
ENV FORCE_CUDA="1"

RUN nvcc --version

RUN wget https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
RUN pip3 install torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl
RUN rm torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl

#RUN cd / && \
    #git clone https://github.com/open-mmlab/mmcv.git && \
    #cd mmcv && git checkout v1.5.0 && \
    #pip install -r requirements/optional.txt && \
    #MMCV_WITH_OPS=1 pip install -e .

RUN cd / && \
    git clone https://github.com/open-mmlab/mmdetection.git && \
    cd mmdetection && git checkout v2.25.1 && \
    pip3 install -v -e .

RUN pip install mmsegmentation>=0.20.0

# 安装 UniAD 所需的库
RUN python3 -m pip install --upgrade pip

# 安装 Python 包
RUN pip3 install google-cloud-bigquery==3.25.0 motmetrics==1.1.3 einops==0.4.1 casadi==3.6.0 pytorch-lightning==1.2.5
RUN apt-get install ffmpeg libsm6 libxext6  -y

# 安装杂项
RUN pip3 install ipython==8.12.3
RUN pip3 install scikit-image==0.21.0
RUN pip3 install yapf==0.40.1

WORKDIR /workspace
COPY ./nuscenes /usr/local/lib/python3.8/dist-packages/nuscenes
RUN pip install torchmetrics==0.11.4
RUN pip install pandas==1.4.4
RUN pip install onnx==1.16.2 onnx_graphsurgeon==0.5.2

# 设置 LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# 安装 gcc 和 g++
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y \
    gcc-9 \
    g++-9 \
    && apt-get upgrade -y libstdc++6 \
    && apt-get dist-upgrade -y

# 添加额外的源并安装 libc6
RUN echo "deb http://repo.huaweicloud.com/ubuntu-ports/ jammy main restricted universe multiverse" >> /etc/apt/sources.list.d/temp.list
RUN apt-get update && apt-get install -y libc6
RUN rm /etc/apt/sources.list.d/temp.list
RUN apt-get update

# 验证 NVIDIA 驱动是否正确安装
#RUN nvidia-smi
