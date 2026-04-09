# 使用 NVIDIA 官方 TensorRT 10.15.1.29 基础镜像
FROM nvcr.io/nvidia/tensorrt:24.12-py3

# 设置工作目录
WORKDIR /workspace

# 安装必要工具
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 克隆 TensorRT 源码
RUN git clone --branch release/10.15 --recursive https://github.com/NVIDIA/TensorRT.git /opt/TensorRT

# 下载并解压示例数据
RUN wget -O /tmp/sample_data.zip \
    https://github.com/NVIDIA/TensorRT/releases/download/v10.15/tensorrt_sample_data_20260203.zip && \
    unzip /tmp/sample_data.zip -d /tmp && \
    unzip /tmp/tensorrt_sample_data_20260203.zip -d /workspace/ && \
    rm /tmp/sample_data.zip /tmp/tensorrt_sample_data_20260203.zip && \
    mv /workspace/tensorrt_sample_data_20260203 /workspace/data

# 编译示例
WORKDIR /opt/TensorRT
RUN mkdir -p build && cd build && \
    cmake .. -DTRT_LIB_DIR=/opt/TensorRT/lib \
             -DTRT_OUT_DIR=$(pwd)/out \
             -DGPU_ARCHS="86" && \
    make -j$(nproc)

# 复制编译好的示例
RUN cp /opt/TensorRT/build/out/sample_onnx_mnist /workspace/

# 设置环境变量
ENV LD_LIBRARY_PATH=/opt/TensorRT/lib:$LD_LIBRARY_PATH

WORKDIR /workspace
CMD ["/bin/bash"]