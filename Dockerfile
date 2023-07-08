# FROM nvcr.io/nvidia/tensorflow:22.05-tf2-py3
# FROM nvcr.io/nvidia/pytorch:22.04-py3
# for contents see https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
FROM nvcr.io/nvidia/pytorch:23.05-py3

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y git vim sudo gpustat libopenexr-dev python3-pybind11 libx11-6

# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
#     && dpkg -i cuda-keyring_1.0-1_all.deb \
#     && apt-get update \
#     && apt install libnccl2=2.12.12-1+cuda11.6 libnccl-dev=2.12.12-1+cuda11.6
# Set the NCCL version
# ENV TF_NCCL_VERSION 2.12.12
# RUN apt install -y wget libnccl2=2.12.12-1+cuda11.6 libnccl-dev=2.12.12-1+cuda11.6

COPY ./requirements.txt /opt/requirements.txt
RUN pip3 install -r /opt/requirements.txt

RUN cd /opt \
    && git clone https://github.com/paulo-herrera/PyEVTK.git \
    && cd PyEVTK/ \
    && python3 setup.py install

ENV PYTHONPATH="/opt/PyEVTK"



ENV KAOLIN_INSTALL_EXPERIMENTAL=1
ENV IGNORE_TORCH_VER=0
# ENV TORCH_CUDA_ARCH_LIST="7.0 7.5"
ENV CUB_HOME=/usr/local/cuda-*/include/
RUN cd /opt \
    && git clone --recursive https://github.com/NVIDIAGameWorks/kaolin \
    && cd kaolin \
    && git checkout v0.11.0 \
    && python3 setup.py develop


RUN cd /opt \
    && git clone https://github.com/tinyobjloader/tinyobjloader \
    && cd tinyobjloader \
    && python -m pip install .

# RUN mkdir -p /workspace/third_party \
#     && cd /workspace/third_party \
#     && git clone https://github.com/nv-tlabs/nglod.git \
#     && cd /workspace/third_party/nglod/sdf-net/lib/extensions \
#     && bash build_ext.sh
