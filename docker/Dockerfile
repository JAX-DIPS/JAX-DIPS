FROM nvcr.io/nvidia/tensorflow:22.05-tf2-py3

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y git vim sudo gpustat

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt install libnccl2=2.12.12-1+cuda11.6 libnccl-dev=2.12.12-1+cuda11.6

# Set the NCCL version
ENV TF_NCCL_VERSION 2.12.12

# RUN apt install -y wget libnccl2=2.12.12-1+cuda11.6 libnccl-dev=2.12.12-1+cuda11.6

COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN cd /opt \
    && git clone https://github.com/paulo-herrera/PyEVTK.git \
    && cd PyEVTK/ \
    && python3 setup.py install

ENV PYTHONPATH="/opt/PyEVTK"
