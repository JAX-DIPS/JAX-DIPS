FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y git vim sudo python3 python3-pip python3-tk

COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN cd /tmp \
    && git clone https://github.com/paulo-herrera/PyEVTK.git \
    && cd PyEVTK/ \
    && python3 setup.py install
