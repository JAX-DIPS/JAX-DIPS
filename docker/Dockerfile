FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN apt update && apt install -y git vim

RUN pip3 install jax jaxlib 
RUN cd /tmp \
    && git clone https://github.com/paulo-herrera/PyEVTK.git \
    && cd PyEVTK/ \
    && python setup.py install
