FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN apt update && apt install -y git vim

COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN cd /tmp \
    && git clone https://github.com/paulo-herrera/PyEVTK.git \
    && cd PyEVTK/ \
    && python setup.py install
