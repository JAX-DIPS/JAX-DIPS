FROM gitlab-master.nvidia.com:5005/dl/dgx/jax:22.10-py3-devel

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y git vim sudo gpustat

COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN cd /opt \
    && git clone https://github.com/paulo-herrera/PyEVTK.git \
    && cd PyEVTK/ \
    && python3 setup.py install

ENV PYTHONPATH="/opt/PyEVTK"
