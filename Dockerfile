FROM tensorflow/tensorflow:1.14.0-gpu-py3
RUN apt update && \
    apt install -y python3-mpi4py libsm6 libxext6 libxrender-dev ssh && \
    pip install stable-baselines gym asciimatics
