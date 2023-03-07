FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && \
    apt-get install -y sudo \
                       python-is-python3 \
                       python3-pip

RUN apt-get -y install ncbi-blast+

ADD torch-requirements-p1.txt /app/
RUN pip install --no-cache-dir -r /app/torch-requirements-p1.txt
ADD torch-requirements-p2.txt /app/
RUN pip install --no-cache-dir -r /app/torch-requirements-p2.txt

COPY ./pss_worker_framework-0.1.1.tar.gz /app
RUN ls -1 /app
RUN pip install /app/pss_worker_framework-0.1.1.tar.gz
ADD ./requirements.txt /app
RUN pip install -r /app/requirements.txt


RUN echo "alias python=python3" >> ~/.bashrc
RUN alias python=python3
RUN python -m pip install pytest

ADD . /app/
RUN echo "export PYTHONPATH=$PYTHONPATH:/app"

WORKDIR /app
CMD ["python", "-u", "/app/runner.py"]
