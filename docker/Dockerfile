FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && \
    apt-get install -y sudo \
                       python-is-python3 \
                       python3-pip 

RUN pip3 install --no-cache-dir torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip install --no-cache-dir torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip install --no-cache-dir torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip install --no-cache-dir torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip install --no-cache-dir torch-geometric

RUN echo "alias python=python3" >> ~/.bashrc
RUN alias python=python3
RUN python -m pip install pytest
RUN pip install biopython
RUN pip install pandas
RUN pip install faiss-cpu
RUN pip install faiss-gpu
RUN pip install einops
RUN pip install matplotlib

RUN apt-get -y install ncbi-blast+ 

COPY /build/ /app/
RUN echo "export PYTHONPATH=$PYTHONPATH:/app"

WORKDIR /app
CMD ["python", "/app/run.py"]
