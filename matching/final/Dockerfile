FROM ubuntu:18.04 

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \ 
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN curl -L -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \     
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install conda-build && \
     /opt/conda/bin/conda install numpy==1.19.2 pyyaml==5.4.1 scipy==1.6.1 ipython flask==1.1.2 mkl nltk==3.5 pandas==1.2.3 && \
     /opt/conda/bin/conda clean -ya 
ENV PATH /opt/conda/bin:$PATH
RUN conda install pytorch==1.7.1 torchvision==0.8.2 cpuonly -c pytorch && /opt/conda/bin/conda clean -ya
RUN python -m nltk.downloader punkt
RUN conda install scikit-learn==0.23.2 catboost==0.23 scrapy==2.4.1 faiss-cpu==1.7.0 -c conda-forge && /opt/conda/bin/conda clean -ya
RUN python -m pip install langdetect==1.0.8

COPY additional_data/ /additional_data/
WORKDIR /workspace
RUN chmod -R a+w /workspace
