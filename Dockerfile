FROM python:slim

MAINTAINER Niels Cautaerts <nielscautaerts@hotmail.com>

RUN apt-get update && apt-get -y upgrade \
    && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates

# Xvfb, see https://stackoverflow.com/questions/59548454/python3-mayavi-in-docker-not-installing
RUN apt-get install -yq --no-install-recommends \
    xvfb \
    x11-utils \
    libx11-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda create -n venv && \
    conda activate venv && \
    conda install -c conda-forge mamba=0.15.3 pip=21.2.4 python=3.8.6 && \
    echo $(python --version)

RUN conda init bash && \
    . /root/.bashrc && \
    conda activate venv &&\
    mamba install -c conda-forge scikit-image=0.18.3 \
                                 matplotlib=3.4.3 \
                                 matplotlib-inline=0.1.3 \
                                 matplotlib-scalebar=0.7.2 \
                                 scikit-learn=0.24.2 \
                                 lmfit=1.0.2 \
                                 numpy=1.20.3 \
                                 numba=0.55.1 \
                                 psutil=5.8.0 \
                                 ipywidgets=7.6.5 \
                                 ipyevents=2.0.1 \
                                 scipy=1.7.1 \
                                 diffpy.structure=3.0.1 \
                                 tqdm=4.62.2 \
                                 transforms3d=0.3.1 \
                                 orix=0.8.2 \
                                 dask=2021.9.1 \
                                 h5py=3.3.0 \
                                 hyperspy=1.6.5 \
                                 pywget=3.2 \
                                 vtk=9.0.3

# Install Jupyter notebook extensions for mayavi
RUN conda init bash && . /root/.bashrc && conda activate venv && \
    mamba install -c conda-forge mayavi=4.7.2 && \
    jupyter nbextension install mayavi --py --sys-prefix && \
    jupyter nbextension enable mayavi --py --sys-prefix

# installing the custom packages
RUN conda init bash && . /root/.bashrc && conda activate venv && \
    git clone https://github.com/pyxem/diffsims.git \
    && cd diffsims && git checkout 2b75329 && pip install -e .

RUN conda init bash && . /root/.bashrc && conda activate venv && \
    git clone https://github.com/pyxem/pyxem.git \
    && cd pyxem && git checkout 30ddc1f5 && pip install -e .

# Add Tini
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]

# Copy necessary files in container
RUN mkdir src
WORKDIR src/
RUN mkdir data
COPY *.ipynb ./
COPY *.py ./
COPY data/*.ctf data/
COPY data/*.hspy data/
COPY data/*.csv data/
COPY data/*.csv data/
RUN mkdir outputs
COPY outputs/*.pickle outputs/
RUN mkdir 210903DataImages
RUN mkdir Figures

# Activate jupyter notebook
# after https://pythonspeed.com/articles/activate-conda-dockerfile/
CMD ["conda", "run", "--no-capture-output", "-n", "venv", "jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
