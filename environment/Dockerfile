# hash:sha256:a21c43a5df849b95d45bc939bd5d2096f229d553d4565257f60b584f36c39e8c
FROM registry.codeocean.allenneuraldynamics.org/codeocean/kilosort2_5-compiled-base:latest

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-freefont-ttf=20120503-7 \
        libffi-dev=3.2.1-8 \
        libvips \
        msttcorefonts \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.22.2 \
    cryptography==41.0.5 \
    jupyter==1.0.0 \
    pyvips==2.2.1 \
    spikeinterface[full,widgets]==0.99.1 \
    wavpack-numcodecs==0.1.5
