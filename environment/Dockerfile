# hash:sha256:e16a0c310288830acecc5999715f009ad708a26884e6e83eb7edcd26c9156643
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
    aind-data-schema==0.19.1 \
    cryptography==42.0.2 \
    jupyter==1.0.0 \
    pyvips==2.2.1 \
    spikeinterface[full,widgets]==0.100.0 \
    wavpack-numcodecs==0.1.5
