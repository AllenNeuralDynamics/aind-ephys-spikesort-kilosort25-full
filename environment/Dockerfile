# hash:sha256:6ec52530d292b68640e5ae8aa43b42227e6b6af133a2e3cd164c913af9e09bdf
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
    aind-data-schema==0.23.1 \
    cryptography==41.0.5 \
    jupyter==1.0.0 \
    pyvips==2.2.1 \
    spikeinterface[full,widgets]==0.99.1 \
    wavpack-numcodecs==0.1.5
