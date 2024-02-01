# hash:sha256:a22e567f77d3078b482a068b32eeed530556bcd56e006a6a93d7ea9bc2df0f6a
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
    spikeinterface[full,widgets]==0.99.1 \
    wavpack-numcodecs==0.1.5

COPY postInstall /
RUN /postInstall
