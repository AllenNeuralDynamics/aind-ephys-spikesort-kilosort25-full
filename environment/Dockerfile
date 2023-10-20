# hash:sha256:9b2d86ce40313198648da967c4b1f5ac29647f3fae84847c0b77cc2f7fa6b428
FROM registry.codeocean.allenneuraldynamics.org/codeocean/kilosort2_5-compiled-base:latest

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

# install libvips
RUN apt-get update \
    && apt-get install -y libvips
# install default fonts
RUN apt-get install -y msttcorefonts fonts-freefont-ttf

RUN pip install -U --no-cache-dir --ignore-installed jupyter

# needed for correct functioning of sortingview
RUN pip install -U --no-cache-dir --ignore-installed \
    pyvips==2.2.1 \
    cryptography==37.0.4 \
    wavpack-numcodecs==0.1.4 \
    numba==0.56.4 \
    aind-data-schema==0.15.11

COPY postInstall /
RUN /postInstall
