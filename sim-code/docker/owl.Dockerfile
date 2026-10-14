# OWL detector sidecar (services/owl_sidecar.py): OWLv2 + CLIP + SuperGlue behind
# a small Flask server. Serves the detector FOUND gate of the four adapted
# baselines. Python 3.10, torch 2.7.1+cu118, as used for the paper numbers.
#
# Build (from sim-code/):
#   docker build -f docker/owl.Dockerfile -t hm3d-m3-owl:dev .
#
# Model weights are not baked in: they are read from the Hugging Face cache
# (HF_HOME, mounted by compose.yaml) or from local directories given via
# OWL_MODEL / CLIP_MODEL. SuperGlue code + weights are user-supplied
# (non-commercial license), see HOW_TO_RUN.md.

ARG CUDA_IMAGE=nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/owl
ENV PATH=/opt/owl/bin:${PATH}

RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        --index-url ${TORCH_INDEX_URL}
COPY docker/owl-requirements.txt /opt/owl-requirements.txt
RUN pip install -r /opt/owl-requirements.txt

COPY services/owl_sidecar.py /workspace/sim/services/owl_sidecar.py
COPY docker/owl-entrypoint.sh /usr/local/bin/owl-entrypoint.sh
RUN chmod +x /usr/local/bin/owl-entrypoint.sh

WORKDIR /workspace/sim
ENTRYPOINT ["/usr/local/bin/owl-entrypoint.sh"]
