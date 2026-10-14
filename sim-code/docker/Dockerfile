# HM3D-M3 simulation image: Habitat 0.1.7 (headless, EGL) + PyTorch 2.4.1/CUDA 11.8
# on Python 3.8.20. The repository itself is NOT copied in; compose.yaml bind-mounts
# sim-code/ at /workspace/sim.
#
# Every RUN step below mirrors, command for command, the native install that was
# verified end to end (self-tests, runner imports, EGO-Swarm GPU smoke); see
# HOW_TO_RUN.md section 1. The paper's own image built habitat-sim from source at the
# same commit (856d4b0, headless, no Bullet); here the identical-commit headless conda
# build from the aihabitat channel is used instead.
#
# Build (from sim-code/):
#   docker build -f docker/Dockerfile -t hm3d-m3:dev .
#   docker build -f docker/Dockerfile --build-arg WITH_BASELINES=0 -t hm3d-m3:lavira .
# Behind an HTTP proxy add, e.g.:
#   --network host --build-arg HTTPS_PROXY=http://127.0.0.1:7893 --build-arg HTTP_PROXY=http://127.0.0.1:7893
#
# Build args:
#   WITH_BASELINES=1   also install requirements-baselines.txt + detectron2 0.6
#                      (needed by the adapted baselines; LaViRA-M3 does not need them).
#                      GroundingDINO is always installed: every LaViRA-M3 row imports it.
#   MAX_JOBS           parallel compile jobs for GroundingDINO / detectron2

ARG CUDA_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG MICROMAMBA_URL=https://github.com/mamba-org/micromamba-releases/releases/download/2.0.5-0/micromamba-linux-64
ARG PYTHON_VERSION=3.8.20
ARG HABITAT_SIM_SPEC=habitat-sim=0.1.7=py3.8_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7
ARG TORCH_VERSION=2.4.1
ARG TORCHVISION_VERSION=0.19.1
ARG TORCHAUDIO_VERSION=2.4.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118
ARG HABITAT_LAB_COMMIT=d6ed1c0a0e786f16f261de2beafe347f4186d0d8
ARG GROUNDINGDINO_COMMIT=57535c5a79791cb76e36fdb64975271354f10251
ARG DETECTRON2_REF=v0.6
ARG WITH_BASELINES=1
ARG MAX_JOBS=8

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=C.UTF-8 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
    EGL_PLATFORM=surfaceless \
    MAGNUM_LOG=quiet \
    HABITAT_SIM_LOG=quiet \
    GLOG_minloglevel=2 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONNOUSERSITE=1 \
    MAMBA_ROOT_PREFIX=/opt/conda \
    HOME=/opt/m3home
# HOME is fixed so that caches (e.g. CLIP's ~/.cache/clip) have a known mount point.
RUN mkdir -p /opt/m3home/.cache && chmod -R a+rwx /opt/m3home

# --- system packages (compilers for GroundingDINO / detectron2, EGL runtime) --------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config git curl ca-certificates patch rsync unzip \
        libgl1 libegl1 libopengl0 libglvnd0 libgl1-mesa-dev libegl1-mesa-dev libglvnd-dev \
        libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# EGL vendor file: route libglvnd to the NVIDIA EGL driver injected by the
# NVIDIA container toolkit (headless rendering without an X server).
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# --- micromamba + Python 3.8.20 + habitat-sim 0.1.7 headless (commit 856d4b0) -------
RUN curl -fsSL ${MICROMAMBA_URL} -o /usr/local/bin/micromamba \
    && chmod +x /usr/local/bin/micromamba \
    && micromamba create -y -p /opt/conda/envs/m3 -c conda-forge -c aihabitat --override-channels \
        python=${PYTHON_VERSION} pip "${HABITAT_SIM_SPEC}" \
    && micromamba remove -y --force -p /opt/conda/envs/m3 libegl libgl libglx libglvnd \
    && micromamba clean -afy
# The solver pulls conda-forge's libglvnd in (via matplotlib -> qt-main, ffmpeg -> sdl2).
# habitat-sim's RPATH would load that libEGL, which has no NVIDIA vendor file and
# mixes with the system libOpenGL: "GL::Context: cannot retrieve OpenGL version".
# Removing it (nothing in sim-code/ uses conda's Qt / ffmpeg) makes habitat-sim use the
# system libglvnd + 10_nvidia.json.
ENV PATH=/opt/conda/envs/m3/bin:${PATH} \
    CONDA_DEFAULT_ENV=m3

# --- PyTorch 2.4.1 + cu118 -----------------------------------------------------------
RUN pip install "setuptools<70" wheel \
    && pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} --index-url ${TORCH_INDEX_URL}

# --- pinned Python requirements -----------------------------------------------------
COPY requirements.txt /opt/hm3d-m3/requirements.txt
RUN pip install -r /opt/hm3d-m3/requirements.txt

# --- habitat-lab 0.1.7 (challenge-2021) + HM3D-M3 patch -----------------------------
# Shallow fetch of the pinned commit (a full clone through a proxy is slow and flaky).
COPY docker/patches/habitat-lab-vln.patch /opt/hm3d-m3/habitat-lab-vln.patch
RUN git init -q /opt/habitat-lab \
    && cd /opt/habitat-lab \
    && git fetch -q --depth 1 https://github.com/facebookresearch/habitat-lab.git ${HABITAT_LAB_COMMIT} \
    && git -c advice.detachedHead=false checkout -q FETCH_HEAD \
    && git apply /opt/hm3d-m3/habitat-lab-vln.patch \
    && python setup.py develop --all --no-deps

# --- GroundingDINO (required by every LaViRA-M3 row; CUDA ops built with nvcc 11.8) --
# --no-deps: its setup.py would install opencv-python and supervision==0.4.0 over the
# pins; its real runtime deps (addict, yapf, ...) are in requirements.txt.
RUN git init -q /opt/GroundingDINO \
    && git -C /opt/GroundingDINO fetch -q --depth 1 https://github.com/IDEA-Research/GroundingDINO.git ${GROUNDINGDINO_COMMIT} \
    && git -C /opt/GroundingDINO -c advice.detachedHead=false checkout -q FETCH_HEAD \
    && MAX_JOBS=${MAX_JOBS} FORCE_CUDA=1 pip install --no-build-isolation --no-deps -e /opt/GroundingDINO

# --- adapted-baseline extras -------------------------------------------------------
COPY requirements-baselines.txt /opt/hm3d-m3/requirements-baselines.txt
RUN if [ "${WITH_BASELINES}" = "1" ]; then \
        pip install -r /opt/hm3d-m3/requirements-baselines.txt \
        && git init -q /opt/detectron2 \
        && git -C /opt/detectron2 fetch -q --depth 1 https://github.com/facebookresearch/detectron2.git \
            refs/tags/${DETECTRON2_REF}:refs/tags/${DETECTRON2_REF} \
        && git -C /opt/detectron2 -c advice.detachedHead=false checkout -q ${DETECTRON2_REF} \
        && MAX_JOBS=${MAX_JOBS} FORCE_CUDA=1 pip install --no-build-isolation -e /opt/detectron2 ; \
    fi

# Re-assert the pins (the source builds must not drift them) and smoke-import.
RUN pip install -r /opt/hm3d-m3/requirements.txt \
    && python -c "import torch, habitat_sim, habitat, groundingdino, segment_anything; print('habitat-sim', habitat_sim.__version__, 'torch', torch.__version__)"

WORKDIR /workspace/sim
CMD ["bash"]
