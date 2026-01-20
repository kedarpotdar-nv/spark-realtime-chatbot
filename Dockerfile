# =============================================================================
# spark-realtime-chatbot Dockerfile for DGX Spark (CUDA 13.0 / GB10)
# =============================================================================
# This Dockerfile builds CTranslate2 from source with CUDA support for the
# GB10 architecture (sm_121) and installs all dependencies.
#
# Build: docker build -t spark-realtime-chatbot .
# Run:   docker run --gpus all -p 8443:8443 spark-realtime-chatbot
# =============================================================================

FROM nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# =============================================================================
# Stage 1: System Dependencies
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python and build tools
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    cmake \
    build-essential \
    git \
    # Audio/media processing
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    # OpenCV dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # Utilities
    curl \
    openssl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 2: Build CTranslate2 from Source
# =============================================================================
# Following instructions from README.md for local ASR with CUDA support.
# CMake doesn't know Blackwell natively, so we pass CUDAARCHS=121.

WORKDIR /build

# Set CUDA architecture for Blackwell (sm_121)
ENV CUDAARCHS="121"
ENV TORCH_CUDA_ARCH_LIST="12.1"

# Clone and build CTranslate2 for Blackwell
RUN git clone --recursive https://github.com/OpenNMT/CTranslate2.git && \
    cd CTranslate2 && \
    # Patch: Comment out cuda_select_nvcc_arch_flags (CMake doesn't know Blackwell)
    # and directly set the gencode flags for sm_121
    sed -i 's/cuda_select_nvcc_arch_flags/#cuda_select_nvcc_arch_flags/' CMakeLists.txt && \
    sed -i 's/list(APPEND CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS_READABLE})/list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_121,code=sm_121")/' CMakeLists.txt && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=OFF \
        -DWITH_MKL=OFF \
        -DOPENMP_RUNTIME=NONE \
        -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Install CTranslate2 Python bindings
RUN pip3 install --break-system-packages /build/CTranslate2/python

# =============================================================================
# Stage 3: Python Dependencies
# =============================================================================

WORKDIR /app

# Upgrade pip (ignore system packages that can't be uninstalled)
RUN pip3 install --break-system-packages --upgrade --ignore-installed pip setuptools wheel

# Install PyTorch with CUDA 13.0 support
# Install torch first, then torchvision (--no-deps to avoid version conflict, torch already installed)
RUN pip3 install --break-system-packages --no-cache-dir \
    'torch==2.9.1+cu130' --extra-index-url https://download.pytorch.org/whl/cu130 && \
    pip3 install --break-system-packages --no-cache-dir \
    'torchvision==0.24.1+cu130' --no-deps --extra-index-url https://download.pytorch.org/whl/cu130

# Install faster-whisper WITHOUT dependencies (use our built CTranslate2)
# Then install its deps manually, excluding ctranslate2
RUN pip3 install --break-system-packages --no-cache-dir faster-whisper --no-deps && \
    pip3 install --break-system-packages --no-cache-dir \
    tokenizers \
    huggingface-hub \
    onnxruntime \
    av && \
    # Verify our CTranslate2 wasn't overwritten
    python3 -c "import ctranslate2; print('CTranslate2 location:', ctranslate2.__file__)"

# Install TTS (Kokoro) and related dependencies
RUN pip3 install --break-system-packages --no-cache-dir \
    kokoro \
    kokoro-tts \
    soundfile \
    scipy \
    phonemizer-fork \
    misaki \
    loguru

# Install web framework and async dependencies
RUN pip3 install --break-system-packages --no-cache-dir \
    fastapi \
    'uvicorn[standard]' \
    websockets \
    aiohttp \
    python-multipart \
    httpx

# Install DeepFace and its dependencies for face recognition
# tf-keras is required by deepface for TensorFlow backend
RUN pip3 install --break-system-packages --no-cache-dir \
    tensorflow \
    tf-keras \
    deepface \
    opencv-python \
    mtcnn \
    retina-face

# Install NLP dependencies (spaCy for text processing)
# Misaki (used by Kokoro TTS) needs en_core_web_sm for G2P
RUN pip3 install --break-system-packages --no-cache-dir \
    spacy \
    transformers && \
    python3 -m spacy download en_core_web_sm && \
    python3 -m spacy download en_core_web_trf || true

# Install remaining utility packages
RUN pip3 install --break-system-packages --no-cache-dir \
    numpy \
    pandas \
    pydantic \
    python-dotenv \
    tqdm \
    regex \
    safetensors

# =============================================================================
# Stage 4: Verify Installation
# =============================================================================

# Verify our custom CTranslate2 is still installed (not overwritten by pip)
# It should be in /build/CTranslate2/python or system site-packages
RUN python3 -c "import ctranslate2; print('CTranslate2 version:', ctranslate2.__version__); print('CTranslate2 path:', ctranslate2.__file__)"

# Verify faster-whisper can be imported
RUN python3 -c "from faster_whisper import WhisperModel; print('faster-whisper OK')"

# Verify Kokoro TTS
RUN python3 -c "from kokoro import KPipeline; print('Kokoro TTS OK')"

# =============================================================================
# Stage 5: Application Setup
# =============================================================================

# Copy application code
COPY server.py .
COPY config.py .
COPY audio.py .
COPY tools.py .
COPY prompts.py .
COPY clients/ ./clients/
COPY static/ ./static/
COPY demo_files/ ./demo_files/
COPY launch-https.sh .
RUN chmod +x launch-https.sh

# Create required directories
RUN mkdir -p audio_cache

# =============================================================================
# Environment Configuration
# =============================================================================

# Allow pip to install system-wide (needed for any runtime downloads)
ENV PIP_BREAK_SYSTEM_PACKAGES=1

ENV PORT=8443 \
    # ASR Configuration (local mode for lower latency)
    ASR_MODE=local \
    ASR_MODEL=Systran/faster-distil-whisper-large-v3 \
    ASR_DEVICE=cuda \
    ASR_COMPUTE_TYPE=float16 \
    # TTS Configuration
    TTS_DEVICE=cuda \
    KOKORO_LANG=a \
    KOKORO_VOICE=af_bella \
    KOKORO_SPEED=1.2 \
    TTS_OVERLAP=true \
    # LLM Configuration (expects external llama-server)
    LLM_SERVER_URL=http://localhost:8080/v1/chat/completions \
    LLM_MODEL=qwen3-vl \
    LLM_MAX_TOKENS=4096 \
    # Nemotron Configuration (expects external llama-server)
    NEMOTRON_SERVER_URL=http://localhost:8005/v1/chat/completions \
    NEMOTRON_MODEL=nemotron-3-nano \
    # HuggingFace cache
    HF_HOME=/root/.cache/huggingface

# Expose HTTPS port
EXPOSE 8443

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -k -f https://localhost:${PORT}/health || exit 1

# Default command - start server with local ASR and TTS overlap
CMD ["./launch-https.sh", "--local-asr", "--tts-overlap"]
