#!/usr/bin/env bash
# Build the patched Kimi K3.1 vLLM runtime tree and all 5 custom CUDA extensions
# inside the pinned upstream vLLM nightly container (no GPU required at build time).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=UPSTREAM_COMMIT
. "$REPO_ROOT/engine/UPSTREAM_COMMIT"

RUNTIME_DIR="${RUNTIME_DIR:-$REPO_ROOT/runtime}"
TREE="$RUNTIME_DIR/vllm-k31"
ENVD="$RUNTIME_DIR/env"
ARCH_LIST="${ARCH_LIST:-8.0;12.0}"
MAX_JOBS="${MAX_JOBS:-16}"
IMG="${IMG:-$CONTAINER_IMAGE}"
WHEEL_NAME="vllm-0.34.0rc1.dev1147+gef687f617-cp38-abi3-manylinux_2_28_x86_64.whl"

mkdir -p "$ENVD/wheels" "$ENVD/home" "$ENVD/pip-cache" "$ENVD/pycache" "$RUNTIME_DIR/build"

if [ ! -d "$TREE/.git" ]; then
  echo "[1/4] Cloning $UPSTREAM_REPO and checking out $UPSTREAM_COMMIT..."
  git clone --filter=blob:none "$UPSTREAM_REPO" "$TREE"
  git -C "$TREE" checkout --detach "$UPSTREAM_COMMIT"
  echo "[1/4] Applying engine/vllm-k31-sm80.patch..."
  git -C "$TREE" apply --index "$REPO_ROOT/engine/vllm-k31-sm80.patch"
else
  echo "[1/4] Existing tree at $TREE found; skipping clone/apply."
fi

if [ ! -s "$ENVD/wheels/$WHEEL_NAME" ]; then
  echo "[2/4] Downloading precompiled upstream wheel..."
  curl -fL --retry 3 -o "$ENVD/wheels/$WHEEL_NAME" "$PRECOMPILED_WHEEL_URL"
else
  echo "[2/4] Precompiled wheel already present."
fi

VQ_SM120=0
case "$ARCH_LIST" in *12.0*) VQ_SM120=1 ;; esac

LIBCUDA_MOUNT=()
if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  LIBCUDA_REAL="$(readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1)"
  LIBCUDA_MOUNT=(-v "$LIBCUDA_REAL:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro")
fi

echo "[3/4] Setting up container venv + CUDA 13 toolkit and installing editable vllm + sm80 plugin..."
docker run --rm --init \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -e HOME="$ENVD/home" -e PIP_CACHE_DIR="$ENVD/pip-cache" \
  -e PYTHONPYCACHEPREFIX="$ENVD/pycache" -e PYTHONDONTWRITEBYTECODE=1 \
  -e MAX_JOBS="$MAX_JOBS" \
  "${LIBCUDA_MOUNT[@]}" \
  -v "$RUNTIME_DIR:$RUNTIME_DIR" \
  -w "$TREE" \
  --entrypoint bash "$IMG" -c "
    set -euo pipefail
    SP=/usr/local/lib/python3.14/dist-packages
    P=\$SP/nvidia/cu13
    T='$ENVD/cuda13'
    mkdir -p \"\$T/bin\" \"\$T/lib64\"
    for f in \"\$P\"/bin/*; do ln -sfn \"\$f\" \"\$T/bin/\$(basename \"\$f\")\"; done
    [ -e /usr/local/cuda/bin/cuobjdump ] && ln -sfn /usr/local/cuda/bin/cuobjdump \"\$T/bin/cuobjdump\"
    for d in include nvvm lib; do ln -sfn \"\$P/\$d\" \"\$T/\$d\"; done
    for f in \"\$P\"/lib/*; do ln -sfn \"\$f\" \"\$T/lib64/\$(basename \"\$f\")\"; done
    ln -sfn \"\$P/lib/libcudart.so.13\" \"\$T/lib64/libcudart.so\"
    export CUDA_HOME=\"\$T\"
    export PATH=\"\$T/bin:\$PATH\"

    [ -x '$ENVD/venv-gpu/bin/python' ] || python3 -m venv --system-site-packages '$ENVD/venv-gpu'
    . '$ENVD/venv-gpu/bin/activate'
    export VLLM_USE_PRECOMPILED=1
    export VLLM_PRECOMPILED_WHEEL_LOCATION='$ENVD/wheels/$WHEEL_NAME'
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.34.0rc1.dev1147+k31
    export SETUPTOOLS_SCM_PRETEND_VERSION=0.34.0rc1.dev1147+k31
    python -c 'import setuptools_rust' 2>/dev/null || pip install 'setuptools-rust>=1.9.0'
    pip install --no-deps --no-build-isolation -e '$TREE'
    pip install --no-deps --no-build-isolation -e '$TREE/plugins/sm80'

    echo '[4/4] Building the 5 custom CUDA extensions for ARCH_LIST=$ARCH_LIST...'
    python csrc/exl3/build_standalone.py --out '$TREE/vllm' --build-dir '$RUNTIME_DIR/build/exl3' --arch-list '$ARCH_LIST'
    VQ_FORCE_SM120='$VQ_SM120' python csrc/k31_vq/build_vq_standalone.py --out-dir '$TREE/vllm'
    python csrc/k31_dense/build_dense_k31_standalone.py --out-dir '$TREE/vllm' --arch-list '$ARCH_LIST'
    python csrc/k31_glue/build_glue_standalone.py --out-dir '$TREE/vllm' --arch-list '$ARCH_LIST'
    python csrc/k31_p2p/build_p2p_standalone.py --out-dir '$TREE/vllm' --arch-list '$ARCH_LIST'
    rm -rf '$TREE/vllm/obj'

    for so in _exl3_C.abi3.so _k31_vq_C.abi3.so _k31_dense_C.abi3.so _k31_glue_C.abi3.so _k31_p2p_C.abi3.so; do
      [ -s \"$TREE/vllm/\$so\" ] || { echo \"MISSING \$so\"; exit 1; }
      ls -lh \"$TREE/vllm/\$so\"
    done
  "

echo "Build complete. Runtime tree ready at: $RUNTIME_DIR"
