# Engine Patch and CUDA Extension Build

`engine/vllm-k31-sm80.patch` applies cleanly to `vllm-project/vllm` at commit `ef687f6179b752c028767e610addbda67ed36d03` and runs inside the pinned upstream container `vllm/vllm-openai:nightly-647a76866b4559a74e27d1c6b55fbffe85c07ce4` (Python 3.14, PyTorch 2.14.0+cu130).

## What `build-engine.sh` Produces

`bash engine/build-engine.sh` creates `./runtime/` (override with `RUNTIME_DIR=/path/to/runtime`) containing:

1. `runtime/vllm-k31/`: the patched vLLM Python tree, the precompiled upstream vLLM extensions extracted from `vllm-0.34.0rc1.dev1147+gef687f617`, and the five custom CUDA extensions compiled into `runtime/vllm-k31/vllm/`:
   - `_k31_vq_C.abi3.so`: `vq_d2_b3` / `vq_d2_b4` 2D vector-quantized MoE decode and prefill kernels (`TP=2` `I=1536` and `TP=4` `STATIC_I=768` fast-paths).
   - `_k31_dense_C.abi3.so`: narrow-N and split-K EXL3 dense linear kernels for Kimi K3.1 shapes.
   - `_k31_glue_C.abi3.so`: fused KDA RMSNorm/conv/speculative-verify kernels (`VLLM_K31_GLUE_KDA_SPEC=1`) and MLA/DSA norm + gate fusions (`VLLM_K31_GLUE_ALL=1`).
   - `_k31_p2p_C.abi3.so`: BAR1 PCIe peer-to-peer TP2 all-reduce extension.
   - `_exl3_C.abi3.so`: EXL3 lattice GEMM/MoE kernels (`sm_80` and `sm_120`).
2. `runtime/env/venv-gpu/` and `runtime/env/cuda13/`: the container virtual environment (`--system-site-packages` atop the nightly image) and symlinked CUDA 13.0 toolkit used at runtime for Triton/CUDA JIT compilation.

## Build Options

```bash
# Default: compile fatbins for both CMP 170HX (sm_80) and Blackwell (sm_120)
bash engine/build-engine.sh

# CMP-only single-host or multi-host build (sm_80 only):
ARCH_LIST="8.0" bash engine/build-engine.sh

# Custom output directory and parallel compile jobs:
RUNTIME_DIR=/srv/k31-runtime MAX_JOBS=16 bash engine/build-engine.sh
```

No visible GPU is required on the build host (`NVIDIA_VISIBLE_DEVICES=void` during compilation). On a two-host setup, build once and `rsync -a runtime/ <other-host>:/same/path/runtime/` so the container paths match on every rank.
