# Multi-Node Setup and Operational Traps

`multinode/` starts and stops the multi-node vLLM multiprocess layout across one or two hosts using plain `docker run` invocations (one container per pipeline stage).

## Files

- `host.env.example`: copy to `multinode/host.env` on every host and fill in your runtime/weights/views paths, 100 GbE link IPs and NIC interface names, NVMe read-cap block device, and per-stage GPU indices.
- `layouts/tp2_pp4_2bw_6cmp.env`: dedicated production layout (`TP2 x PP4`, `PART=28,22,22,21`, `KVGIB_LIST=7.90,7.20,7.20,7.60`, 399,384 GPU HBM KV tokens + 64 GiB host-RAM KV offload, multimodal vision up to 8 images/prompt, prefix caching, 6-bit EXL3 DSpark with 32+64 Markov shortlist, suffix hybrid, adaptive `k` tiers `1,2,3,4,5,7,9`, dynamic PipeSpec with KDA bank flip, FP32 target LM head, and direct shard argmax).
- `layouts/tp2_pp4_2bw_6cmp_coresident.env`: co-resident GPU workload layout (`PART=28,22,22,21`, `KVGIB_LIST=2.60,2.70,2.60,2.75`, ~135.5K GPU HBM KV + 64 GiB host-RAM KV offload = ~1.09M total KV pool, `MAX_BATCHED_TOKENS=1024`, post-capture PyTorch allocator caps `VLLM_CUDA_MEMORY_FRACTION_STAGE0=0.955` and `VLLM_CUDA_MEMORY_FRACTION_CMP=0.925`) keeping 3.9 to 5.5 GiB free per GPU across 90K+ prefills for pause-based co-resident workloads.
- `layouts/tp4_pp2_8cmp.env`: template for an eight-card 80 GB CMP 170HX chassis (`TP4 x PP2`, `PART=47,46`).
- `node.sh`: launches a single pipeline rank (`DRYRUN=1` prints the exact `docker run` command and verifies mounts, view manifest, image, and GPU count without starting a container). Supports `VISION=1` (`--limit-mm-per-prompt`) vs `VISION=0` (`--language-model-only`) and `KV_OFFLOAD_GIB` (`--kv-offloading-size`).
- `up.sh` / `down.sh`: launches or stops all ranks from the leader host (rank 0 locally, higher ranks locally or over SSH when `NODE_HOST != MASTER_IP`) and waits for `/health` HTTP 200.

## Critical Operational Traps

1. **Cap consumer NVMe reads (`NVME_READ_CAP` in `host.env`, e.g. `/dev/nvme0n1:450mb`):**
   Uncapped multi-process weight loads across 558.6 GiB of `.safetensors` files can overheat a consumer NVMe drive past 80 C in under three minutes and drop it off the PCIe bus. Always keep local verified copies rather than serving shards over NFS.
2. **Socket-local / PHB TP2 pairing and PCIe Gen4 x16 verification:**
   Before launching, confirm every CMP stage card reports `81920 MiB` in `nvidia-smi`, `Speed 16GT/s, Width x16` in `lspci -vv`, and working TRAP31 peer-to-peer inside its TP2 pair. Pair cards on the same CPU socket and PCIe root complex (`PHB` or `NODE` in `nvidia-smi topo -m`) with stage 1 and stage 3 on the socket hosting the 100 GbE NIC so TP2 all-reduce never crosses the inter-socket fabric. Exclude any slot that fell back to Gen3 from active TP2 pairs.
3. **Host-RAM KV offload (`--kv-offloading-size 64`) and `PYTORCH_CUDA_ALLOC_CONF`:**
   vLLM's `OffloadingConnector` rejects `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (virtual memory segments cannot back pinned DMA pages). Use `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8` instead. At `KV_OFFLOAD_GIB=64`, vLLM maps `8.56 GB` per rank in `/dev/shm` (`~17.2 GB` on the 2-GPU stage-0 host, `~51.4 GB` on the 6-GPU CMP host), so confirm host `/dev/shm` has ample headroom under `--ipc host`.
4. **Post-capture PyTorch allocator caps (`VLLM_CUDA_MEMORY_FRACTION_STAGE0` / `VLLM_CUDA_MEMORY_FRACTION_CMP`):**
   When `--kv-cache-memory-bytes` is passed, upstream vLLM skips `torch.cuda.set_per_process_memory_fraction`, which leaves `garbage_collection_threshold:0.8` inactive and lets PyTorch's caching allocator expand during long prefills (90K+ tokens) until `<1 GiB` remains free. Applying `set_per_process_memory_fraction` before CUDA graph capture fails with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` (`cuMemFree_v2` inside stream capture). `engine/vllm-k31-sm80.patch` applies `torch.cuda.empty_cache()` + `set_per_process_memory_fraction` after CUDA graph capture finishes (`0.955` on stage 0, `0.925` on CMP stages in the co-resident layout), holding 3.9 to 5.5 GiB free per GPU across 90K+ prefills. Leave both set to `0` on the dedicated max-HBM (`7.90,7.20,7.20,7.60`) layout.
5. **Keep `KV_DTYPE=bfloat16` when `SPEC=dspark` is active:**
   `engine/vllm-k31-sm80.patch` supports 528-byte NoPE `fp8_ds_mla` KV for `SPEC=none`, but when DSpark is enabled its 5 BF16 GQA layers share vLLM's unified `BLHNC` block pool on stage 3. Switching the target to `fp8_ds_mla` raises `block_size` from `3,072` to `4,608` tokens, enlarging DSpark's per-block page from 6 MiB to 9 MiB and shrinking net stage-3 KV block count.
6. **Keep `VLLM_DSPARK_PER_K_WIDTH=0` when `VLLM_K31_PIPESPEC_DYNAMIC=1` is enabled:**
   Shrinking DSpark's query block when adaptive `k` drops to 3 or lower leaves draft positions 5 to 10 with zero confidence and starves PipeSpec chunk-A lookahead.
7. **Keep `VLLM_K31_GLUE_KDA_RECORDS=0` under full CUDA graphs and `BANK_FLIP=1`:**
   Leave `VLLM_K31_GLUE_KDA_SPEC=1` enabled and `VLLM_K31_GLUE_KDA_RECORDS=0` disabled.
8. **Vision weights live on stage 0 (`VISION=1`):**
   `tools/k31q_views.py` links `vision.safetensors` (895 MB, 168 BF16 tensors) into stage 0 (`+0.85 GiB` on stage 0, zero extra VRAM on stages 1 to 3). Keep `vision.safetensors` on the stage-0 host whenever `VISION=1`.
9. **Keep `--reasoning-parser kimi_k3` and `--tool-call-parser kimi_k3` enabled:**
   Without the `kimi_k3` parsers, K3 reasoning markers appear in raw `content`. When testing with short `max_tokens`, allocate at least 128 tokens so reasoning does not consume the whole budget before `content` begins.
