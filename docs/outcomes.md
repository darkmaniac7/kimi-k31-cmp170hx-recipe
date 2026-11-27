# Measured Outcomes and Design Notes

All single-stream decode figures below were measured on 2026-11-25 against the OpenAI-compatible HTTP endpoint on stage 0 (`TP2 x PP4`, partition `28, 22, 22, 21`: two RTX PRO 6000 Blackwell 96 GB GPUs at 400 W on stage 0 over 100 GbE RoCE, and six 80 GB CMP 170HX cards at PCIe Gen4 x16 and 250 to 300 W across stages 1 to 3).

The single-stream decode benchmark (`tools/bench_multi.py` with `tools/prompts_decode_multi.jsonl`) runs 18 deterministic prompts (`temperature=0`, `max_tokens=512`): 6 structured JSON/YAML/SQL prompts, 6 code generation prompts, and 6 analytical prose prompts.

## Production Configuration (`tp2_pp4_2bw_6cmp.env`)

| Bucket | Median tok/s | Pooled tok/s | Mean tok/s | Median step (ms) | Accepted tok/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Structured (`n=6`) | 105.03 | 96.87 | 100.28 | 39.42 | 3.88 |
| Code (`n=6`) | 77.93 | 78.18 | 79.17 | 40.82 | 3.22 |
| Prose (`n=6`) | 55.75 | 54.65 | 54.69 | 42.75 | 2.35 |
| **All (`n=18`)** | **75.02** | **72.44** | **78.05** | **40.95** | **3.02** |

Additional baselines on the same hardware:
- **Without speculative decoding (`SPEC=none`, `T=0`):** 28.4 to 29.3 tok/s single stream (34.2 to 35.2 ms/token); 55.7 tok/s aggregate at 2 concurrent streams; 87.6 tok/s aggregate at 4 concurrent streams (21.9 to 22.9 tok/s per stream).
- **Cold prefill (unique prompts, prefix caching off):** 354 tok/s at 3,838 tokens, 585 tok/s at 15,352 tokens, 596 tok/s at 17,000 tokens, 668 tok/s at 32,600 tokens, and 717 tok/s at 58,746 tokens.
- **KV pool (dedicated vs co-resident profiles + 64 GiB host-RAM offload):**
  - **Dedicated (`tp2_pp4_2bw_6cmp.env`):** 399,384 GPU HBM tokens (`KVGIB_LIST=7.90,7.20,7.20,7.60`, 3.05x at 128K; up to 262,144 single-stream) plus ~952K warm prefix tokens in host RAM (`--kv-offloading-size 64`, ~1.35M total KV pool).
  - **Co-resident (`tp2_pp4_2bw_6cmp_coresident.env`):** ~135.5K GPU HBM tokens (`KVGIB_LIST=2.60,2.70,2.60,2.75`, >128K single-stream) plus ~952K warm prefix tokens in host RAM (`--kv-offloading-size 64`, **`~1.09M` total KV pool**), paired with post-capture PyTorch allocator caps (`VLLM_CUDA_MEMORY_FRACTION_STAGE0=0.955`, `VLLM_CUDA_MEMORY_FRACTION_CMP=0.925`) that hold 3.9 to 5.5 GiB free per GPU across 90K+ prefills while pause-based (`SIGSTOP`/`SIGCONT`) co-resident GPU workloads remain resident.
- **Prefix caching + host-RAM KV offload measurements (`--enable-prefix-caching --kv-offloading-size 64`):**
  - Async `GPU_to_CPU` DMA measured **18.1 GB/s** (`1.68 GB` in `92.7 ms`, `<1 s` KV paging latency on DDR4-3200 RDIMMs; `8.56 GB/rank` mmap in `/dev/shm`).
  - Turn-2 cached TTFT drops from `22.72 s` cold to `5.82 s` warm on a 14.3K prompt, and from `128.4 s` cold to **`5.03 s` warm** on a 90,163-token prompt (`25x` faster TTFT).
- **Multimodal vision (`VISION=1`, `--limit-mm-per-prompt '{"image":8}'`):**
  - Loads all 168 BF16 vision tensors (`vision.safetensors`, `+0.85 GiB` on stage 0 only, zero extra VRAM on stages 1 to 3) and handles streaming image + text prompts (verified up to `1280x720` PNG inputs).
- **Non-speculative `fp8_ds_mla` KV support:** `engine/vllm-k31-sm80.patch` also adds a 528-byte NoPE / 656-byte RoPE `fp8_ds_mla` reader to `TritonMLASparseBackend` for `SPEC=none` multi-stream deployments. When `SPEC=dspark` is active, keep `KV_DTYPE=bfloat16` (`block_size=3072`): DSpark's 5 BF16 GQA layers (`2,048 B/tok` per layer) share vLLM's unified `BLHNC` block pool on stage 3, so raising `block_size` to `4,608` under `fp8_ds_mla` enlarges DSpark's per-block page and reduces stage-3 block count.

## What Each Component Contributes

1. **`k31q` v1 hybrid VQ + 6-bit EXL3 quantization (`_k31_vq_C`, `_k31_dense_C`, `_exl3_C`):**
   - Compresses the 896 routed experts (`top_k=16`, `I=3072`) in all 92 MoE layers with 2D 3-bit vector quantization (`vq_d2_b3`, 8-entry codebook, group-128 input scales, 1.625 bpw all-in) and keeps all attention projections (69 KDA + 24 gated MLA), shared experts, routed-expert shared up/down projections, layer 0's dense MLP, and the LM head in 6-bit EXL3 (`mul1` lattice, ~35.2 dB SQNR).
   - Fits all 93 layers in 558.6 GiB on disk and 66.1 to 69.8 GiB of weights per CMP rank in the balanced `28,22,22,21` partition (keeping all 5 DSpark tap layers `[76,80,84,88,92]` local to stage 3), leaving room on 80 GB cards for CUDA graphs, the drafter, and 399,384 KV tokens.
2. **6-bit EXL3 DSpark drafter with cross-stage taps, 32+64 Markov shortlist, suffix hybrid, and adaptive `k`:**
   - Quantizing `Inferact/Kimi-K3.1-Pro-DSpark` (2.5B, 5 sliding-window GQA layers) to 6-bit EXL3 reduces its stage-3 footprint from 2.71 GiB to 1.11 GiB per TP2 rank (`VLLM_DSPARK_EXL3_K6=1`).
   - Hidden states from the 5 tapped target layers ride packed PP hops (`VLLM_PP_PACKED_HOP=1`, `VLLM_PP_HOP_CAT_PACK=1`) to stage 3.
   - Restricting DSpark's rank-512 Markov head inside CUDA graphs to a 32-base + 64-successor shortlist (`VLLM_DSPARK_MARKOV_SHORTLIST=32`) saves 0.20 to 0.49 ms/step at unchanged acceptance; combined with online prompt/output suffix matching (`VLLM_DSPARK_SUFFIX_HYBRID=1`, `min_match_len=5`) and confidence-driven adaptive draft depth (`tiers: [1,2,3,4,5,7,9]` inside an 11-token capture window), single-stream speed rises from ~29 tok/s to 105 / 78 / 56 tok/s.
3. **Dynamic confidence-gated PipeSpec with KDA bank-flip commit:**
   - Starts stage 0's next speculative chunk before stages 1 to 3 finish verifying the active chunk whenever DSpark confidence across positions 3 to 5 clears `0.28` (`VLLM_K31_PIPESPEC=1`, `VLLM_K31_PIPESPEC_DYNAMIC=1`, `VLLM_K31_PIPESPEC_CKS=3,4,5`, `VLLM_K31_PIPESPEC_MIN_CONF=0.28`).
   - Commits verified KDA recurrent states on chunk B via an in-place state-slot permutation (`VLLM_K31_PIPESPEC_BANK_FLIP=1`) instead of copying 3 MiB state blocks per KDA layer.
4. **FP32 target LM head with BF16 drafter isolation and direct shard argmax:**
   - Running the target's 6-bit EXL3 `7168 -> 163840` LM head in FP32 (`VLLM_K31_FP32_HEAD=1`) while keeping DSpark's shared LM head path in BF16 adds ~0.05 ms on stage 3, eliminates greedy BF16 tie-flipping during verification, raises code draft acceptance by 5.5 to 7.8 percentage points across early draft positions, and pairs with `VLLM_SM80_TARGET_SHARD_ARGMAX_DIRECT=1` for zero-sync per-shard argmax.

## What Did Not Pay (Keep Off)

- **`VLLM_DSPARK_PER_K_WIDTH=1` with `VLLM_K31_PIPESPEC_DYNAMIC=1`:** Narrowing DSpark's query block when `k <= 3` leaves positions 5 to 10 at zero confidence and cuts PipeSpec chunk-A lookahead hits by more than half. Leave `VLLM_DSPARK_PER_K_WIDTH=0`.
- **`VLLM_K31_GLUE_KDA_RECORDS=1` under full CUDA graphs and `BANK_FLIP=1`:** Compact KDA state records require runtime fold indices that are absent during full CUDA graph capture. Leave `VLLM_K31_GLUE_KDA_SPEC=1` and `VLLM_K31_GLUE_KDA_RECORDS=0`.
- **Top-k routed expert pruning (`VLLM_K31Q_MOE_MAX_TOPK` below 16):** Degrades output quality on sensitive prompts; keep disabled (`0`).

## Multimodal Vision, Host-RAM KV Offload, and Post-Capture Allocator Caps

- **Multimodal vision on stage 0:** Removing `--language-model-only` and setting `--limit-mm-per-prompt '{"image":8}'` activates Kimi K3.1 Pro's 27-layer vision tower (`vision.safetensors`) solely on stage 0 (`+0.85 GiB` BF16), while stages 1 to 3 instantiate meta placeholders with zero additional VRAM cost.
- **Host-RAM KV offload (`OffloadingConnector`):** `--enable-prefix-caching --kv-offloading-size 64` maps `8.56 GB/rank` in host `/dev/shm` (`~17.2 GB` on stage 0, `~51.4 GB` on the 6-GPU CMP host), expanding a compact 135.5K HBM cache into a **`~1.09M`-token** warm prefix pool with `<1 s` paging latency on DDR4-3200 RDIMMs. Do not combine with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (rejected by `OffloadingConnector`); use `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8`.
- **Post-capture PyTorch allocator caps (`gpu_worker.py`):** Under `--kv-cache-memory-bytes`, vLLM skips `torch.cuda.set_per_process_memory_fraction`, which leaves `garbage_collection_threshold:0.8` inactive and allows `CUDACachingAllocator` to grow during 90K+ prefills until `<1 GiB` remains free. Calling `set_per_process_memory_fraction` during `init_device()` triggers `cuMemFree_v2` inside CUDA graph stream capture (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`). Applying `torch.cuda.empty_cache()` and `set_per_process_memory_fraction` right after CUDA graph capture completes (`VLLM_CUDA_MEMORY_FRACTION_STAGE0=0.955`, `VLLM_CUDA_MEMORY_FRACTION_CMP=0.925`) caps allocator expansion cleanly with zero capture errors and holds 3.9 to 5.5 GiB free per GPU across 129.5K-token warmstart passes (including 90.2K single prompts).
- **Socket-local dual-PHB CMP topology:** On a 7/7 PCIe Gen4 x16 CMP chassis (3 cards on NUMA 0, 4 cards on NUMA 1 with the 100 GbE NIC), pairing stage 1 and stage 3 on NUMA 1 (`NODE` + `PHB`, both local to `mlx5_0`) and stage 2 as a `PHB` pair on NUMA 0 eliminates every cross-socket `SYS` hop and lets every container pin cleanly to its local NUMA CPUs.

