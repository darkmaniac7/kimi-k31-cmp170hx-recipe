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
- **KV pool:** 399,384 tokens across the four stages (`PART=28,22,22,21`, `KVGIB_LIST=7.90,7.20,7.20,7.60`, `max_model_len=131072` default or up to `262144` single-stream, `max_num_seqs=4`, 3.05x concurrency at 128K).

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
