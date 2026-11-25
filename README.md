# Kimi K3.1 Pro on 6x 80 GB CMP 170HX + 2x RTX PRO 6000 Blackwell (`vLLM TP2 x PP4`)

Serve Moonshot AI's 2.8T-parameter **Kimi K3.1 Pro** across two hosts (2x RTX PRO 6000 Blackwell 96 GB + 6x unlocked 80 GB NVIDIA CMP 170HX at PCIe Gen4 x16 over 100 GbE) at **105.0 / 77.9 / 55.8 tok/s** single-stream decode (structured / code / prose medians, temperature 0) with a 131,072-token per-request context (up to 262,144 single-stream) and a 399,384-token KV cache pool.

This repository holds the engine patch (`sm_80` + `sm_120`), five standalone CUDA extension builders (`_k31_vq_C`, `_k31_dense_C`, `_k31_glue_C`, `_k31_p2p_C`, `_exl3_C`), per-stage checkpoint view and VRAM planning tools, multi-node Docker launchers, and the 18-prompt benchmark runner behind the [cmp.wiki recipe](https://cmp.wiki/wiki/Kimi_K3.1_Pro_on_6x_8_GB_plus_2x_RTX_PRO_6000_(vLLM_TP2_x_PP4)).

## Checkpoints and Hardware Prerequisites

| Resource | Location |
| --- | --- |
| Target checkpoint (`558.6 GiB`, `k31q` v1: 1.625-bit `vq_d2_b3` VQ MoE + 6-bit EXL3 dense) | [darkmaniac7/Kimi-K3.1-Pro-abliterated-k31q](https://huggingface.co/darkmaniac7/Kimi-K3.1-Pro-abliterated-k31q) |
| Speculative drafter (`1.83 GiB`, 6-bit EXL3 conversion of `Inferact/Kimi-K3.1-Pro-DSpark`) | [darkmaniac7/Kimi-K3.1-Pro-DSpark-EXL3-K6](https://huggingface.co/darkmaniac7/Kimi-K3.1-Pro-DSpark-EXL3-K6) |
| 80 GB driver patch (`fb-geometry-80g.patch`) and PCIe Gen4 boot kit (`tools/pcie/`) | [darkmaniac7/cmp170hx-8gb-vfi-tools](https://github.com/darkmaniac7/cmp170hx-8gb-vfi-tools) |
| Detailed hardware & serving notes | [Kimi K3.1 Pro on 6x 8 GB plus 2x RTX PRO 6000 (vLLM TP2 x PP4)](https://cmp.wiki/wiki/Kimi_K3.1_Pro_on_6x_8_GB_plus_2x_RTX_PRO_6000_(vLLM_TP2_x_PP4)) |

Even at 1.625 bits per routed expert weight, the quantized 93-layer checkpoint occupies 558.6 GiB on disk before embeddings, drafter, CUDA graphs, and KV/KDA state. Each of the six CMP 170HX cards must be unlocked to 80 GB (10 active FBPs) and trained to PCIe Gen4 x16 (`16 GT/s`), paired on the same CPU socket and root complex with TRAP31 PCIe P2P enabled.

## Measured Speeds (`TP2 x PP4`, `28,22,22,21`, Temperature 0)

| Workload / Metric | Result |
| --- | --- |
| 1 stream, median: structured / code / prose | **105.0 / 77.9 / 55.8 tok/s** (overall median 75.0, mean 78.1) |
| 1 stream, pooled: structured / code / prose | **96.9 / 78.2 / 54.7 tok/s** (overall pool 72.4) |
| Step time & accepted tokens per step | **39.4 / 40.8 / 42.8 ms**; **3.88 / 3.22 / 2.35 tok/step** (3.02 overall) |
| 1 stream without speculative decoding | 28.4 to 29.3 tok/s (34.2 to 35.2 ms/token) |
| Concurrent streams (measured without speculation) | 2 streams: 55.7 tok/s aggregate; 4 streams: 87.6 tok/s aggregate |
| Cold prefill (no prefix cache): 3.8k / 15.4k / 17k / 32.6k / 58.7k | 354 / 585 / 596 / 668 / 717 tok/s |
| KV cache capacity | **399,384 tokens** (default `MAX_LEN=131072`, 3.05x concurrency at 128K; supports up to 262,144 single-stream) |
| VRAM per card (loaded + drafter + graphs + KV) | Blackwells: 92.7 / 96 GiB; CMP stages 1 to 3: 74.5 / 75.2 / 77.5 / 80 GiB |
| Cold NVMe startup to `/health` 200 | 4 to 5 minutes |

Full measurement notes, component gains, and options that did not pay: [docs/outcomes.md](docs/outcomes.md).

## Quickstart

### 1. Build the engine tree and CUDA extensions

Both hosts run the pinned upstream image `vllm/vllm-openai:nightly-647a76866b4559a74e27d1c6b55fbffe85c07ce4` with a read-only mounted engine tree built from `vllm-project/vllm` at `ef687f6179b752c028767e610addbda67ed36d03` plus `engine/vllm-k31-sm80.patch`:

```bash
# Builds ./runtime/vllm-k31 and ./runtime/env (sm_80 + sm_120 fatbins by default;
# set ARCH_LIST="8.0" on a CMP-only deployment). No visible GPU is required at build time.
bash engine/build-engine.sh
```

Copy or rsync `./runtime/` to the second host so both hosts mount the identical engine tree and virtual environment.

### 2. Verify checkpoints and build per-stage symlink views

Download [darkmaniac7/Kimi-K3.1-Pro-abliterated-k31q](https://huggingface.co/darkmaniac7/Kimi-K3.1-Pro-abliterated-k31q) and [darkmaniac7/Kimi-K3.1-Pro-DSpark-EXL3-K6](https://huggingface.co/darkmaniac7/Kimi-K3.1-Pro-DSpark-EXL3-K6) onto local NVMe on each host (at minimum, layers 0 to 27 + `embed.safetensors` on the stage-0 host, and layers 28 to 92 + `embed.safetensors`, `head.safetensors`, `vision.safetensors`, and DSpark on the CMP host), verify SHA-256 digests, and build stage views:

```bash
python3 tools/verify_weights.py /srv/models/Kimi-K3.1-Pro-abliterated-k31q
python3 tools/verify_weights.py --dspark /srv/models/Kimi-K3.1-Pro-DSpark-EXL3-K6

# Stage 0 host (layers 0..27):
python3 tools/k31q_views.py /srv/models/Kimi-K3.1-Pro-abliterated-k31q /srv/views 28,22,22,21 --tag k31p28 --stages 0

# CMP host (stages 1, 2, 3 = layers 28..49, 50..71, 72..92):
python3 tools/k31q_views.py /srv/models/Kimi-K3.1-Pro-abliterated-k31q /srv/views 28,22,22,21 --tag k31p28 --stages 1,2,3
```

### 3. Configure hosts and launch

Copy `multinode/host.env.example` to `multinode/host.env` on both hosts, set your 100 GbE link IPs, network interfaces, paths, and socket-local GPU index pairs, test with `DRYRUN=1`, and launch:

```bash
cp multinode/host.env.example multinode/host.env
# edit multinode/host.env on both hosts
DRYRUN=1 bash multinode/up.sh multinode/layouts/tp2_pp4_2bw_6cmp.env
bash multinode/up.sh multinode/layouts/tp2_pp4_2bw_6cmp.env
```

### 4. Benchmark

```bash
KURL=http://127.0.0.1:8041 python3 tools/bench_multi.py
```

## Repository Layout

- [engine/](engine/): `vllm-k31-sm80.patch` against `vllm-project/vllm@ef687f6179`, `UPSTREAM_COMMIT`, `build-engine.sh`, and extension notes.
- [multinode/](multinode/): `host.env.example`, `node.sh`, `up.sh`, `down.sh`, `layouts/tp2_pp4_2bw_6cmp.env` (measured production layout) and `layouts/tp4_pp2_8cmp.env` (8-card CMP template), plus two-host setup and operational traps.
- [tools/](tools/): `k31q_views.py` (per-stage view builder), `k31_stage_plan.py` (VRAM capacity planner), `verify_weights.py` (manifest SHA-256 verifier), `bench_multi.py`, and `prompts_decode_multi.jsonl`.
- [docs/outcomes.md](docs/outcomes.md): measured speeds, per-component breakdown, and options that did not pay.

## Credits and Licenses

- Base model: [moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro) by Moonshot AI (Kimi K3.1-Pro License).
- Abliteration edit: [moeadham/Kimi-K3.1-Pro-abliterated](https://huggingface.co/moeadham/Kimi-K3.1-Pro-abliterated).
- Speculative drafter: [Inferact/Kimi-K3.1-Pro-DSpark](https://huggingface.co/Inferact/Kimi-K3.1-Pro-DSpark) by Inferact and the Red Hat AI Speculators team.
- Engine foundations: [vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache-2.0), [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) (`sm_80` platform, EXL3 integration, custom PCIe all-reduce), [cebeuq/glm53-cmp170hx-recipe](https://github.com/cebeuq/glm53-cmp170hx-recipe) (multi-stage PipeSpec pattern), and [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) (EXL3 lattice codebook format, MIT).
- Hardware unlock driver and boot tools: [amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) and [darkmaniac7/cmp170hx-8gb-vfi-tools](https://github.com/darkmaniac7/cmp170hx-8gb-vfi-tools).

Recipe launchers, tools, and documentation in this repository are licensed under the [MIT License](LICENSE). `engine/vllm-k31-sm80.patch` modifies Apache-2.0 vLLM code and is distributed under the [Apache License 2.0](LICENSE-APACHE) (preserving third-party notices under `plugins/sm80/licenses/`).
