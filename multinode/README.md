# Multi-Node Setup and Operational Traps

`multinode/` starts and stops the multi-node vLLM multiprocess layout across one or two hosts using plain `docker run` invocations (one container per pipeline stage).

## Files

- `host.env.example`: copy to `multinode/host.env` on every host and fill in your runtime/weights/views paths, 100 GbE link IPs and NIC interface names, NVMe read-cap block device, and per-stage GPU indices.
- `layouts/tp2_pp4_2bw_6cmp.env`: the measured production layout (`TP2 x PP4`, `PART=29,21,23,20`, `KVGIB_LIST=3.40,2.60,2.90,2.80`, 6-bit EXL3 DSpark with 32+64 Markov shortlist, suffix hybrid, adaptive `k` tiers `1,2,3,4,5,7,9`, dynamic PipeSpec with KDA bank flip, FP32 target LM head, and direct shard argmax).
- `layouts/tp4_pp2_8cmp.env`: template for an eight-card 80 GB CMP 170HX chassis (`TP4 x PP2`, `PART=47,46`).
- `node.sh`: launches a single pipeline rank (`DRYRUN=1` prints the exact `docker run` command and verifies mounts, view manifest, image, and GPU count without starting a container).
- `up.sh` / `down.sh`: launches or stops all ranks from the leader host (rank 0 locally, higher ranks locally or over SSH when `NODE_HOST != MASTER_IP`) and waits for `/health` HTTP 200.

## Critical Operational Traps

1. **Cap consumer NVMe reads (`NVME_READ_CAP` in `host.env`, e.g. `/dev/nvme0n1:450mb`):**
   Uncapped multi-process weight loads across 558.6 GiB of `.safetensors` files can overheat a consumer NVMe drive past 80 C in under three minutes and drop it off the PCIe bus. Always keep local verified copies rather than serving shards over NFS.
2. **Socket-local TP2 pairing and PCIe Gen4 x16 verification:**
   Before launching, confirm every CMP stage card reports `81920 MiB` in `nvidia-smi`, `Speed 16GT/s, Width x16` in `lspci -vv`, and working TRAP31 peer-to-peer inside its TP2 pair. Keep both cards of each TP2 pair on the same CPU socket / NUMA node so custom PCIe P2P all-reduce never crosses the inter-socket fabric, and exclude any slot that fell back to Gen3 from active TP2 pairs.
3. **Keep `VLLM_DSPARK_PER_K_WIDTH=0` when `VLLM_K31_PIPESPEC_DYNAMIC=1` is enabled:**
   Shrinking DSpark's query block when adaptive `k` drops to 3 or lower leaves draft positions 5 to 10 with zero confidence and starves PipeSpec chunk-A lookahead.
4. **Keep `VLLM_K31_GLUE_KDA_RECORDS=0` under full CUDA graphs and `BANK_FLIP=1`:**
   Leave `VLLM_K31_GLUE_KDA_SPEC=1` enabled and `VLLM_K31_GLUE_KDA_RECORDS=0` disabled.
5. **Keep `--reasoning-parser kimi_k3` and `--tool-call-parser kimi_k3` enabled:**
   Without the `kimi_k3` parsers, K3 reasoning markers appear in raw `content`. When testing with short `max_tokens`, allocate at least 128 tokens so reasoning does not consume the whole budget before `content` begins.
