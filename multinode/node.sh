#!/usr/bin/env bash
# Launch one vLLM multiprocess pipeline node:
#   node.sh <layout.env> <node-rank> <node-ip> <ifname> <gpu-indices> <kv-gib>
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/host.env" ] || { echo "Missing $HERE/host.env (copy from host.env.example)" >&2; exit 2; }
# shellcheck disable=SC1091
. "$HERE/host.env"

LAYOUT_FILE="${1:?usage: node.sh <layout.env> <rank> <node-ip> <ifname> <gpus> <kv-gib>}"
NR="$2"
MY_IP="$3"
IFNAME="$4"
GPUS="$5"
KVGIB="$6"

# shellcheck disable=SC1090
. "$LAYOUT_FILE"

NN="${#NODES[@]}"
TREE="$RUNTIME_DIR/vllm-k31"
ENVD="$RUNTIME_DIR/env"
VENV="$ENVD/venv-gpu"
CUDA_HOME_C="$ENVD/cuda13"
VIEW="$VIEWS_ROOT/${VTAG}-s${NR}"
CACHE="$STATE_DIR/cache-r${NR}"
LOGS="$STATE_DIR/logs"
CNAME="k31-${TAG}-n${NR}"

if [ "$NR" = "0" ]; then
  UTIL="$UTIL_STAGE0"
  MEM="$MEM_STAGE0"
  HEADLESS=()
else
  UTIL="$UTIL_TAIL"
  MEM="$MEM_TAIL"
  HEADLESS=(--headless)
fi

PREFIX_ARGS=(--no-enable-prefix-caching)
[ "${PREFIX_CACHING:-0}" = "1" ] && PREFIX_ARGS=(--enable-prefix-caching)

SPEC_ARGS=()
if [ "${SPEC:-none}" = "dspark" ]; then
  SPECJSON="{\"method\":\"dspark\",\"model\":\"$DSPARK_DIR\",\"kv_cache_dtype\":\"bfloat16\",\"num_speculative_tokens\":${SPEC_NUM_TOKENS},\"adaptive_k\":{\"tiers\":${SPEC_TIERS},\"confidence\":{\"thresholds\":${SPEC_THRESHOLDS},\"hysteresis\":0.12,\"accept_weight\":0.12,\"max_age\":2},\"log_interval\":64}}"
  SPEC_ARGS=(--speculative-config "$SPECJSON")
fi

IB_ARGS=()
[ -e /dev/infiniband ] && IB_ARGS=(--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1)

IO_ARGS=()
[ -n "${NVME_READ_CAP:-}" ] && IO_ARGS=(--device-read-bps "$NVME_READ_CAP")

KVB="$(python3 -c "print(int(float('$KVGIB') * 2**30))")"
PRE="export VIRTUAL_ENV='$VENV' PATH='$VENV/bin:$CUDA_HOME_C/bin':\"\$PATH\"; exec '$VENV/bin/python' -m vllm.entrypoints.cli.main serve \"\$@\""

# shellcheck disable=SC2206
EXTRA_ARR=(${EXTRA_SERVE_ARGS:-})

CMD=(
  docker run -d --name "$CNAME" --hostname "$CNAME" --init
  --gpus "\"device=$GPUS\""
  --network host --ipc host --shm-size 32g --pids-limit 8192
  --memory "$MEM"
  "${IB_ARGS[@]}" "${IO_ARGS[@]}"
  -e "VLLM_HOST_IP=$MY_IP"
  -e "NCCL_SOCKET_IFNAME=$IFNAME"
  -e "GLOO_SOCKET_IFNAME=$IFNAME"
  -e "VLLM_PP_LAYER_PARTITION=$PART"
  -e "NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
  -e "PYTHONPATH=$TREE"
  -e PYTHONDONTWRITEBYTECODE=1
  -e "CUDA_HOME=$CUDA_HOME_C"
  -e "HOME=$CACHE/home"
  -e "VLLM_CACHE_ROOT=$CACHE/vllm"
  -e "TRITON_CACHE_DIR=$CACHE/triton"
  -e "TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor"
  -e "CUDA_CACHE_PATH=$CACHE/cuda-jit"
  -e CUDA_CACHE_MAXSIZE=4294967296
  "${ENGINE_ENV[@]}"
  -v "$TREE:$TREE:ro"
  -v "$ENVD:$ENVD:ro"
  -v "$VIEWS_ROOT:$VIEWS_ROOT:ro"
  -v "$WEIGHTS_ROOT:$WEIGHTS_ROOT:ro"
  -v "$CACHE:$CACHE"
  -v "$LOGS:$LOGS"
  --entrypoint bash "$IMG" -c "$PRE" k31
  "$VIEW"
  --served-model-name "$SERVED_MODEL_NAME"
  --load-format k31q --quantization k31q --kv-cache-dtype "${KV_DTYPE:-bfloat16}"
  --attention-backend CUSTOM --language-model-only --trust-remote-code
  --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP"
  --distributed-executor-backend mp
  --nnodes "$NN" --node-rank "$NR" --master-addr "$MASTER_IP" --master-port "$MASTER_PORT"
  "${HEADLESS[@]}"
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$UTIL"
  --kv-cache-memory-bytes "$KVB"
  --max-num-seqs "$MAX_SEQS"
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  "${PREFIX_ARGS[@]}"
  --compilation-config "{\"cudagraph_capture_sizes\":[$CG_SIZES]}"
  --host "$API_BIND_HOST" --port "$API_PORT"
  "${SPEC_ARGS[@]}"
  "${EXTRA_ARR[@]}"
)

if [ "${DRYRUN:-0}" = "1" ]; then
  echo "[DRYRUN rank $NR/$NN on $MY_IP GPUs=$GPUS VIEW=$VIEW KVGIB=$KVGIB]"
  [ -d "$TREE/vllm" ] || { echo "  MISSING $TREE/vllm" >&2; exit 1; }
  [ -x "$VENV/bin/python" ] || { echo "  MISSING $VENV/bin/python" >&2; exit 1; }
  [ -s "$VIEW/k31q_manifest.json" ] || { echo "  MISSING $VIEW/k31q_manifest.json" >&2; exit 1; }
  printf '  %q' "${CMD[@]}"
  echo
  exit 0
fi

mkdir -p "$CACHE" "$LOGS"
if docker ps -a --format '{{.Names}}' | grep -qx "$CNAME"; then
  echo "Container $CNAME already exists; run multinode/down.sh first." >&2
  exit 2
fi
"${CMD[@]}"
echo "Started $CNAME on $MY_IP (GPUs $GPUS, view $VIEW)"
