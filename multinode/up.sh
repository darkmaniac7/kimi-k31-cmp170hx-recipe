#!/usr/bin/env bash
# Start all pipeline stages from the master host and wait for /health HTTP 200.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
[ -f "$HERE/host.env" ] || { echo "Missing $HERE/host.env (copy from host.env.example)" >&2; exit 2; }
# shellcheck disable=SC1091
. "$HERE/host.env"

LAYOUT_REL="${1:-multinode/layouts/tp2_pp4_2bw_6cmp.env}"
case "$LAYOUT_REL" in
  /*) LAYOUT_FILE="$LAYOUT_REL"; LAYOUT_REL="${LAYOUT_FILE#"$REPO_ROOT"/}" ;;
  *)  LAYOUT_FILE="$REPO_ROOT/$LAYOUT_REL" ;;
esac
# shellcheck disable=SC1090
. "$LAYOUT_FILE"

IFS=',' read -r -a KV_ARR <<< "$KVGIB_LIST"
NN="${#NODES[@]}"
[ "$NN" = "$PP" ] || { echo "NODES has $NN entries, layout PP=$PP" >&2; exit 2; }

for r in $(seq 0 $((NN - 1))); do
  IFS=':' read -r nip nif ngpus <<< "${NODES[$r]}"
  kv="${KV_ARR[$r]}"
  if [ "$nip" = "$MASTER_IP" ]; then
    DRYRUN="${DRYRUN:-0}" bash "$HERE/node.sh" "$LAYOUT_FILE" "$r" "$nip" "$nif" "$ngpus" "$kv"
  else
    # Assumes the repo is checked out at the same path on the remote host (override with REMOTE_REPO_DIR).
    RDIR="${REMOTE_REPO_DIR:-$REPO_ROOT}"
    # shellcheck disable=SC2086
    ssh $SSH_OPTS "${SSH_USER}@${nip}" "DRYRUN='${DRYRUN:-0}' bash '$RDIR/multinode/node.sh' '$RDIR/$LAYOUT_REL' '$r' '$nip' '$nif' '$ngpus' '$kv'"
  fi
  [ "$r" = "0" ] && [ "${DRYRUN:-0}" != "1" ] && sleep 5
done

[ "${DRYRUN:-0}" = "1" ] && exit 0

HEALTH_URL="http://127.0.0.1:${API_PORT}/health"
[ "$API_BIND_HOST" != "127.0.0.1" ] && [ "$API_BIND_HOST" != "0.0.0.0" ] && HEALTH_URL="http://${API_BIND_HOST}:${API_PORT}/health"

echo "Waiting up to ${UP_TIMEOUT_MIN:-30} minutes for $HEALTH_URL..."
t0="$(date +%s)"
limit=$(( ${UP_TIMEOUT_MIN:-30} * 60 ))
while [ $(( $(date +%s) - t0 )) -lt "$limit" ]; do
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$HEALTH_URL" || true)"
  if [ "$code" = "200" ]; then
    echo "READY after $(( $(date +%s) - t0 ))s: $HEALTH_URL -> HTTP 200"
    exit 0
  fi
  sleep 15
done

echo "TIMEOUT waiting for $HEALTH_URL; inspect container logs (docker logs k31-${TAG}-n0)." >&2
exit 1
