#!/usr/bin/env bash
# Stop and remove all containers of a layout across hosts.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
[ -f "$HERE/host.env" ] || { echo "Missing $HERE/host.env" >&2; exit 2; }
# shellcheck disable=SC1091
. "$HERE/host.env"

LAYOUT_REL="${1:-multinode/layouts/tp2_pp4_2bw_6cmp.env}"
case "$LAYOUT_REL" in
  /*) LAYOUT_FILE="$LAYOUT_REL" ;;
  *)  LAYOUT_FILE="$REPO_ROOT/$LAYOUT_REL" ;;
esac
# shellcheck disable=SC1090
. "$LAYOUT_FILE"

NN="${#NODES[@]}"
for r in $(seq $((NN - 1)) -1 0); do
  IFS=':' read -r nip _ _ <<< "${NODES[$r]}"
  cname="k31-${TAG}-n${r}"
  if [ "$nip" = "$MASTER_IP" ]; then
    docker rm -f "$cname" 2>/dev/null || true
  else
    # shellcheck disable=SC2086
    ssh $SSH_OPTS "${SSH_USER}@${nip}" "docker rm -f '$cname' 2>/dev/null || true"
  fi
  echo "Stopped $cname on $nip"
done
