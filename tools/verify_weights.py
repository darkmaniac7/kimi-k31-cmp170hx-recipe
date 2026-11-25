#!/usr/bin/env python3
"""Verify SHA-256 checksums and byte sizes of Kimi-K3.1-Pro-abliterated-k31q or DSpark EXL3 K6."""
import argparse
import hashlib
import json
import os
import sys


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(16 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_dspark(directory: str) -> int:
    man_path = os.path.join(directory, "k31q_dspark_manifest.json")
    man = json.load(open(man_path))
    fn = man["file"]
    fp = os.path.join(directory, fn)
    got_b = os.path.getsize(fp)
    if got_b != man["bytes"]:
        print(f"FAIL {fn}: size {got_b} != {man['bytes']}", file=sys.stderr)
        return 1
    got_sha = sha256_file(fp)
    if got_sha != man["sha256"]:
        print(f"FAIL {fn}: sha256 {got_sha} != {man['sha256']}", file=sys.stderr)
        return 1
    print(f"OK DSpark {fn}: {got_b} bytes, sha256={got_sha[:16]}...")
    return 0


def verify_target(directory: str, sizes_only: bool) -> int:
    man_path = os.path.join(directory, "k31q_manifest.json")
    man = json.load(open(man_path))
    items = []
    for fn, info in sorted(man.get("other", {}).items()):
        items.append((fn, info["bytes"], info["sha256"]))
    for lid in range(len(man["layers"])):
        entry = man["layers"][str(lid)]
        items.append((entry["file"][0], entry["bytes"], entry["sha256"][0]))

    bad = 0
    for idx, (fn, want_b, want_sha) in enumerate(items, 1):
        fp = os.path.join(directory, fn)
        if not os.path.isfile(fp):
            print(f"MISSING [{idx}/{len(items)}] {fn}", file=sys.stderr)
            bad += 1
            continue
        got_b = os.path.getsize(fp)
        if got_b != want_b:
            print(f"SIZE MISMATCH [{idx}/{len(items)}] {fn}: {got_b} != {want_b}", file=sys.stderr)
            bad += 1
            continue
        if not sizes_only:
            got_sha = sha256_file(fp)
            if got_sha != want_sha:
                print(f"SHA256 MISMATCH [{idx}/{len(items)}] {fn}: {got_sha} != {want_sha}", file=sys.stderr)
                bad += 1
                continue
        print(f"OK [{idx}/{len(items)}] {fn} ({got_b / 1e9:.3f} GB)", flush=True)

    if bad:
        print(f"FAILED: {bad} of {len(items)} files failed verification", file=sys.stderr)
        return 1
    print(f"VERIFIED: all {len(items)} shards match k31q_manifest.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", help="Path to downloaded checkpoint directory")
    ap.add_argument("--dspark", action="store_true", help="Verify DSpark EXL3 K6 directory instead of target")
    ap.add_argument("--sizes-only", action="store_true", help="Check file presence and byte counts only")
    args = ap.parse_args()
    if args.dspark:
        return verify_dspark(args.directory)
    return verify_target(args.directory, args.sizes_only)


if __name__ == "__main__":
    sys.exit(main())
