#!/usr/bin/env python3
"""k31q_views.py <src-dir> <view-root> <partition, e.g. 24,10,21,21,17> [--tag NAME] [--base DIR] [--stages 1,2,3]
                 [--no-drafter]
Per-stage views of a k31q checkpoint (per-layer symlink layout): view-root/NAME-s<k>
holds symlinks to the stage's layer files, codebooks, config and tokenizer files, embed on stage 0 (and on the last stage unless
--no-drafter), vision on stage 0 when the checkpoint has it, head on the last stage, and a filtered
model.safetensors.index.json and k31q_manifest.json (a stage's loader must never see names of files it does not have).

--base DIR   second source directory, searched for every file that is not in <src-dir>. This is how the 1.625 variant is
             built: <src-dir> = k31q-v1-1p625-mid (manifest and index for all 93 layers, layer files 39 to 70),
             --base = k31q-v1-1p80 (every other file). Manifest and index always come from <src-dir>.
--stages     build only these stage indices (a host holds only its own stages' files); default all.
A layer file whose size differs from the manifest's `bytes` (single-file layers) stops the build.
Removes only entries inside a view dir it is rebuilding. Prints one line per stage with layer range and GiB."""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("root"); ap.add_argument("partition")
    ap.add_argument("--tag", default="k31"); ap.add_argument("--base", default=None)
    ap.add_argument("--stages", default=None); ap.add_argument("--no-drafter", action="store_true")
    a = ap.parse_args()
    src, root = os.path.abspath(a.src), os.path.abspath(a.root)
    dirs = [src] + ([os.path.abspath(a.base)] if a.base else [])
    part = [int(x) for x in a.partition.split(",")]
    want = None if a.stages is None else {int(x) for x in a.stages.split(",")}

    def find(name):
        for d in dirs:
            p = os.path.join(d, name)
            if os.path.exists(p): return p
        return None

    man = json.load(open(src + "/k31q_manifest.json"))
    idx = json.load(open(src + "/model.safetensors.index.json"))
    n = len(man["layers"])
    if sum(part) != n: raise SystemExit(f"partition sums to {sum(part)}, checkpoint has {n} layers")
    small = sorted({e for d in dirs for e in os.listdir(d)
                    if not os.path.isdir(os.path.join(d, e)) and not e.endswith(".safetensors")
                    and e not in ("model.safetensors.index.json", "k31q_manifest.json", "DONE") and not e.startswith(".")})
    has_vision = find("vision.safetensors") is not None or "vision.safetensors" in man.get("other", {})
    first = 0
    for k, cnt in enumerate(part):
        layers = range(first, first + cnt); first += cnt
        if want is not None and k not in want: continue
        files = {f for L in layers for f in man["layers"][str(L)]["file"]} | {"codebooks.safetensors"}
        if k == 0:
            files.add("embed.safetensors")
            if has_vision: files.add("vision.safetensors")   # the wrapper class builds the tower on stage 0
        if k == len(part) - 1:
            files.add("head.safetensors")
            if not a.no_drafter: files.add("embed.safetensors")   # the DSpark draft class builds the target embedding on the last rank
        paths = {f: find(f) for f in files}
        missing = sorted(f for f, p in paths.items() if p is None)
        if missing: raise SystemExit(f"stage {k}: missing {missing[:6]}{' ...' if len(missing) > 6 else ''} ({len(missing)} files)")
        for L in layers:
            rec = man["layers"][str(L)]
            if len(rec["file"]) == 1 and rec.get("bytes"):
                got = os.path.getsize(paths[rec["file"][0]])
                if got != rec["bytes"]: raise SystemExit(f"stage {k}: {rec['file'][0]} has {got} bytes, manifest says {rec['bytes']}")
        dst = f"{root}/{a.tag}-s{k}"; os.makedirs(dst, exist_ok=True)
        for e in os.listdir(dst): os.unlink(os.path.join(dst, e))
        for f in sorted(files): os.symlink(paths[f], os.path.join(dst, f))
        for e in small: os.symlink(find(e), os.path.join(dst, e))
        sub = dict(idx); sub["weight_map"] = {t: f for t, f in idx["weight_map"].items() if f in files}
        json.dump(sub, open(dst + "/model.safetensors.index.json", "w"))
        m2 = dict(man); m2["layers"] = {str(L): man["layers"][str(L)] for L in layers}
        m2["other"] = {f: v for f, v in man["other"].items() if f in files}
        m2["stage"] = {"index": k, "of": len(part), "first_layer": layers[0], "last_layer": layers[-1]}
        json.dump(m2, open(dst + "/k31q_manifest.json", "w"))
        gib = sum(os.path.getsize(p) for p in paths.values()) / 2**30
        srcs = sorted({os.path.basename(os.path.dirname(p)) for p in paths.values()})
        print(f"VIEW {a.tag}-s{k} layers {layers[0]}-{layers[-1]} files={len(files)} GiB={gib:.3f} from={'+'.join(srcs)} dir={dst}", flush=True)


if __name__ == "__main__":
    main()
