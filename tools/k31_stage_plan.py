#!/usr/bin/env python3
"""k31_stage_plan.py: pack K3.1 decoder layers into TP2 pipeline stages and show what each card holds.

Input: a k31q manifest (per-layer bytes) when one exists (--manifest), else a rate table (--rates, default = the KLX-like
placeholder: all-in bpw per layer for the routed experts) plus dense at --dense-bits. Output: contiguous layer ranges per stage,
bytes per rank, free space and the pairing waste of unequal pairs. GiB = 2^30 bytes. No GPU, no host access.
"""
import argparse, json

GIB = 2 ** 30
EXPERT_W = 896 * 3 * 3584 * 3072            # routed expert weights per MoE layer (29.595 B)
N_LAYERS = 93
MLA = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92}   # 0-based


def dense_weights(layer):
    """Dense (EXL3) weights of one layer, from the HF shapes."""
    h, lat, sh = 7168, 3584, 6144
    if layer in MLA:   # q_a 7168->1536, q_b 1536->96*192, kv_a 7168->576, kv_b 512->96*256, o 12288->7168, g 7168->12288
        attn = h * 1536 + 1536 * 96 * 192 + h * 576 + 512 * 96 * 256 + 12288 * h + h * 12288
    else:              # KDA: q,k,v,g 7168->12288, o 12288->7168, b/f_a/f_b low-rank (counted as one 7168x12288 in total)
        attn = 5 * h * 12288 + h * 12288
    if layer == 0:
        mlp = 3 * 33792 * h
    else:
        mlp = 3 * h * sh + 2 * h * lat
    return attn + mlp


def placeholder_rates():
    r = {}
    for layer in range(1, N_LAYERS):
        if 39 <= layer <= 70:
            r[layer] = 2.625
        elif layer in (9, 10, 11, 38):
            r[layer] = 2.125
        else:
            r[layer] = 1.625
    return r


def layer_bytes(args):
    if args.manifest:
        m = json.load(open(args.manifest))
        return [m["layers"][str(i)]["bytes"] for i in range(N_LAYERS)], "manifest"
    rates = placeholder_rates()
    if args.rates:
        rates.update({int(k): float(v) for k, v in json.load(open(args.rates)).items()})
    if args.uniform:
        rates = {i: args.uniform for i in range(1, N_LAYERS)}
    out = []
    for i in range(N_LAYERS):
        b = dense_weights(i) * args.dense_bits / 8 + 60e6           # + plain tensors (norms, router gate, conv, indexer)
        if i:
            b += EXPERT_W * rates[i] / 8
        out.append(b)
    return out, "rate table"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest"); p.add_argument("--rates"); p.add_argument("--uniform", type=float)
    p.add_argument("--dense-bits", type=float, default=6.0)
    p.add_argument("--stages", default="BW:90.2,90.2 C01:77.6,77.6 C23:77.6,77.6 C45:77.6,77.6 C6X:77.6,38.4",
                   help="name:usable GiB of rank 0,rank 1 per stage, in pipeline order")
    p.add_argument("--overhead", type=float, default=3.5, help="GiB per card: context, graphs, workspaces")
    p.add_argument("--kv-per-mla", type=float, default=0.46, help="GiB per card per MLA layer in the stage (256k-token pool)")
    p.add_argument("--state-per-kda", type=float, default=0.03, help="GiB per card per KDA layer (states of 4 sequences)")
    p.add_argument("--head", type=float, default=3.3, help="GiB per card on stage 0 for embed (BF16) and on the last for head + drafter")
    a = p.parse_args()
    sizes, src = layer_bytes(a)
    stages = []
    for s in a.stages.split():
        n, caps = s.split(":"); c = [float(x) for x in caps.split(",")]
        stages.append((n, c))
    print(f"layer sizes from {src}: total {sum(sizes) / GIB:.1f} GiB, experts mean "
          f"{(sum(sizes) - sum(dense_weights(i) * a.dense_bits / 8 + 60e6 for i in range(N_LAYERS))) * 8 / (EXPERT_W * 92):.3f} bpw all-in"
          if src == "rate table" else f"layer sizes from manifest: total {sum(sizes) / GIB:.1f} GiB")
    print(f"{'stage':6} {'layers':>9} {'n':>3} {'weights/rank':>12} {'kv+state':>9} {'fixed':>6} {'cap(min)':>9} {'free/rank':>9} {'pair waste':>10}")
    nxt = 0; ok = True
    for si, (name, caps) in enumerate(stages):
        cap = min(caps); waste = max(caps) - cap
        fixed = a.overhead + (a.head if si in (0, len(stages) - 1) else 0)
        first = nxt; w = 0.0; kv = 0.0
        while nxt < N_LAYERS:
            lw = sizes[nxt] / 2 / GIB
            lkv = a.kv_per_mla if nxt in MLA else a.state_per_kda
            if fixed + w + lw + kv + lkv > cap:
                break
            w += lw; kv += lkv; nxt += 1
        rng = f"{first}-{nxt - 1}" if nxt > first else "none"
        print(f"{name:6} {rng:>9} {nxt - first:3d} {w:12.1f} {kv:9.1f} {fixed:6.1f} {cap:9.1f} {cap - fixed - w - kv:9.1f} {waste:10.1f}")
    if nxt < N_LAYERS:
        ok = False
        rest = sum(sizes[nxt:]) / 2 / GIB
        print(f"DOES NOT FIT: layers {nxt}-{N_LAYERS - 1} left over, {rest:.1f} GiB per rank ({2 * rest:.1f} GiB in all)")
    print("VLLM_PP_LAYER_PARTITION=" + ",".join(str(x) for x in _part(stages, sizes, a)) if ok else "no partition")


def _part(stages, sizes, a):
    out = []; nxt = 0
    for si, (name, caps) in enumerate(stages):
        cap = min(caps); fixed = a.overhead + (a.head if si in (0, len(stages) - 1) else 0); w = kv = 0.0; first = nxt
        while nxt < N_LAYERS:
            lw = sizes[nxt] / 2 / GIB; lkv = a.kv_per_mla if nxt in MLA else a.state_per_kda
            if fixed + w + lw + kv + lkv > cap:
                break
            w += lw; kv += lkv; nxt += 1
        out.append(nxt - first)
    return out


if __name__ == "__main__":
    main()
