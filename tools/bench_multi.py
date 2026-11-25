#!/usr/bin/env python3
"""k31_bench_multi.py: multi-prompt decode benchmark and offline A/B analyzer for K3.1 vLLM.

Drop-in compatible with k31_bench.py env vars:
  KURL          OpenAI-compatible server base URL (default http://127.0.0.1:8041)
  KMODEL        Model name (default kimi-k3.1)
  KOUT          Output directory for JSONL logs (default ./logs/bench)
  KINDS         Comma-separated kind order/subset (default code,prose,structured)
  KTEMP         Sampling temperature (default 0)
  KTOPP         top_p sent only when KTEMP > 0 (default 1.0)
  KSEED         seed sent only when KTEMP > 0 (default 1234)
  KPROMPTS      Prompt JSONL path (default prompts_decode_multi.jsonl beside this script)
  KWARMUP       Set to 1 to run 1 untimed warmup prompt before timed runs (default 0)
  KWARMUP_TOK   Max tokens for the untimed warmup prompt (default 64)
  KSLEEP        Seconds to wait before reading post-request /metrics (default 1.0)

Modes:
  health
  decode-multi <tag> [max_tokens=512] [n_per_kind=6]
  decode       <tag> [max_tokens=512]             (alias for decode-multi <tag> <max_tokens> 1)
  summarize    [--json] <jsonl_a> [jsonl_b]

Zero third-party dependencies (Python 3 stdlib only). Changes no engine state.
"""
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8041"
DEFAULT_MODEL = "kimi-k3.1"
DEFAULT_OUT = "./logs/bench"
DEFAULT_KINDS = "code,prose,structured"
DEFAULT_PROMPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts_decode_multi.jsonl")
NOTHINK = {"thinking": False}
DEFAULT_PER_POS_LEN = 12


def _cfg():
    return {
        "url": os.environ.get("KURL", DEFAULT_URL).rstrip("/"),
        "model": os.environ.get("KMODEL", DEFAULT_MODEL),
        "out": os.environ.get("KOUT", DEFAULT_OUT),
        "kinds": [k.strip() for k in os.environ.get("KINDS", DEFAULT_KINDS).split(",") if k.strip()],
        "temp": float(os.environ.get("KTEMP", "0")),
        "topp": float(os.environ.get("KTOPP", "1.0")),
        "seed": int(os.environ.get("KSEED", "1234")),
        "prompts": os.environ.get("KPROMPTS", DEFAULT_PROMPTS),
        "warmup": os.environ.get("KWARMUP", "0").strip().lower() in ("1", "true", "yes", "on"),
        "warmup_tok": int(os.environ.get("KWARMUP_TOK", "64")),
        "sleep_s": float(os.environ.get("KSLEEP", "1.0")),
    }


def req(path, body=None, timeout=3600, base_url=None):
    url = (base_url if base_url is not None else _cfg()["url"]).rstrip("/") + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    r = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=timeout)


def metrics(base_url=None):
    m = {}
    try:
        with req("/metrics", timeout=30, base_url=base_url) as resp:
            txt = resp.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)[:200]}
    for line in txt.splitlines():
        if (
            line.startswith("vllm:spec_decode")
            or line.startswith("vllm:generation_tokens_total")
            or line.startswith("vllm:prompt_tokens_total")
        ):
            k, v = line.rsplit(" ", 1)
            name = k.split("{")[0]
            if "per_pos" in name and 'position="' in k:
                pos = k.split('position="', 1)[1].split('"', 1)[0]
                name = f"{name}[{pos}]"
            m[name] = m.get(name, 0.0) + float(v)
    return m


def rec(tag, record, out_dir=None):
    target_dir = out_dir if out_dir is not None else _cfg()["out"]
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, f"{tag}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return path


def load_prompts(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj or "kind" not in obj or "prompt" not in obj:
                raise ValueError(f"{path}:{lineno}: prompt entry must have id, kind, and prompt")
            items.append(obj)
    return items


def select_prompts(all_prompts, kinds, n_per_kind):
    by_kind = {}
    for p in all_prompts:
        by_kind.setdefault(p["kind"], []).append(p)
    selected = []
    for k in kinds:
        if k not in by_kind:
            raise ValueError(f"unknown kind {k!r}; available kinds: {sorted(by_kind.keys())}")
        pool = by_kind[k]
        selected.extend(pool[:n_per_kind] if n_per_kind > 0 else pool)
    return selected


def stream_chat(prompt, max_tokens, cfg=None, extra=None):
    c = cfg if cfg is not None else _cfg()
    body = {
        "model": c["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": c["temp"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": NOTHINK,
    }
    if c["temp"] > 0:
        body.update({"top_p": c["topp"], "seed": c["seed"]})
    if extra:
        body.update(extra)
    t0 = time.time()
    first = last = None
    text, usage, finish, chunks = [], None, None, 0
    with req("/v1/chat/completions", body, base_url=c["url"]) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    now = time.time()
                    first = first or now
                    last = now
                    text.append(piece)
                    chunks += 1
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return {
        "t0": t0,
        "first": first,
        "last": last,
        "text": "".join(text),
        "usage": usage,
        "finish": finish,
        "chunks": chunks,
    }


def _extract_per_pos(dm):
    prefix = "vllm:spec_decode_num_accepted_tokens_per_pos_total["
    max_idx = DEFAULT_PER_POS_LEN - 1
    found = {}
    for k, v in dm.items():
        if k.startswith(prefix) and k.endswith("]"):
            idx_str = k[len(prefix):-1]
            if idx_str.isdigit():
                idx = int(idx_str)
                found[idx] = v
                if idx > max_idx:
                    max_idx = idx
    return [found.get(i, 0) for i in range(max_idx + 1)]


def run_one_prompt(tag, prompt_entry, max_tokens, cfg):
    kind_group = prompt_entry["kind"]
    prompt_id = prompt_entry["id"]
    m0 = metrics(base_url=cfg["url"])
    s = stream_chat(prompt_entry["prompt"], max_tokens, cfg=cfg)
    if cfg["sleep_s"] > 0:
        time.sleep(cfg["sleep_s"])
    m1 = metrics(base_url=cfg["url"])
    dm = {x: m1.get(x, 0.0) - m0.get(x, 0.0) for x in m1 if not x.startswith("_")}
    n = (s["usage"] or {}).get("completion_tokens") or 0
    ttft = (s["first"] - s["t0"]) if s["first"] is not None else None
    dec_s = (s["last"] - s["first"]) if (s["first"] is not None and s["last"] is not None) else None
    tps = (n - 1) / dec_s if dec_s and dec_s > 0 and n > 1 else None
    drafts = dm.get("vllm:spec_decode_num_drafts_total", 0.0)
    acc = dm.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    dtok = dm.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    steps = drafts if drafts > 0 else max(n - 1, 0)
    per_pos = _extract_per_pos(dm)
    r = {
        "tag": tag,
        "kind": "decode-" + kind_group,
        "kind_group": kind_group,
        "prompt_id": prompt_id,
        "time": time.strftime("%FT%T"),
        "temperature": cfg["temp"],
        "completion_tokens": n,
        "prompt_tokens": (s["usage"] or {}).get("prompt_tokens"),
        "ttft_s": ttft,
        "decode_s": dec_s,
        "decode_tps": tps,
        "finish": s["finish"],
        "spec": {
            "drafts": drafts,
            "draft_tokens": dtok,
            "accepted": acc,
            "tokens_per_step": (1.0 + acc / drafts) if drafts > 0 else None,
            "per_pos": per_pos,
        },
        "ms_per_step": (1000.0 * dec_s / steps) if dec_s and dec_s > 0 and steps > 0 else None,
        "text": s["text"],
    }
    if cfg["temp"] > 0:
        r["top_p"] = cfg["topp"]
        r["seed"] = cfg["seed"]
    return r


def format_prompt_line(r):
    tag = r["tag"]
    pid = r.get("prompt_id") or r.get("kind_group") or r.get("kind")
    n = r.get("completion_tokens") or 0
    ttft = r.get("ttft_s")
    tps = r.get("decode_tps")
    ms = r.get("ms_per_step")
    spec = r.get("spec") or {}
    drafts = spec.get("drafts") or 0
    acc = spec.get("accepted") or 0
    dtok = spec.get("draft_tokens") or 0
    tok_step = spec.get("tokens_per_step")
    finish = r.get("finish")
    if tps is not None and ttft is not None and ms is not None:
        sp = (
            f" | spec steps {drafts:.0f} accepted {acc:.0f}/{dtok:.0f} tokens/step {tok_step:.2f}"
            if drafts > 0 and tok_step is not None
            else ""
        )
        return f"{tag} decode {pid}: n={n} ttft {ttft:.2f}s decode {tps:.2f} tok/s ms/step {ms:.2f}{sp} finish {finish}"
    txt = (r.get("text") or "")[:120]
    return f"{tag} decode {pid}: n={n} finish {finish} (no timing) text[:120]={txt!r}"


def decode_multi(tag, max_tokens=512, n_per_kind=6, cfg=None):
    c = cfg if cfg is not None else _cfg()
    all_prompts = load_prompts(c["prompts"])
    selected = select_prompts(all_prompts, c["kinds"], n_per_kind)
    if not selected:
        raise ValueError("no prompts selected")

    if c["warmup"]:
        wp = selected[0]
        wtok = min(max_tokens, max(1, c["warmup_tok"]))
        ws = stream_chat(wp["prompt"], wtok, cfg=c)
        if c["sleep_s"] > 0:
            time.sleep(c["sleep_s"])
        wn = (ws["usage"] or {}).get("completion_tokens") or 0
        print(
            f"{tag} warmup {wp['id']}: n={wn} finish {ws['finish']} (untimed, not logged)",
            flush=True,
        )

    records = []
    out_path = None
    for entry in selected:
        r = run_one_prompt(tag, entry, max_tokens, c)
        out_path = rec(tag, r, out_dir=c["out"])
        records.append(r)
        print(format_prompt_line(r), flush=True)

    summary = summarize_records(records, label=tag, kind_order=c["kinds"])
    print("", flush=True)
    print(format_summary(summary), flush=True)
    if out_path:
        print(f"wrote {len(records)} records -> {out_path}", flush=True)
    return records, summary


def _percentile(sorted_vals, p):
    m = len(sorted_vals)
    if m == 0:
        return None
    if m == 1:
        return float(sorted_vals[0])
    pos = p * (m - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return float(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]))


def _trimmed_mean(sorted_vals):
    m = len(sorted_vals)
    if m == 0:
        return None
    if m < 3:
        return float(statistics.mean(sorted_vals))
    trim = max(1, int(m * 0.15))
    if 2 * trim >= m:
        trim = (m - 1) // 2
    core = sorted_vals[trim:m - trim] if trim > 0 else sorted_vals
    return float(statistics.mean(core))


def _record_kind_group(r):
    kg = r.get("kind_group")
    if kg:
        return str(kg)
    kind = str(r.get("kind") or "")
    if kind.startswith("decode-"):
        return kind[len("decode-"):]
    return None


def load_decode_jsonl(path):
    records = []
    seen_per_kind = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            r = json.loads(line)
            kg = _record_kind_group(r)
            if not kg:
                continue
            seen_per_kind[kg] = seen_per_kind.get(kg, 0) + 1
            if not r.get("kind_group"):
                r = dict(r)
                r["kind_group"] = kg
            if not r.get("prompt_id"):
                r = dict(r)
                r["prompt_id"] = f"{kg}-{seen_per_kind[kg]}"
            records.append(r)
    return records


def _aggregate_group(recs):
    tps_vals = sorted(float(r["decode_tps"]) for r in recs if r.get("decode_tps") is not None)
    ms_vals = sorted(float(r["ms_per_step"]) for r in recs if r.get("ms_per_step") is not None)
    ttft_vals = sorted(float(r["ttft_s"]) for r in recs if r.get("ttft_s") is not None)

    sum_tok_m1 = 0.0
    sum_dec_s = 0.0
    sum_steps = 0.0
    sum_dec_s_steps = 0.0
    sum_drafts = 0.0
    sum_draft_tokens = 0.0
    sum_accepted = 0.0
    max_pos_len = 0

    for r in recs:
        n = r.get("completion_tokens") or 0
        dec_s = r.get("decode_s")
        if dec_s is not None and dec_s > 0 and n > 1:
            sum_tok_m1 += (n - 1)
            sum_dec_s += float(dec_s)
        spec = r.get("spec") or {}
        drafts = float(spec.get("drafts") or 0.0)
        dtok = float(spec.get("draft_tokens") or 0.0)
        acc = float(spec.get("accepted") or 0.0)
        steps = drafts if drafts > 0 else max(n - 1, 0)
        if dec_s is not None and dec_s > 0 and steps > 0:
            sum_steps += steps
            sum_dec_s_steps += float(dec_s)
        if drafts > 0:
            sum_drafts += drafts
            sum_draft_tokens += dtok
            sum_accepted += acc
            pp = spec.get("per_pos") or []
            if len(pp) > max_pos_len:
                max_pos_len = len(pp)

    sum_per_pos = [0.0] * max_pos_len
    if sum_drafts > 0 and max_pos_len > 0:
        for r in recs:
            spec = r.get("spec") or {}
            if float(spec.get("drafts") or 0.0) > 0:
                pp = spec.get("per_pos") or []
                for i, val in enumerate(pp):
                    sum_per_pos[i] += float(val or 0.0)

    pooled_per_pos_rates = []
    active_k = 0
    if sum_drafts > 0 and max_pos_len > 0:
        inferred_k = int(round(sum_draft_tokens / sum_drafts)) if sum_draft_tokens > 0 else 0
        last_nz = max((i + 1 for i, c in enumerate(sum_per_pos) if c > 0), default=0)
        active_k = min(max_pos_len, max(inferred_k, last_nz))
        pooled_per_pos_rates = [sum_per_pos[i] / sum_drafts for i in range(active_k)]

    q1 = _percentile(tps_vals, 0.25)
    q3 = _percentile(tps_vals, 0.75)
    iqr = (q3 - q1) if (q1 is not None and q3 is not None) else None
    n_length = sum(1 for r in recs if r.get("finish") == "length")
    n_short = sum(1 for r in recs if r.get("finish") != "length")

    return {
        "n_prompts": len(recs),
        "n_timed": len(tps_vals),
        "n_length": n_length,
        "n_short": n_short,
        "median_tps": float(statistics.median(tps_vals)) if tps_vals else None,
        "mean_tps": float(statistics.mean(tps_vals)) if tps_vals else None,
        "trimmed_mean_tps": _trimmed_mean(tps_vals),
        "q1_tps": q1,
        "q3_tps": q3,
        "iqr_tps": iqr,
        "min_tps": tps_vals[0] if tps_vals else None,
        "max_tps": tps_vals[-1] if tps_vals else None,
        "pooled_tps": (sum_tok_m1 / sum_dec_s) if sum_dec_s > 0 else None,
        "median_ms_per_step": float(statistics.median(ms_vals)) if ms_vals else None,
        "pooled_ms_per_step": (1000.0 * sum_dec_s_steps / sum_steps) if sum_steps > 0 else None,
        "median_ttft_s": float(statistics.median(ttft_vals)) if ttft_vals else None,
        "sum_tokens_m1": sum_tok_m1,
        "sum_decode_s": sum_dec_s,
        "sum_drafts": sum_drafts,
        "sum_draft_tokens": sum_draft_tokens,
        "sum_accepted": sum_accepted,
        "pooled_tokens_per_step": ((sum_accepted + sum_drafts) / sum_drafts) if sum_drafts > 0 else None,
        "pooled_accept_rate": (sum_accepted / sum_draft_tokens) if sum_draft_tokens > 0 else None,
        "active_k": active_k,
        "pooled_per_pos_counts": sum_per_pos[:active_k],
        "pooled_per_pos_rates": pooled_per_pos_rates,
        "prompt_ids": [r.get("prompt_id") for r in recs],
    }


def summarize_records(records, label="", kind_order=None):
    by_kind = {}
    discovered = []
    for r in records:
        kg = _record_kind_group(r)
        if not kg:
            continue
        if kg not in by_kind:
            by_kind[kg] = []
            discovered.append(kg)
        by_kind[kg].append(r)

    if kind_order:
        ordered = [k for k in kind_order if k in by_kind]
        for k in discovered:
            if k not in ordered:
                ordered.append(k)
    else:
        default_pref = ["code", "prose", "structured"]
        ordered = [k for k in default_pref if k in by_kind]
        for k in discovered:
            if k not in ordered:
                ordered.append(k)

    groups = {k: _aggregate_group(by_kind[k]) for k in ordered}
    all_recs = [r for k in ordered for r in by_kind[k]]
    overall = _aggregate_group(all_recs) if all_recs else _aggregate_group([])
    tags = sorted({str(r.get("tag")) for r in all_recs if r.get("tag")})
    temps = sorted({float(r["temperature"]) for r in all_recs if r.get("temperature") is not None})
    return {
        "label": label or (",".join(tags) if tags else "unknown"),
        "tags": tags,
        "temperatures": temps,
        "kind_order": ordered,
        "kinds": groups,
        "overall": overall,
    }


def _fmt(val, spec=".2f", fallback="-"):
    if val is None:
        return fallback
    return format(val, spec)


def _fmt_delta(a, b, spec="+.2f", pct=True, fallback="-"):
    if a is None or b is None:
        return fallback
    d = b - a
    s = format(d, spec)
    if pct and a != 0:
        s += f" ({(d / a) * 100.0:+.1f}%)"
    return s


def format_summary(summary):
    lines = []
    temps = ",".join(f"{t:g}" for t in summary["temperatures"]) if summary["temperatures"] else "?"
    lines.append(f"=== Summary: {summary['label']} (n={summary['overall']['n_prompts']}, T={temps}) ===")
    header = (
        f"{'kind':<11} {'n':>4} {'len':>5} {'med_tps':>8} {'mean_tps':>8} {'trim_tps':>8} "
        f"{'pool_tps':>8} {'iqr':>6} {'min..max':>13} {'med_ms':>7} {'tok/st':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    rows = [(k, summary["kinds"][k]) for k in summary["kind_order"]]
    if len(summary["kind_order"]) > 1:
        rows.append(("ALL", summary["overall"]))

    for name, g in rows:
        mm = (
            f"{g['min_tps']:.1f}..{g['max_tps']:.1f}"
            if g["min_tps"] is not None and g["max_tps"] is not None
            else "-"
        )
        len_str = f"{g['n_length']}/{g['n_prompts']}"
        lines.append(
            f"{name:<11} {g['n_prompts']:>4} {len_str:>5} "
            f"{_fmt(g['median_tps']):>8} {_fmt(g['mean_tps']):>8} {_fmt(g['trimmed_mean_tps']):>8} "
            f"{_fmt(g['pooled_tps']):>8} {_fmt(g['iqr_tps']):>6} {mm:>13} "
            f"{_fmt(g['median_ms_per_step']):>7} {_fmt(g['pooled_tokens_per_step']):>6}"
        )

    max_k = max((g["active_k"] for _, g in rows), default=0)
    if max_k > 0:
        lines.append("")
        pos_hdr = f"{'kind':<11} {'drafts':>7} {'acc/dtok':>9} " + " ".join(f"{'p' + str(i):>6}" for i in range(max_k))
        lines.append("Pooled per-position acceptance rate:")
        lines.append(pos_hdr)
        lines.append("-" * len(pos_hdr))
        for name, g in rows:
            if g["sum_drafts"] <= 0:
                continue
            acc_pct = f"{g['pooled_accept_rate'] * 100.0:.1f}%" if g["pooled_accept_rate"] is not None else "-"
            rates = []
            for i in range(max_k):
                if i < len(g["pooled_per_pos_rates"]):
                    rates.append(f"{g['pooled_per_pos_rates'][i] * 100.0:5.1f}%")
                else:
                    rates.append(f"{'-':>6}")
            lines.append(f"{name:<11} {g['sum_drafts']:>7.0f} {acc_pct:>9} " + " ".join(rates))

    short_kinds = [f"{k}({g['n_short']})" for k, g in summary["kinds"].items() if g["n_short"] > 0]
    if short_kinds:
        lines.append(f"NOTE: early finishes (finish != length): {', '.join(short_kinds)}")
    return "\n".join(lines)


def compare_summaries(recs_a, recs_b, label_a="baseline", label_b="candidate"):
    sa = summarize_records(recs_a, label=label_a)
    sb = summarize_records(recs_b, label=label_b)

    kinds = []
    for k in sa["kind_order"] + sb["kind_order"]:
        if k not in kinds:
            kinds.append(k)

    idx_a = {r["prompt_id"]: r for r in recs_a if r.get("prompt_id") and r.get("decode_tps") is not None}
    idx_b = {r["prompt_id"]: r for r in recs_b if r.get("prompt_id") and r.get("decode_tps") is not None}

    kind_deltas = {}
    for k in kinds:
        ga = sa["kinds"].get(k)
        gb = sb["kinds"].get(k)
        kind_deltas[k] = _compute_group_delta(ga, gb, idx_a, idx_b, kind_filter=k)

    overall_delta = _compute_group_delta(sa["overall"], sb["overall"], idx_a, idx_b, kind_filter=None)
    return {
        "baseline": sa,
        "candidate": sb,
        "kind_order": kinds,
        "deltas": kind_deltas,
        "overall_delta": overall_delta,
    }


def _compute_group_delta(ga, gb, idx_a, idx_b, kind_filter=None):
    def _d(key):
        va = ga.get(key) if ga else None
        vb = gb.get(key) if gb else None
        if va is None or vb is None:
            return {"a": va, "b": vb, "abs": None, "pct": None}
        diff = vb - va
        pct = (diff / va * 100.0) if va != 0 else None
        return {"a": va, "b": vb, "abs": diff, "pct": pct}

    shared_ids = sorted(set(idx_a.keys()) & set(idx_b.keys()))
    if kind_filter is not None:
        shared_ids = [pid for pid in shared_ids if idx_a[pid].get("kind_group") == kind_filter]

    paired_pct = []
    wins = 0
    for pid in shared_ids:
        ta = idx_a[pid]["decode_tps"]
        tb = idx_b[pid]["decode_tps"]
        if ta and ta > 0:
            paired_pct.append((tb - ta) / ta * 100.0)
        if tb > ta:
            wins += 1

    ra = (ga or {}).get("pooled_per_pos_rates") or []
    rb = (gb or {}).get("pooled_per_pos_rates") or []
    klen = max(len(ra), len(rb))
    per_pos_pp = []
    for i in range(klen):
        va = ra[i] if i < len(ra) else None
        vb = rb[i] if i < len(rb) else None
        per_pos_pp.append((vb - va) * 100.0 if (va is not None and vb is not None) else None)

    return {
        "n_a": (ga or {}).get("n_prompts", 0),
        "n_b": (gb or {}).get("n_prompts", 0),
        "median_tps": _d("median_tps"),
        "pooled_tps": _d("pooled_tps"),
        "median_ms_per_step": _d("median_ms_per_step"),
        "pooled_ms_per_step": _d("pooled_ms_per_step"),
        "pooled_tokens_per_step": _d("pooled_tokens_per_step"),
        "paired_n": len(shared_ids),
        "paired_wins": wins,
        "paired_median_pct": float(statistics.median(paired_pct)) if paired_pct else None,
        "per_pos_a": ra,
        "per_pos_b": rb,
        "per_pos_delta_pp": per_pos_pp,
    }


def format_comparison(comp):
    sa = comp["baseline"]
    sb = comp["candidate"]
    lines = [
        f"=== A/B Comparison: A={sa['label']} (n={sa['overall']['n_prompts']}) vs B={sb['label']} (n={sb['overall']['n_prompts']}) ===",
    ]
    header = (
        f"{'kind':<11} {'nA/nB':>5} {'med_tps A->B (delta)':>30} {'pool_tps A->B (delta)':>30} "
        f"{'med_ms A->B (delta)':>26} {'tok/st A->B (delta)':>25} {'wins':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    rows = [(k, comp["deltas"][k]) for k in comp["kind_order"]]
    if len(comp["kind_order"]) > 1:
        rows.append(("ALL", comp["overall_delta"]))

    def _ab(cell, spec=".2f", dspec="+.2f"):
        a, b, d, pct = cell["a"], cell["b"], cell["abs"], cell["pct"]
        if a is None or b is None:
            return f"{_fmt(a, spec)}->{_fmt(b, spec)}"
        p_s = f" {pct:+.1f}%" if pct is not None else ""
        return f"{a:{spec}}->{b:{spec}} ({d:{dspec}}{p_s})"

    for name, d in rows:
        wins_s = f"{d['paired_wins']}/{d['paired_n']}" if d["paired_n"] > 0 else "-"
        lines.append(
            f"{name:<11} {d['n_a']:>2}/{d['n_b']:<2} "
            f"{_ab(d['median_tps']):>30} {_ab(d['pooled_tps']):>30} "
            f"{_ab(d['median_ms_per_step']):>26} {_ab(d['pooled_tokens_per_step']):>25} {wins_s:>6}"
        )

    max_k = max((len(d["per_pos_delta_pp"]) for _, d in rows), default=0)
    if max_k > 0:
        lines.append("")
        lines.append("Pooled per-position acceptance delta (B - A in percentage points):")
        pos_hdr = f"{'kind':<11} " + " ".join(f"{'p' + str(i):>7}" for i in range(max_k))
        lines.append(pos_hdr)
        lines.append("-" * len(pos_hdr))
        for name, d in rows:
            if not d["per_pos_delta_pp"]:
                continue
            vals = []
            for i in range(max_k):
                if i < len(d["per_pos_delta_pp"]) and d["per_pos_delta_pp"][i] is not None:
                    vals.append(f"{d['per_pos_delta_pp'][i]:+7.1f}")
                else:
                    vals.append(f"{'-':>7}")
            lines.append(f"{name:<11} " + " ".join(vals))
    return "\n".join(lines)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        sys.exit(
            "usage: k31_bench_multi.py <health | decode-multi <tag> [max_tokens=512] [n_per_kind=6] "
            "| decode <tag> [max_tokens=512] | summarize [--json] <jsonl_a> [jsonl_b]>"
        )
    mode = args[0]
    if mode == "health":
        c = _cfg()
        with req("/health", timeout=10, base_url=c["url"]) as h:
            status = h.status
        with req("/v1/models", timeout=10, base_url=c["url"]) as m:
            models_txt = json.dumps(json.loads(m.read().decode("utf-8")))[:300]
        print(status, models_txt)
        return 0

    if mode == "decode-multi":
        if len(args) < 2:
            sys.exit("usage: k31_bench_multi.py decode-multi <tag> [max_tokens=512] [n_per_kind=6]")
        tag = args[1]
        max_tokens = int(args[2]) if len(args) > 2 else 512
        n_per_kind = int(args[3]) if len(args) > 3 else 6
        decode_multi(tag, max_tokens=max_tokens, n_per_kind=n_per_kind)
        return 0

    if mode == "decode":
        if len(args) < 2:
            sys.exit("usage: k31_bench_multi.py decode <tag> [max_tokens=512]")
        tag = args[1]
        max_tokens = int(args[2]) if len(args) > 2 else 512
        decode_multi(tag, max_tokens=max_tokens, n_per_kind=1)
        return 0

    if mode == "summarize":
        as_json = False
        rest = args[1:]
        if "--json" in rest:
            as_json = True
            rest = [x for x in rest if x != "--json"]
        if len(rest) not in (1, 2):
            sys.exit("usage: k31_bench_multi.py summarize [--json] <jsonl_a> [jsonl_b]")
        def _label(p):
            base = os.path.basename(p)
            if p.startswith("/dev/fd/") or base.isdigit():
                return ""
            return base

        path_a = rest[0]
        recs_a = load_decode_jsonl(path_a)
        label_a = _label(path_a)
        if len(rest) == 1:
            sa = summarize_records(recs_a, label=label_a)
            print(json.dumps(sa, indent=2) if as_json else format_summary(sa))
            return 0
        path_b = rest[1]
        recs_b = load_decode_jsonl(path_b)
        label_b = _label(path_b)
        comp = compare_summaries(recs_a, recs_b, label_a=label_a, label_b=label_b)
        if as_json:
            print(json.dumps(comp, indent=2))
        else:
            print(format_summary(comp["baseline"]))
            print("")
            print(format_summary(comp["candidate"]))
            print("")
            print(format_comparison(comp))
        return 0

    sys.exit(f"unknown mode {mode}")


if __name__ == "__main__":
    raise SystemExit(main())
