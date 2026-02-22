#!/usr/bin/env python3
import argparse
import json
import math
import random
import re
from collections import OrderedDict
from pathlib import Path


LIST_RE = re.compile(r"\[\s*-?\d+(?:\s*,\s*-?\d+){2}\s*\]")
CSV3_RE = re.compile(r"(?<!\d)(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)(?!\d)")


def parse_args():
    p = argparse.ArgumentParser(description="Build synthetic GRPO-2 pred JSON from GT JSON.")
    p.add_argument("--input-json", required=True, help="Path to GT JSON (query2-only or multiturn).")
    p.add_argument("--output-json", required=True, help="Path to synthetic pred JSON.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--id-min", type=int, default=1)
    p.add_argument("--id-max", type=int, default=16)
    p.add_argument(
        "--mix",
        default="wrong0:20,one_not_order:25,two_not_order:15,three_not_order:15,one_correct_order:20,two_correct_order:10,three_correct_order:5",
        help="Comma-separated bucket:weight; normalized automatically.",
    )
    return p.parse_args()


def parse_mix(spec: str):
    out = OrderedDict()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split(":", 1)
        out[k.strip()] = float(v.strip())
    if not out or sum(out.values()) <= 0:
        raise ValueError("Invalid --mix")
    return out


def extract_triplet(text: str):
    t = text or ""
    m = LIST_RE.search(t)
    if m:
        nums = [int(x) for x in re.findall(r"-?\d+", m.group(0))]
        if len(nums) == 3:
            return nums, m.span()
    m2 = CSV3_RE.search(t)
    if m2:
        return [int(m2.group(1)), int(m2.group(2)), int(m2.group(3))], m2.span()
    return None, None


def replace_triplet(text: str, new_ids, span):
    rep = f"[{new_ids[0]}, {new_ids[1]}, {new_ids[2]}]"
    return text[: span[0]] + rep + text[span[1] :]


def locate_human_turn_with_triplet(convs):
    # Prefer last human turn (query2 in multiturn), then fallback to any turn.
    for i in range(len(convs) - 1, -1, -1):
        turn = convs[i]
        if str(turn.get("from", "")).strip().lower() != "human":
            continue
        nums, span = extract_triplet(str(turn.get("value", "")))
        if nums is not None:
            return i, nums, span
    for i in range(len(convs) - 1, -1, -1):
        turn = convs[i]
        nums, span = extract_triplet(str(turn.get("value", "")))
        if nums is not None:
            return i, nums, span
    return None, None, None


def make_case(gt, case, rng, pool):
    gt = list(gt)
    gt_pos = {v: i for i, v in enumerate(gt)}

    def sample_wrong(k):
        cand = [x for x in pool if x not in set(gt)]
        if len(cand) < k:
            raise ValueError("Not enough wrong IDs; widen --id-min/--id-max.")
        return rng.sample(cand, k)

    if case == "three_correct_order":
        return gt[:]
    if case == "wrong0":
        return sample_wrong(3)
    if case == "three_not_order":
        while True:
            out = gt[:]
            rng.shuffle(out)
            if out != gt:
                return out
    if case == "one_correct_order":
        keep_idx = rng.randrange(3)
        out = [None] * 3
        out[keep_idx] = gt[keep_idx]
        wrong = sample_wrong(2)
        wi = 0
        for i in range(3):
            if out[i] is None:
                out[i] = wrong[wi]
                wi += 1
        return out
    if case == "two_correct_order":
        keep = sorted(rng.sample([0, 1, 2], 2))
        out = gt[:]
        miss = [i for i in [0, 1, 2] if i not in keep][0]
        out[miss] = sample_wrong(1)[0]
        return out
    if case == "one_not_order":
        cid = rng.choice(gt)
        wrong = sample_wrong(2)
        for _ in range(200):
            out = [cid] + wrong[:]
            rng.shuffle(out)
            if out.count(cid) == 1 and out[gt_pos[cid]] != cid:
                return out
        return wrong + [cid]
    if case == "two_not_order":
        cids = rng.sample(gt, 2)
        wid = sample_wrong(1)[0]
        for _ in range(500):
            out = cids + [wid]
            rng.shuffle(out)
            good = sum(out[gt_pos[c]] == c for c in cids)
            if good <= 1:
                return out
        return [cids[1], wid, cids[0]]
    raise ValueError(f"Unknown case: {case}")


def main():
    args = parse_args()
    in_path = Path(args.input_json).expanduser().resolve()
    out_path = Path(args.output_json).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_path}")

    data = json.loads(in_path.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    pool = list(range(args.id_min, args.id_max + 1))
    weights = parse_mix(args.mix)

    N = len(data)
    s = sum(weights.values())
    raw = {k: N * v / s for k, v in weights.items()}
    cnt = {k: int(math.floor(raw[k])) for k in weights}
    left = N - sum(cnt.values())
    frac_order = sorted(weights.keys(), key=lambda k: (raw[k] - cnt[k]), reverse=True)
    for k in frac_order[:left]:
        cnt[k] += 1

    idxs = list(range(N))
    rng.shuffle(idxs)
    assign = {}
    cursor = 0
    for k in weights:
        for i in idxs[cursor : cursor + cnt[k]]:
            assign[i] = k
        cursor += cnt[k]

    bad = 0
    for i, rec in enumerate(data):
        convs = rec.get("conversations", [])
        if not isinstance(convs, list) or not convs:
            bad += 1
            continue
        turn_idx, gt, span = locate_human_turn_with_triplet(convs)
        if gt is None:
            bad += 1
            continue
        pred = make_case(gt, assign[i], rng, pool)
        txt = str(convs[turn_idx].get("value", ""))
        convs[turn_idx]["value"] = replace_triplet(txt, pred, span)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] saved: {out_path}")
    print(f"[OK] total={N}, malformed/no-triplet={bad}")
    for k in weights:
        print(f"  {k}: {cnt[k]}")


if __name__ == "__main__":
    main()

