#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from localization_metrics import average_precision, mean_average_precision, ndcg_at_k, recall_at_k


LIST_RE = re.compile(r"\[\s*-?\d+(?:\s*,\s*-?\d+){2}\s*\]")


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate synthetic GRPO-2 pred mix by comparing predicted region lists against GT lists."
    )
    p.add_argument("--gt-json", required=True, help="Path to GT JSON (query2-only or multiturn JSON).")
    p.add_argument("--pred-json", required=True, help="Path to synthetic pred JSON (same IDs as GT).")
    p.add_argument(
        "--target-weights",
        default="wrong0:20,one_not_order:25,two_not_order:15,three_not_order:15,one_correct_order:20,two_correct_order:10,three_correct_order:5",
        help="Comma-separated bucket:weight list. Will be normalized to 100%%.",
    )
    p.add_argument("-k", type=int, default=3, help="K for Recall@K and nDCG@K (same as baseline eval).")
    p.add_argument("--save-json", default="", help="Optional path to save evaluation JSON.")
    return p.parse_args()


def _extract_list_from_text(text: str):
    m = LIST_RE.search(text or "")
    if not m:
        return None
    nums = [int(x) for x in re.findall(r"-?\d+", m.group(0))]
    if len(nums) != 3:
        return None
    return nums


def _extract_list_from_record(rec: dict):
    convs = rec.get("conversations", [])
    if isinstance(convs, list):
        # Prefer the last human turn containing [a,b,c] (works for multiturn + query2-only).
        for turn in reversed(convs):
            if not isinstance(turn, dict):
                continue
            if str(turn.get("from", "")).strip().lower() != "human":
                continue
            arr = _extract_list_from_text(str(turn.get("value", "")))
            if arr is not None:
                return arr
    return None


def _bucket(gt, pred):
    gt_pos = {v: i for i, v in enumerate(gt)}
    c = sum(1 for x in pred if x in gt_pos)
    p = sum(1 for i, x in enumerate(pred) if i < 3 and i < len(gt) and x == gt[i])

    if c == 0:
        return "wrong0"
    if c == 1:
        return "one_correct_order" if p == 1 else "one_not_order"
    if c == 2:
        return "two_correct_order" if p == 2 else "two_not_order"
    if c == 3:
        return "three_correct_order" if p == 3 else "three_not_order"
    return "invalid"


def _parse_target_weights(spec: str):
    out = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split(":", 1)
        out[k.strip()] = float(v.strip())
    s = sum(out.values())
    if s <= 0:
        raise ValueError("target weights sum must be > 0")
    return {k: (v / s) * 100.0 for k, v in out.items()}


def _pad_missing_with_impossible(pred, target_len: int = 3):
    if len(pred) >= target_len:
        return pred
    out = list(pred)
    filler = 99
    while len(out) < target_len:
        if filler not in out:
            out.append(filler)
        filler += 1
    return out


def _safe_pred_for_at_k(pred, k: int):
    if len(pred) >= k:
        return pred
    sentinel = -10**9
    return pred + [sentinel] * (k - len(pred))


def main():
    args = parse_args()
    gt_path = Path(args.gt_json).expanduser().resolve()
    pred_path = Path(args.pred_json).expanduser().resolve()
    if not gt_path.exists():
        raise FileNotFoundError(f"missing --gt-json: {gt_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"missing --pred-json: {pred_path}")

    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    pred_data = json.loads(pred_path.read_text(encoding="utf-8"))

    gt_by_id = {str(r.get("id", f"idx_{i}")): r for i, r in enumerate(gt_data)}
    pred_by_id = {str(r.get("id", f"idx_{i}")): r for i, r in enumerate(pred_data)}
    common_ids = sorted(set(gt_by_id) & set(pred_by_id))

    counts = Counter()
    bad_gt = 0
    bad_pred = 0
    preds = []
    gts = []

    for sid in common_ids:
        gt_list = _extract_list_from_record(gt_by_id[sid])
        pred_list = _extract_list_from_record(pred_by_id[sid])
        if gt_list is None:
            bad_gt += 1
            continue
        if pred_list is None:
            bad_pred += 1
            continue
        counts[_bucket(gt_list, pred_list)] += 1
        pred3 = _pad_missing_with_impossible(pred_list, target_len=3)
        preds.append(pred3)
        gts.append(gt_list)

    used = sum(counts.values())
    actual_pct = {k: (counts[k] * 100.0 / used if used else 0.0) for k in sorted(counts)}
    target_pct = _parse_target_weights(args.target_weights)

    all_keys = sorted(set(actual_pct) | set(target_pct))
    diff = {k: actual_pct.get(k, 0.0) - target_pct.get(k, 0.0) for k in all_keys}

    if preds:
        recall_vals = []
        ndcg_vals = []
        ap_vals = []
        for p, g in zip(preds, gts):
            p_at_k = _safe_pred_for_at_k(p, args.k)
            recall_vals.append(recall_at_k(p_at_k, g, args.k))
            ndcg_vals.append(ndcg_at_k(p_at_k, g, args.k))
            ap_vals.append(average_precision(p, g))
        baseline_metrics = {
            "num_samples": len(preds),
            f"recall@{args.k}": sum(recall_vals) / len(recall_vals),
            f"ndcg@{args.k}": sum(ndcg_vals) / len(ndcg_vals),
            "mean_ap": sum(ap_vals) / len(ap_vals),
            "map": mean_average_precision(preds, gts),
        }
    else:
        baseline_metrics = {
            "num_samples": 0,
            f"recall@{args.k}": 0.0,
            f"ndcg@{args.k}": 0.0,
            "mean_ap": 0.0,
            "map": 0.0,
        }

    out = {
        "gt_json": gt_path.as_posix(),
        "pred_json": pred_path.as_posix(),
        "num_gt": len(gt_data),
        "num_pred": len(pred_data),
        "num_common_ids": len(common_ids),
        "bad_gt_parse": bad_gt,
        "bad_pred_parse": bad_pred,
        "used_for_bucketing": used,
        "counts": dict(counts),
        "actual_pct": actual_pct,
        "target_pct_normalized": target_pct,
        "actual_minus_target_pct": diff,
        "baseline_localization_metrics": baseline_metrics,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.save_json:
        save_path = Path(args.save_json).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {save_path}")


if __name__ == "__main__":
    main()
