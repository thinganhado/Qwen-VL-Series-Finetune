#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from localization_metrics import average_precision, mean_average_precision, ndcg_at_k, recall_at_k


DEFAULT_MODEL_DIRS = [
    "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/qwen3_8b_stage1_merged_test/",
    "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/qwen25_7b_stage1_merged_test/",
    "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/qwen25_3b_stage1_merged_test/",
]
NO_NORMALIZE_DIR = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/qwen25_3b_stage1_merged_test/"


def _extract_ints(text: str) -> List[int]:
    return [int(x) for x in re.findall(r"-?\d+", text or "")]


def _first_k_unique(nums: List[int], k: int = 3) -> List[int]:
    seen = set()
    out = []
    for n in nums:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
        if len(out) == k:
            break
    return out


def _safe_pred_for_at_k(pred: List[int], k: int) -> List[int]:
    # nDCG@k implementation expects at least k items.
    if len(pred) >= k:
        return pred
    sentinel = -10**9
    return pred + [sentinel] * (k - len(pred))


def _load_records(model_dir: Path) -> List[dict]:
    files = sorted(model_dir.glob("*/json"))
    records = []
    for fp in files:
        try:
            records.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception:
            continue
    return records


def _build_preds_gts(records: List[dict], normalize_to_3_unique: bool) -> Tuple[List[List[int]], List[List[int]], Dict[str, int]]:
    preds: List[List[int]] = []
    gts: List[List[int]] = []
    stats = {
        "total_records": 0,
        "used_records": 0,
        "skipped_missing_gt": 0,
        "skipped_missing_pred": 0,
    }

    for rec in records:
        stats["total_records"] += 1
        gt = _extract_ints(str(rec.get("gt_regions", "")))
        if not gt:
            stats["skipped_missing_gt"] += 1
            continue

        pred = _extract_ints(str(rec.get("response", "")))
        if normalize_to_3_unique:
            pred = _first_k_unique(pred, k=3)
        if not pred:
            stats["skipped_missing_pred"] += 1
            continue

        gts.append(gt)
        preds.append(pred)
        stats["used_records"] += 1

    return preds, gts, stats


def _compute_metrics(preds: List[List[int]], gts: List[List[int]], k: int) -> Dict[str, float]:
    if not preds:
        return {
            "num_samples": 0,
            f"recall@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            "mean_ap": 0.0,
            "map": 0.0,
        }

    recall_vals = []
    ndcg_vals = []
    ap_vals = []
    for p, g in zip(preds, gts):
        p_at_k = _safe_pred_for_at_k(p, k)
        recall_vals.append(recall_at_k(p_at_k, g, k))
        ndcg_vals.append(ndcg_at_k(p_at_k, g, k))
        ap_vals.append(average_precision(p, g))

    return {
        "num_samples": len(preds),
        f"recall@{k}": sum(recall_vals) / len(recall_vals),
        f"ndcg@{k}": sum(ndcg_vals) / len(ndcg_vals),
        "mean_ap": sum(ap_vals) / len(ap_vals),
        "map": mean_average_precision(preds, gts),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate localization metrics on baseline SFT outputs."
    )
    parser.add_argument(
        "--model-dirs",
        nargs="+",
        default=DEFAULT_MODEL_DIRS,
        help="Model output root dirs (each containing sample_id/json files).",
    )
    parser.add_argument(
        "--no-normalize-dir",
        default=NO_NORMALIZE_DIR,
        help="Directory that should NOT apply the 3-unique-number normalization.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=3,
        help="K for Recall@K and nDCG@K.",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional output JSON path for aggregated metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    no_norm = Path(args.no_normalize_dir).resolve()
    all_results = {}

    for model_dir_raw in args.model_dirs:
        model_dir = Path(model_dir_raw).resolve()
        model_name = model_dir.name.rstrip("/\\")
        if not model_dir.exists():
            all_results[model_name] = {"error": f"missing_dir: {model_dir.as_posix()}"}
            continue

        records = _load_records(model_dir)
        apply_norm = model_dir != no_norm
        preds, gts, stats = _build_preds_gts(records, normalize_to_3_unique=apply_norm)
        metrics = _compute_metrics(preds, gts, k=args.k)

        all_results[model_name] = {
            "model_dir": model_dir.as_posix(),
            "normalization": "first_3_unique_from_response" if apply_norm else "none",
            "stats": stats,
            "metrics": metrics,
        }

    print(json.dumps(all_results, ensure_ascii=False, indent=2))
    if args.save_json:
        out = Path(args.save_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
