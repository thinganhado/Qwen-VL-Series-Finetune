#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from localization_metrics import average_precision, mean_average_precision, ndcg_at_k, recall_at_k


DEFAULT_ROOT = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_strongVLM/"


def _extract_ints(text: str) -> List[int]:
    # Normalization rule: strip all text and keep numbers by appearance order.
    return [int(x) for x in re.findall(r"-?\d+", text or "")]


def _pad_missing_with_impossible(pred: List[int], target_len: int = 3) -> List[int]:
    # Fill missing slots with impossible IDs (>16).
    if len(pred) >= target_len:
        return pred
    out = list(pred)
    filler = 99
    while len(out) < target_len:
        if filler not in out:
            out.append(filler)
        filler += 1
    return out


def _safe_pred_for_at_k(pred: List[int], k: int) -> List[int]:
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


def _build_preds_gts(records: List[dict]) -> Tuple[List[List[int]], List[List[int]], Dict[str, int]]:
    preds: List[List[int]] = []
    gts: List[List[int]] = []
    stats = {
        "total_records": 0,
        "used_records": 0,
        "skipped_missing_gt": 0,
        "padded_missing_to_3": 0,
    }

    for rec in records:
        stats["total_records"] += 1
        gt = _extract_ints(str(rec.get("gt_regions", "")))
        if not gt:
            stats["skipped_missing_gt"] += 1
            continue

        pred = _extract_ints(str(rec.get("response", "")))
        if len(pred) < 3:
            pred = _pad_missing_with_impossible(pred, target_len=3)
            stats["padded_missing_to_3"] += 1

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
        description="Evaluate localization metrics on baseline_strongVLM outputs."
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help="Root dir containing <model_name>_test folders.",
    )
    parser.add_argument(
        "--model-dirs",
        nargs="+",
        default=[],
        help="Optional explicit model dirs. If omitted, auto-discovers *_test under --root.",
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


def _discover_model_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.endswith("_test")])


def main():
    args = parse_args()
    root = Path(args.root).resolve()

    if args.model_dirs:
        model_dirs = [Path(p).resolve() for p in args.model_dirs]
    else:
        if not root.exists():
            raise FileNotFoundError(f"--root does not exist: {root}")
        model_dirs = _discover_model_dirs(root)

    all_results = {}
    for model_dir in model_dirs:
        model_name = model_dir.name.rstrip("/\\")
        if not model_dir.exists():
            all_results[model_name] = {"error": f"missing_dir: {model_dir.as_posix()}"}
            continue

        records = _load_records(model_dir)
        preds, gts, stats = _build_preds_gts(records)
        metrics = _compute_metrics(preds, gts, k=args.k)
        all_results[model_name] = {
            "model_dir": model_dir.as_posix(),
            "normalization": "extract_numbers_in_order_from_response_text",
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
