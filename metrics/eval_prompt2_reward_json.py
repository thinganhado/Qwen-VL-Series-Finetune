#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
METRICS_ROOT = REPO_ROOT / "metrics"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(METRICS_ROOT) not in sys.path:
    sys.path.insert(0, str(METRICS_ROOT))

from train.reward_funcs_prompt2 import (  # noqa: E402
    FREQ_LABELS,
    PHON_LABELS,
    TIME_LABELS,
    _independent_extract_from_en,
    _normalize_freq,
    _normalize_phon,
    _normalize_time,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate prompt2 outputs from per-sample json files."
    )
    parser.add_argument("--input-root", required=True, help="Root that contains grid/*/json files.")
    parser.add_argument("--glob-pattern", default="grid/*/json", help="Glob pattern under input-root.")
    parser.add_argument("--output-json", default=None, help="Optional summary+detail json.")
    parser.add_argument("--output-jsonl", default=None, help="Optional per-sample jsonl.")
    parser.add_argument(
        "--strict-caption-metrics",
        action="store_true",
        default=True,
        help="Fail fast if ROUGE-L/METEOR/BERTScore dependencies are missing or fail at runtime.",
    )
    parser.add_argument(
        "--no-strict-caption-metrics",
        action="store_false",
        dest="strict_caption_metrics",
        help="Do not fail on caption metric dependency/runtime issues; return null for failed metrics.",
    )
    return parser.parse_args()


_PRED_TUPLE_RE = re.compile(
    r"\(\s*[Cc]?\s*(\d+)\s*,\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*(.*?)\s*\)",
    flags=re.DOTALL,
)


def _normalize_pred_response(text: str) -> str:
    return re.sub(r"\(\s*[Cc]\s*(\d+)\s*,", r"(\1,", str(text or ""))


def _normalize_gt_prompt2_target(prompt2_target: str) -> Optional[str]:
    text = str(prompt2_target or "").strip()
    if not text:
        return None

    chunks = [c.strip() for c in text.split(";") if c.strip()]
    rows = []
    for ch in chunks:
        m = re.match(
            r"^C\d+\s*=\s*(\d+)\s*,\s*T\s*=\s*([^,]+)\s*,\s*F\s*=\s*([^,]+)\s*,\s*P\s*=\s*([^,]+)\s*,\s*En\s*=\s*(.*)$",
            ch,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return None
        cn, t, f, p, en = m.groups()
        rows.append(f"({cn.strip()}, {t.strip()}, {f.strip()}, {p.strip()}, {en.strip()})")

    if len(rows) != 3:
        return None
    return "\n".join(rows)


def _parse_regions_loose(text: str) -> Dict[str, Dict[str, Optional[str]]]:
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for cn_raw, t_raw, f_raw, p_raw, en_raw in _PRED_TUPLE_RE.findall(str(text or "")):
        cn = str(int(cn_raw))
        if cn in out:
            continue
        out[cn] = {
            "Cn": cn,
            "T": _normalize_time(t_raw),
            "F": _normalize_freq(f_raw),
            "P": _normalize_phon(p_raw),
            "En": str(en_raw or "").strip(),
        }
    return out


def _field_metrics(y_true: List[str], y_pred: List[Optional[str]], labels: List[str]) -> Dict[str, float]:
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if p == t)
    acc = (correct / n) if n else 0.0

    f1_per_class = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1_per_class.append(f1)
    macro_f1 = mean(f1_per_class) if f1_per_class else 0.0
    return {"accuracy": acc, "macro_f1": macro_f1}


def _caption_scores(
    gt_caps: List[str], pred_caps: List[str], strict: bool = True
) -> Dict[str, Optional[float]]:
    scores: Dict[str, Optional[float]] = {"ROUGE_L": None, "METEOR": None, "BERTScore_F1": None}
    if not gt_caps:
        return scores

    try:
        from caption_metrics import meteor_score, rouge_l_score

        gts = {i: [gt] for i, gt in enumerate(gt_caps)}
        res = {i: [pd] for i, pd in enumerate(pred_caps)}
        rouge_l, _ = rouge_l_score(gts, res)
        meteor, _ = meteor_score(gts, res)
        scores["ROUGE_L"] = float(rouge_l)
        scores["METEOR"] = float(meteor)
    except Exception as e:
        if strict:
            raise RuntimeError(f"ROUGE_L/METEOR computation failed: {e}") from e

    try:
        from bert_score import score as bert_score

        _, _, f1 = bert_score(pred_caps, gt_caps, lang="en", verbose=False)
        scores["BERTScore_F1"] = float(f1.mean().item())
    except Exception as e:
        if strict:
            raise RuntimeError(f"BERTScore computation failed: {e}") from e

    return scores


def main():
    args = parse_args()
    root = Path(args.input_root).expanduser().resolve()
    paths = sorted(root.glob(args.glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No files found: root={root}, glob={args.glob_pattern}")

    # Dataset-level region-aligned lists for classification
    y_true_t: List[str] = []
    y_pred_t: List[Optional[str]] = []
    y_true_f: List[str] = []
    y_pred_f: List[Optional[str]] = []
    y_true_p: List[str] = []
    y_pred_p: List[Optional[str]] = []

    # Consistency stats over all GT regions (3 per valid sample)
    n_regions = 0
    extractable = {"T": 0, "F": 0, "P": 0}
    agree = {"T": 0, "F": 0, "P": 0}

    # Caption pairs (region-aligned by Cn)
    gt_caps: List[str] = []
    pred_caps: List[str] = []

    per_sample = []
    unreadable = 0
    invalid_gt = 0

    for p in paths:
        try:
            rec = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            unreadable += 1
            continue

        sample_id = str(rec.get("sample_id_raw") or rec.get("sample_id") or p.parent.name)
        gt_norm = _normalize_gt_prompt2_target(str(rec.get("prompt2_target", "")))
        pred_norm = _normalize_pred_response(str(rec.get("response", "")))

        if gt_norm is None:
            invalid_gt += 1
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "status": "invalid_gt",
                    "num_regions": 0,
                }
            )
            continue

        gt_by_cn = _parse_regions_loose(gt_norm)
        pred_by_cn = _parse_regions_loose(pred_norm)

        sample_region_scores = []
        for cn, gt in gt_by_cn.items():
            pred = pred_by_cn.get(cn)
            n_regions += 1

            # Accuracy inputs (GT fields vs predicted fields)
            y_true_t.append(gt["T"])  # type: ignore[arg-type]
            y_true_f.append(gt["F"])  # type: ignore[arg-type]
            y_true_p.append(gt["P"])  # type: ignore[arg-type]
            y_pred_t.append(pred["T"] if pred else None)
            y_pred_f.append(pred["F"] if pred else None)
            y_pred_p.append(pred["P"] if pred else None)

            # Consistency inputs (pred En extractor vs pred field)
            if pred is not None:
                e = _independent_extract_from_en(pred.get("En", ""))
                for k in ("T", "F", "P"):
                    if e[k] is not None:
                        extractable[k] += 1
                        if e[k] == pred.get(k):
                            agree[k] += 1
            # region cons score with missing pred treated as 0
            region_cs = 0.0
            if pred is not None:
                e = _independent_extract_from_en(pred.get("En", ""))
                region_cs = (
                    (1.0 if e["T"] is not None and e["T"] == pred.get("T") else 0.0)
                    + (1.0 if e["F"] is not None and e["F"] == pred.get("F") else 0.0)
                    + (1.0 if e["P"] is not None and e["P"] == pred.get("P") else 0.0)
                ) / 3.0
            sample_region_scores.append(region_cs)

            # Caption metrics inputs (GT En vs pred En), region aligned by Cn
            gt_caps.append(gt.get("En", "") or "")
            pred_caps.append((pred.get("En", "") if pred else "") or "")

        per_sample.append(
            {
                "sample_id": sample_id,
                "status": "ok",
                "num_regions": len(gt_by_cn),
                "ConsScore_sample": (mean(sample_region_scores) if sample_region_scores else 0.0),
            }
        )

    # Accuracy metrics
    t_metrics = _field_metrics(y_true_t, y_pred_t, sorted(TIME_LABELS))
    f_metrics = _field_metrics(y_true_f, y_pred_f, sorted(FREQ_LABELS))
    p_metrics = _field_metrics(y_true_p, y_pred_p, sorted(PHON_LABELS))
    mean_fieldacc_macro_3 = mean([t_metrics["accuracy"], f_metrics["accuracy"], p_metrics["accuracy"]]) if n_regions else 0.0

    # Consistency metrics
    coverage_t = (extractable["T"] / n_regions) if n_regions else 0.0
    coverage_f = (extractable["F"] / n_regions) if n_regions else 0.0
    coverage_p = (extractable["P"] / n_regions) if n_regions else 0.0
    agreement_t = (agree["T"] / extractable["T"]) if extractable["T"] else 0.0
    agreement_f = (agree["F"] / extractable["F"]) if extractable["F"] else 0.0
    agreement_p = (agree["P"] / extractable["P"]) if extractable["P"] else 0.0
    coverage_avg = mean([coverage_t, coverage_f, coverage_p]) if n_regions else 0.0
    agreement_given_extractable_avg = mean([agreement_t, agreement_f, agreement_p]) if n_regions else 0.0
    consscore = (agree["T"] + agree["F"] + agree["P"]) / (3.0 * n_regions) if n_regions else 0.0

    # Caption quality on En
    caption = _caption_scores(gt_caps, pred_caps, strict=args.strict_caption_metrics)

    summary = {
        "num_files": len(paths),
        "num_unreadable": unreadable,
        "num_invalid_gt": invalid_gt,
        "num_regions_scored": n_regions,
        "accuracy": {
            "Accuracy_T": t_metrics["accuracy"],
            "Accuracy_F": f_metrics["accuracy"],
            "Accuracy_P": p_metrics["accuracy"],
            "MacroF1_T": t_metrics["macro_f1"],
            "MacroF1_F": f_metrics["macro_f1"],
            "MacroF1_P": p_metrics["macro_f1"],
            "MeanFieldAcc_macro_3fields": mean_fieldacc_macro_3,
        },
        "consistency": {
            "ConsScore": consscore,
            "Coverage_T": coverage_t,
            "Coverage_F": coverage_f,
            "Coverage_P": coverage_p,
            "CoverageAvg": coverage_avg,
            "AgreementGivenExtractable_T": agreement_t,
            "AgreementGivenExtractable_F": agreement_f,
            "AgreementGivenExtractable_P": agreement_p,
            "AgreementGivenExtractableAvg": agreement_given_extractable_avg,
        },
        "caption_quality_en": {
            "ROUGE_L": caption["ROUGE_L"],
            "METEOR": caption["METEOR"],
            "BERTScore_F1": caption["BERTScore_F1"],
        },
    }

    print(json.dumps(summary, indent=2))

    if args.output_json:
        out = Path(args.output_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "samples": per_sample}, indent=2), encoding="utf-8")

    if args.output_jsonl:
        outl = Path(args.output_jsonl).expanduser().resolve()
        outl.parent.mkdir(parents=True, exist_ok=True)
        with outl.open("w", encoding="utf-8") as f:
            for row in per_sample:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
