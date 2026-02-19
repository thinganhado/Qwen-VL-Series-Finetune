#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_JSON_DIR = (
    "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/En/polished"
)
DEFAULT_GT_CSV = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/region_phone_table_grid.csv"
DEFAULT_OUT_DIR = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/En/explanation"


def _norm_text(v: str) -> str:
    return str(v).strip().lower().replace("-", "_").replace(" ", "_")


def _norm_region_id(v) -> str:
    s = str(v).strip()
    try:
        return str(int(s))
    except Exception:
        return s


def _load_gt(gt_csv: Path):
    gt = {}
    duplicates = []
    with gt_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            region_id = _norm_region_id(row.get("region_id", ""))
            if not sample_id or not region_id:
                continue
            key = (sample_id, region_id)
            val = {
                "time": _norm_text(row.get("T", "")),
                "frequency": _norm_text(row.get("F", "")),
                "phonetic": _norm_text(row.get("P_type", "")),
            }
            if key in gt:
                duplicates.append(key)
                continue
            gt[key] = val
    return gt, duplicates


def _iter_json_files(json_dir: Path):
    if json_dir.is_file() and (json_dir.suffix.lower() == ".json" or json_dir.name == "json"):
        yield json_dir
        return
    for p in json_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".json" or p.name == "json":
            yield p


def _extract_regions(obj: dict, source_file: Path):
    root_sample_id = str(obj.get("sample_id", "")).strip()
    regions = obj.get("regions", [])
    if not isinstance(regions, list):
        return []

    out = []
    for item in regions:
        if not isinstance(item, dict):
            continue
        sample_id = str(item.get("sample_id", "")).strip() or root_sample_id
        region_id = _norm_region_id(item.get("region_id", ""))
        output_explanation = str(item.get("output_explanation", "")).strip()
        output_structured = item.get("output_structured", {})
        out.append(
            {
                "sample_id": sample_id,
                "region_id": region_id,
                "output_explanation": output_explanation,
                "output_structured": output_structured,
                "source_file": str(source_file),
            }
        )
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check region output JSON against region_phone_table_grid.csv and save passing explanations to "
            "<out>/<sample_id>/<region_id>.txt"
        )
    )
    parser.add_argument("--json-dir", default=DEFAULT_JSON_DIR, help=f"JSON file or directory (default: {DEFAULT_JSON_DIR})")
    parser.add_argument("--gt-csv", default=DEFAULT_GT_CSV, help=f"GT CSV path (default: {DEFAULT_GT_CSV})")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help=f"Output explanation root (default: {DEFAULT_OUT_DIR})")
    parser.add_argument(
        "--require-explanation",
        action="store_true",
        help="Require non-empty output_explanation for a passing item.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    json_dir = Path(args.json_dir).expanduser()
    gt_csv = Path(args.gt_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] json_dir: {json_dir}")
    print(f"[info] json_dir exists={json_dir.exists()} is_file={json_dir.is_file()} is_dir={json_dir.is_dir()}")
    if not json_dir.exists():
        raise SystemExit(f"JSON path does not exist: {json_dir}")

    gt_map, gt_duplicates = _load_gt(gt_csv)
    if gt_duplicates:
        print(f"[warn] duplicate GT keys found, ignored later rows: {len(gt_duplicates)}")

    all_items = []
    json_count = 0
    bad_json_files = []
    for jf in _iter_json_files(json_dir):
        json_count += 1
        try:
            obj = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            bad_json_files.append((str(jf), str(e)))
            continue
        all_items.extend(_extract_regions(obj, jf))

    missing_structured = []
    missing_gt = []
    missing_explanation = []
    failing = []
    passing = []

    seen_by_sample = defaultdict(set)
    samples_seen = set()

    for item in all_items:
        sample_id = item["sample_id"]
        region_id = item["region_id"]
        if sample_id:
            samples_seen.add(sample_id)
        if sample_id and region_id:
            seen_by_sample[sample_id].add(region_id)

        os_val = item["output_structured"]
        if not sample_id or not region_id or not isinstance(os_val, dict):
            missing_structured.append(item)
            continue

        gt_key = (sample_id, region_id)
        gt_val = gt_map.get(gt_key)
        if gt_val is None:
            missing_gt.append(item)
            continue

        pred = {
            "time": _norm_text(os_val.get("time", "")),
            "frequency": _norm_text(os_val.get("frequency", "")),
            "phonetic": _norm_text(os_val.get("phonetic", "")),
        }
        ok = pred == gt_val
        if not ok:
            failing.append((item, pred, gt_val))
            continue

        if args.require_explanation and not item["output_explanation"]:
            missing_explanation.append(item)
            continue
        passing.append(item)

    missing_regions_from_json = []
    for (sample_id, region_id), _ in gt_map.items():
        if sample_id not in samples_seen:
            continue
        if region_id not in seen_by_sample[sample_id]:
            missing_regions_from_json.append((sample_id, region_id))

    for item in passing:
        sample_dir = out_dir / item["sample_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        out_file = sample_dir / f"{item['region_id']}.txt"
        out_file.write_text(item["output_explanation"], encoding="utf-8")

    print(f"JSON files scanned: {json_count}")
    print(f"Region items extracted: {len(all_items)}")
    print(f"Passing items: {len(passing)}")
    print(f"Saved explanations to: {out_dir}")
    if json_count == 0 and json_dir.is_dir():
        print("[hint] No .json files found under --json-dir (recursive).")
        print("[hint] Verify the folder and file extension, e.g. run:")
        print(f"[hint] find {json_dir} -type f | head -n 20")

    print("")
    print("[missing] summary")
    print(f"- bad_json_files: {len(bad_json_files)}")
    print(f"- missing_structured_or_ids: {len(missing_structured)}")
    print(f"- missing_gt_lookup: {len(missing_gt)}")
    print(f"- missing_explanation_for_passing: {len(missing_explanation)}")
    print(f"- expected_regions_missing_in_json (for seen samples): {len(missing_regions_from_json)}")

    if bad_json_files:
        print("")
        print("[missing] bad_json_files")
        for p, e in bad_json_files:
            print(f"- {p}: {e}")

    if missing_structured:
        print("")
        print("[missing] missing_structured_or_ids (sample_id, region_id, source)")
        for item in missing_structured:
            print(f"- {item['sample_id']}, {item['region_id']}, {item['source_file']}")

    if missing_gt:
        print("")
        print("[missing] missing_gt_lookup (sample_id, region_id, source)")
        for item in missing_gt:
            print(f"- {item['sample_id']}, {item['region_id']}, {item['source_file']}")

    if missing_explanation:
        print("")
        print("[missing] missing_explanation_for_passing (sample_id, region_id, source)")
        for item in missing_explanation:
            print(f"- {item['sample_id']}, {item['region_id']}, {item['source_file']}")

    if missing_regions_from_json:
        print("")
        print("[missing] expected_regions_missing_in_json (sample_id, region_id)")
        for sample_id, region_id in missing_regions_from_json:
            print(f"- {sample_id}, {region_id}")

    if failing:
        print("")
        print("[failing] sample_id,region_id")
        for item, _, _ in failing:
            print(f"- {item['sample_id']}, {item['region_id']}")

        print("")
        print("[failing] mismatch vs GT")
        for item, pred, gt_val in failing:
            print(f"- sample_id={item['sample_id']} region_id={item['region_id']} source={item['source_file']}")
            print(f"  pred: time={pred['time']} frequency={pred['frequency']} phonetic={pred['phonetic']}")
            print(
                f"  gt:   time={gt_val['time']} frequency={gt_val['frequency']} phonetic={gt_val['phonetic']}"
            )


if __name__ == "__main__":
    main()
