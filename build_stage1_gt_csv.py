#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

DEFAULT_OUTPUT_CSV = (
    "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_gt.csv"
)


def _region_sort_key(x: str):
    return int(x) if x.isdigit() else x


def _load_sample_to_regions(csv_path: Path) -> dict[str, set[str]]:
    sample_to_regions = defaultdict(set)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            region_id = str(row.get("region_id", "")).strip()
            if not sample_id or not region_id:
                continue
            sample_to_regions[sample_id].add(region_id)
    return sample_to_regions


def _load_sample_to_ordered_regions(order_csv: Path) -> dict[str, list[str]]:
    sample_to_rows = defaultdict(list)
    with order_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            region_id = str(row.get("region_id", "")).strip()
            diff_raw = str(row.get("diff", "")).strip()
            if not sample_id or not region_id:
                continue
            try:
                diff = float(diff_raw)
            except ValueError:
                # Missing/invalid diff gets lowest priority.
                diff = float("-inf")
            rid_for_tie = int(region_id) if region_id.isdigit() else region_id
            sample_to_rows[sample_id].append((region_id, diff, rid_for_tie))

    sample_to_ordered_regions = {}
    for sample_id, rows in sample_to_rows.items():
        # Order by diff descending, then smaller region_id first on ties.
        rows_sorted = sorted(rows, key=lambda x: (-x[1], x[2]))
        ordered_unique = []
        seen = set()
        for region_id, _, _ in rows_sorted:
            if region_id in seen:
                continue
            seen.add(region_id)
            ordered_unique.append(region_id)
        sample_to_ordered_regions[sample_id] = ordered_unique
    return sample_to_ordered_regions


def _validate_region_sets(
    original_map: dict[str, set[str]],
    order_map: dict[str, list[str]],
) -> tuple[bool, dict[str, list[str]], dict[str, list[str]], dict[str, tuple[list[str], list[str]]]]:
    missing_in_order = {}
    extra_in_order = {}
    mismatched_sets = {}

    original_keys = set(original_map.keys())
    order_keys = set(order_map.keys())

    for sample_id in sorted(original_keys - order_keys):
        missing_in_order[sample_id] = sorted(original_map[sample_id], key=_region_sort_key)
    for sample_id in sorted(order_keys - original_keys):
        extra_in_order[sample_id] = sorted(set(order_map[sample_id]), key=_region_sort_key)

    for sample_id in sorted(original_keys & order_keys):
        original_set = set(original_map[sample_id])
        order_set = set(order_map[sample_id])
        if original_set != order_set:
            mismatched_sets[sample_id] = (
                sorted(original_set, key=_region_sort_key),
                sorted(order_set, key=_region_sort_key),
            )

    ok = not missing_in_order and not extra_in_order and not mismatched_sets
    return ok, missing_in_order, extra_in_order, mismatched_sets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build stage-1 GT CSV with img_path and combined region IDs."
    )
    parser.add_argument(
        "--region-csv",
        required=True,
        help="Input CSV with at least: sample_id, region_id",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing spectrogram images",
    )
    parser.add_argument(
        "--image-suffix",
        default="_grid_img_edge_number_axes.png",
        help="Suffix appended to sample_id to form image filename",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--strict-three",
        action="store_true",
        help="If set, only keep samples with exactly 3 unique region IDs",
    )
    parser.add_argument(
        "--order-csv",
        default=None,
        help=(
            "Optional CSV used to order region IDs per sample. "
            "Must include sample_id, region_id, diff. "
            "Sorting is by diff desc, tie by smaller region_id."
        ),
    )
    parser.add_argument(
        "--check-against-csv",
        default=None,
        help=(
            "Optional original CSV for region-set validation before writing output. "
            "If provided together with --order-csv, each sample must have identical region sets."
        ),
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit with error if region-set validation fails.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    region_csv = Path(args.region_csv)
    image_dir = Path(args.image_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    sample_to_regions = _load_sample_to_regions(region_csv)

    sample_to_ordered_regions = None
    if args.order_csv:
        order_csv = Path(args.order_csv)
        sample_to_ordered_regions = _load_sample_to_ordered_regions(order_csv)

    if args.check_against_csv and sample_to_ordered_regions is not None:
        check_csv = Path(args.check_against_csv)
        original_map = _load_sample_to_regions(check_csv)
        ok, missing_in_order, extra_in_order, mismatched_sets = _validate_region_sets(
            original_map=original_map,
            order_map=sample_to_ordered_regions,
        )
        print(
            "[check] region-set consistency:",
            "PASS" if ok else "FAIL",
            f"(missing={len(missing_in_order)}, extra={len(extra_in_order)}, mismatch={len(mismatched_sets)})",
        )

        # Print a few examples for fast debugging on cluster logs.
        max_show = 10
        if missing_in_order:
            print("[check] examples missing_in_order:")
            for sample_id in list(missing_in_order.keys())[:max_show]:
                print(f"  {sample_id}: expected={missing_in_order[sample_id]}")
        if extra_in_order:
            print("[check] examples extra_in_order:")
            for sample_id in list(extra_in_order.keys())[:max_show]:
                print(f"  {sample_id}: got={extra_in_order[sample_id]}")
        if mismatched_sets:
            print("[check] examples mismatched_sets:")
            for sample_id in list(mismatched_sets.keys())[:max_show]:
                exp, got = mismatched_sets[sample_id]
                print(f"  {sample_id}: expected={exp} got={got}")

        if (not ok) and args.fail_on_mismatch:
            raise SystemExit("Region-set validation failed.")

    rows = []
    dropped = 0
    for sample_id, region_set in sample_to_regions.items():
        if sample_to_ordered_regions is not None and sample_id in sample_to_ordered_regions:
            region_list = [rid for rid in sample_to_ordered_regions[sample_id] if rid in region_set]
            # Safety fallback if order CSV misses some IDs despite validation disabled.
            missing_ids = [rid for rid in sorted(region_set, key=_region_sort_key) if rid not in set(region_list)]
            region_list.extend(missing_ids)
        else:
            region_list = sorted(region_set, key=_region_sort_key)
        if args.strict_three and len(region_list) != 3:
            dropped += 1
            continue

        img_path = image_dir / f"{sample_id}{args.image_suffix}"
        regions = ",".join(region_list)
        rows.append({"img_path": str(img_path), "regions": regions})

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["img_path", "regions"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to: {output_csv}")
    if args.strict_three:
        print(f"Dropped {dropped} samples (not exactly 3 unique region IDs)")


if __name__ == "__main__":
    main()
