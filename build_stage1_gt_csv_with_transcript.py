#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_OUTPUT_CSV = (
    "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_gt_with_transcript.csv"
)


def _region_sort_key(x: str):
    if x.isdigit():
        return (0, int(x))
    return (1, x)


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


def _load_sample_region_diff(order_csv: Path) -> dict[str, dict[str, float]]:
    sample_to_region_diff = defaultdict(dict)
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
            old = sample_to_region_diff[sample_id].get(region_id, float("-inf"))
            # If duplicate rows exist for same sample/region, keep highest diff.
            if diff > old:
                sample_to_region_diff[sample_id][region_id] = diff
    return sample_to_region_diff


def _validate_diff_coverage(
    original_map: dict[str, set[str]],
    diff_map: dict[str, dict[str, float]],
) -> tuple[bool, dict[str, list[str]], dict[str, list[str]]]:
    missing_samples = {}
    missing_regions = {}

    for sample_id in sorted(original_map.keys()):
        if sample_id not in diff_map:
            missing_samples[sample_id] = sorted(original_map[sample_id], key=_region_sort_key)
            continue
        missing = [rid for rid in sorted(original_map[sample_id], key=_region_sort_key) if rid not in diff_map[sample_id]]
        if missing:
            missing_regions[sample_id] = missing

    ok = not missing_samples and not missing_regions
    return ok, missing_samples, missing_regions


def _extract_transcript_word_tier(mfa_json_path: Path) -> str:
    # Mirrors the turn-json to one-line transcript logic in qwen_region_artifact_prompt.py.
    if not mfa_json_path.exists():
        return ""
    try:
        obj = json.loads(mfa_json_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    entries = obj.get("tiers", {}).get("words", {}).get("entries", [])
    parts = []
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        try:
            start = float(entry[0])
            end = float(entry[1])
            word = str(entry[2]).strip()
        except Exception:
            continue
        if not word:
            continue
        parts.append(f"[{start:.2f}-{end:.2f}] {word}")
    return " ".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build stage-1 GT CSV with img_path, combined region IDs, transcript, and explanations placeholder."
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
        "--transcript-json-root",
        required=True,
        help="Directory containing MFA-style transcript JSON files named <sample_id>.json",
    )
    parser.add_argument(
        "--image-suffix",
        default="_grid_img_edge_number_axes.png",
        help="Suffix appended to sample_id to form image filename",
    )
    parser.add_argument(
        "--transcript-json-suffix",
        default=".json",
        help="Suffix appended to sample_id to form transcript JSON filename",
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
            "Optional original CSV for validation before writing output. "
            "If provided with --order-csv, every original (sample_id, region_id) must exist in order CSV."
        ),
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit with error if region-set validation fails.",
    )
    parser.add_argument(
        "--fail-on-missing-transcript",
        action="store_true",
        help="Exit with error if transcript JSON is missing or cannot be parsed.",
    )
    parser.add_argument(
        "--explanations-placeholder",
        default="",
        help="Placeholder text written into the explanations column.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    region_csv = Path(args.region_csv)
    image_dir = Path(args.image_dir)
    transcript_json_root = Path(args.transcript_json_root)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    sample_to_regions = _load_sample_to_regions(region_csv)

    sample_to_region_diff = None
    if args.order_csv:
        order_csv = Path(args.order_csv)
        sample_to_region_diff = _load_sample_region_diff(order_csv)

    if args.check_against_csv and sample_to_region_diff is not None:
        check_csv = Path(args.check_against_csv)
        original_map = _load_sample_to_regions(check_csv)
        ok, missing_samples, missing_regions = _validate_diff_coverage(
            original_map=original_map,
            diff_map=sample_to_region_diff,
        )
        print(
            "[check] diff-coverage consistency:",
            "PASS" if ok else "FAIL",
            f"(missing_samples={len(missing_samples)}, missing_regions={len(missing_regions)})",
        )

        max_show = 10
        if missing_samples:
            print("[check] examples missing_samples:")
            for sample_id in list(missing_samples.keys())[:max_show]:
                print(f"  {sample_id}: expected_regions={missing_samples[sample_id]}")
        if missing_regions:
            print("[check] examples missing_regions:")
            for sample_id in list(missing_regions.keys())[:max_show]:
                print(f"  {sample_id}: missing_regions={missing_regions[sample_id]}")

        if (not ok) and args.fail_on_mismatch:
            raise SystemExit("Diff-coverage validation failed.")

    rows = []
    dropped = 0
    missing_transcript_count = 0

    for sample_id, region_set in sample_to_regions.items():
        if sample_to_region_diff is not None:
            region_diff = sample_to_region_diff.get(sample_id, {})
            region_list = sorted(
                region_set,
                key=lambda rid: (-region_diff.get(rid, float("-inf")), _region_sort_key(rid)),
            )
        else:
            region_list = sorted(region_set, key=_region_sort_key)

        if args.strict_three and len(region_list) != 3:
            dropped += 1
            continue

        img_path = image_dir / f"{sample_id}{args.image_suffix}"
        transcript_json_path = transcript_json_root / f"{sample_id}{args.transcript_json_suffix}"
        transcript_text = _extract_transcript_word_tier(transcript_json_path)

        if not transcript_text:
            missing_transcript_count += 1
            if args.fail_on_missing_transcript:
                raise SystemExit(
                    f"Missing/invalid transcript for sample_id='{sample_id}' at: {transcript_json_path}"
                )

        rows.append(
            {
                "img_path": str(img_path),
                "regions": ",".join(region_list),
                "transcript": transcript_text,
                "explanations": args.explanations_placeholder,
            }
        )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["img_path", "regions", "transcript", "explanations"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to: {output_csv}")
    if args.strict_three:
        print(f"Dropped {dropped} samples (not exactly 3 unique region IDs)")
    print(f"Rows with missing/invalid transcript: {missing_transcript_count}")


if __name__ == "__main__":
    main()