#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

DEFAULT_INPUT_CSV = (
    "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_gt.csv"
)
DEFAULT_OUTPUT_CSV = (
    "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_gt_with_transcript.csv"
)


def _load_input_rows(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = str(row.get("img_path", "")).strip()
            regions = str(row.get("regions", "")).strip()
            explanations = str(row.get("explanations", ""))
            if not img_path or not regions:
                continue
            rows.append(
                {
                    "img_path": img_path,
                    "regions": regions,
                    "explanations": explanations,
                }
            )
    return rows


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


def _sample_id_from_img_path(img_path: str, image_suffix: str) -> str:
    filename = Path(img_path).name
    if image_suffix and filename.endswith(image_suffix):
        return filename[: -len(image_suffix)]
    return Path(filename).stem


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build stage-1 GT CSV with transcript from an existing img_path/regions CSV."
    )
    parser.add_argument(
        "--input-csv",
        "--region-csv",
        dest="input_csv",
        default=DEFAULT_INPUT_CSV,
        help=f"Input CSV with at least: img_path, regions (default: {DEFAULT_INPUT_CSV})",
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
        help="If set, only keep rows with exactly 3 unique region IDs in the regions column.",
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

    input_csv = Path(args.input_csv)
    transcript_json_root = Path(args.transcript_json_root)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    input_rows = _load_input_rows(input_csv)

    rows = []
    dropped = 0
    missing_transcript_count = 0
    skipped_no_sample_id = 0

    for row in input_rows:
        img_path = row["img_path"]
        regions_raw = row["regions"]
        region_list = [rid.strip() for rid in regions_raw.split(",") if rid.strip()]

        if args.strict_three and len(set(region_list)) != 3:
            dropped += 1
            continue

        sample_id = _sample_id_from_img_path(img_path, args.image_suffix)
        if not sample_id:
            skipped_no_sample_id += 1
            continue

        transcript_json_path = transcript_json_root / f"{sample_id}{args.transcript_json_suffix}"
        transcript_text = _extract_transcript_word_tier(transcript_json_path)

        if not transcript_text:
            missing_transcript_count += 1
            if args.fail_on_missing_transcript:
                raise SystemExit(
                    f"Missing/invalid transcript for sample_id='{sample_id}' derived from img_path='{img_path}' at: {transcript_json_path}"
                )

        rows.append(
            {
                "img_path": img_path,
                "regions": regions_raw,
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
        print(f"Dropped {dropped} rows (not exactly 3 unique region IDs)")
    if skipped_no_sample_id:
        print(f"Skipped {skipped_no_sample_id} rows (could not derive sample_id from img_path)")
    print(f"Rows with missing/invalid transcript: {missing_transcript_count}")


if __name__ == "__main__":
    main()
