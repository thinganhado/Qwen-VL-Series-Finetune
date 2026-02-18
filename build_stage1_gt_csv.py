#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


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
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--strict-three",
        action="store_true",
        help="If set, only keep samples with exactly 3 unique region IDs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    region_csv = Path(args.region_csv)
    image_dir = Path(args.image_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    sample_to_regions = defaultdict(set)
    with region_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row["sample_id"]).strip()
            region_id = str(row["region_id"]).strip()
            if not sample_id or not region_id:
                continue
            sample_to_regions[sample_id].add(region_id)

    rows = []
    dropped = 0
    for sample_id, region_set in sample_to_regions.items():
        region_list = sorted(region_set, key=lambda x: int(x) if x.isdigit() else x)
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
