#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert stage-1 GT CSV (img_path,regions) to LLaVA-format SFT JSON."
    )
    parser.add_argument("--input-csv", required=True, help="CSV with columns: img_path, regions")
    parser.add_argument("--output-json", default=None, help="Output JSON path (single file mode)")
    parser.add_argument("--image-folder", default=None, help="If set, store image path relative to this folder")
    parser.add_argument(
        "--user-prompt",
        default="Select the top 3 regions that most likely contain spoof artifacts.",
        help="User prompt text to include in each sample",
    )
    parser.add_argument(
        "--json-array-target",
        action="store_true",
        help="Write assistant target as JSON array string, e.g. [1, 2, 3].",
    )
    parser.add_argument("--train-json", default=None, help="Train split JSON path")
    parser.add_argument("--val-json", default=None, help="Val split JSON path")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio in split mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split mode")
    parser.add_argument(
        "--split-by-path",
        action="store_true",
        help="Split train/val by img_path substring keys instead of random split.",
    )
    parser.add_argument(
        "--train-key",
        default="_LA_T_",
        help="Substring in img_path that routes a sample to train when --split-by-path is used.",
    )
    parser.add_argument(
        "--val-key",
        default="_LA_D_",
        help="Substring in img_path that routes a sample to val when --split-by-path is used.",
    )
    parser.add_argument(
        "--strict-path-split",
        action="store_true",
        help="If set with --split-by-path, drop samples that match neither key.",
    )
    return parser.parse_args()


def to_target(regions_text: str, json_array_target: bool) -> str:
    parts = [p.strip() for p in regions_text.split(",") if p.strip()]
    if json_array_target:
        vals = []
        for p in parts:
            vals.append(int(p) if p.isdigit() else p)
        return json.dumps(vals)
    return ",".join(parts)


def build_record(idx: int, img_path: str, target: str, user_prompt: str) -> dict:
    return {
        "id": f"stage1_{idx:06d}",
        "image": img_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{user_prompt}"},
            {"from": "gpt", "value": target},
        ],
    }


def maybe_relpath(img_path: Path, image_folder: Path | None) -> str:
    if image_folder is None:
        return str(img_path)
    try:
        return str(img_path.relative_to(image_folder))
    except ValueError:
        return str(img_path)


def write_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    image_folder = Path(args.image_folder) if args.image_folder else None

    records = []
    path_aware_rows = []
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            img_path = Path(str(row["img_path"]).strip())
            regions = str(row["regions"]).strip()
            target = to_target(regions, json_array_target=args.json_array_target)
            image_field = maybe_relpath(img_path, image_folder)
            rec = build_record(idx=idx, img_path=image_field, target=target, user_prompt=args.user_prompt)
            records.append(rec)
            path_aware_rows.append((str(img_path), rec))

    split_mode = args.train_json is not None and args.val_json is not None
    if split_mode:
        if args.split_by_path:
            train_data = []
            val_data = []
            unmatched = []
            for raw_img_path, rec in path_aware_rows:
                if args.train_key in raw_img_path:
                    train_data.append(rec)
                elif args.val_key in raw_img_path:
                    val_data.append(rec)
                else:
                    unmatched.append(rec)

            if unmatched and not args.strict_path_split:
                train_data.extend(unmatched)

            print(f"Path split keys: train='{args.train_key}', val='{args.val_key}'")
            print(f"Unmatched samples: {len(unmatched)}")
            if args.strict_path_split and unmatched:
                print("Dropped unmatched samples due to --strict-path-split")
        else:
            random.seed(args.seed)
            random.shuffle(records)
            n_val = int(len(records) * args.val_ratio)
            val_data = records[:n_val]
            train_data = records[n_val:]
        train_path = Path(args.train_json)
        val_path = Path(args.val_json)
        write_json(train_path, train_data)
        write_json(val_path, val_data)
        print(f"Saved train: {len(train_data)} -> {train_path}")
        print(f"Saved val:   {len(val_data)} -> {val_path}")
        return

    if args.output_json is None:
        raise ValueError("Provide --output-json (single file) OR both --train-json and --val-json (split mode).")

    out_path = Path(args.output_json)
    write_json(out_path, records)
    print(f"Saved {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
