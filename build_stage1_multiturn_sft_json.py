#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert CSV (img_path,regions,transcript,explanations) to LLaVA-format "
            "multi-turn SFT JSON. Turn-1 assistant target is GT regions by default."
        )
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="CSV with columns: img_path, regions, transcript, explanations(optional)",
    )
    parser.add_argument("--output-json", default=None, help="Output JSON path (single file mode)")
    parser.add_argument("--image-folder", default=None, help="If set, store image path relative to this folder")

    parser.add_argument(
        "--turn1-target-column",
        default="regions",
        help="CSV column used as turn-1 assistant target (GT regions).",
    )
    parser.add_argument(
        "--turn2-regions-column",
        default="",
        help=(
            "Optional CSV column to use for turn-2 region context. "
            "If empty, uses turn-1 target column."
        ),
    )
    parser.add_argument(
        "--require-turn1-target",
        action="store_true",
        help="Fail if turn-1 target column is missing/empty for any row.",
    )

    parser.add_argument(
        "--turn1-system-prompt",
        default=(
            "As an expert in deepfake speech spectrogram forensics, you can detect regions containing deepfake artifacts "
            "by analysing spectrogram segments. Return only the JSON array of your three chosen region IDs, ordered "
            "from most to least prominent spoof artifact evidence."
        ),
        help="System prompt text to embed in turn-1 user message.",
    )
    parser.add_argument(
        "--turn1-user-prompt",
        default="Select the top 3 regions that most likely contain spoof artifacts.",
        help="User prompt for turn-1 message (image-only step).",
    )

    parser.add_argument(
        "--turn2-system-prompt",
        default=(
            "You have selected 3 regions containing artifacts. Please analyze the regions in 3 dimensions: spectral, "
            "temporal, and phonetic using the spectrogram and transcript. For each region in the same order of "
            "selection, provide a description and interpret the audio impact."
        ),
        help="System prompt text to embed in turn-2 user message.",
    )
    parser.add_argument(
        "--turn2-user-prompt",
        default="Produce a forensic explanation bound to each artifact region.",
        help="User prompt for turn-2 message.",
    )
    parser.add_argument(
        "--turn2-user-template",
        default=(
            "{turn2_user_prompt}\n"
            "Transcript: {transcript}\n"
            "Selected regions: {prompt1_output}"
        ),
        help=(
            "Template for turn-2 user message body. Supported placeholders: "
            "{turn2_user_prompt}, {transcript}, {prompt1_output}, {sample_id}."
        ),
    )

    parser.add_argument(
        "--json-array-regions",
        action="store_true",
        help="Normalize regions into JSON array string, e.g. [1, 2, 3].",
    )
    parser.add_argument(
        "--explanations-default",
        default="[TODO_EXPLANATION]",
        help="Fallback explanation text when CSV explanations is empty.",
    )

    parser.add_argument("--train-json", default=None, help="Train split JSON path")
    parser.add_argument("--val-json", default=None, help="Val split JSON path")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio in random split mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random split mode")
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


def normalize_regions(regions_text: str, as_json_array: bool) -> str:
    parts = [p.strip() for p in str(regions_text).split(",") if p.strip()]
    if not as_json_array:
        return ",".join(parts)

    vals = []
    for p in parts:
        vals.append(int(p) if p.isdigit() else p)
    return json.dumps(vals)


def maybe_relpath(img_path: Path, image_folder: Path | None) -> str:
    if image_folder is None:
        return str(img_path)
    try:
        return str(img_path.relative_to(image_folder))
    except ValueError:
        return str(img_path)


def compose_user_turn(system_prompt: str, user_body: str) -> str:
    # Keep role pairing compatible with the repo's SFT parser (user/assistant pairs only).
    return f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_body}"


def build_record(
    idx: int,
    img_path: str,
    turn1_target: str,
    turn2_regions_context: str,
    transcript: str,
    explanation: str,
    turn1_system_prompt: str,
    turn1_user_prompt: str,
    turn2_system_prompt: str,
    turn2_user_prompt: str,
    turn2_user_template: str,
) -> dict:
    sample_id = Path(img_path).stem

    turn1_user = compose_user_turn(
        system_prompt=turn1_system_prompt,
        user_body=turn1_user_prompt,
    )

    turn2_body = turn2_user_template.format_map(
        {
            "turn2_user_prompt": turn2_user_prompt,
            "transcript": transcript,
            "prompt1_output": turn2_regions_context,
            "sample_id": sample_id,
        }
    )
    turn2_user = compose_user_turn(
        system_prompt=turn2_system_prompt,
        user_body=turn2_body,
    )

    return {
        "id": f"stage1_mt_{idx:06d}",
        "image": img_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{turn1_user}"},
            {"from": "gpt", "value": turn1_target},
            {"from": "human", "value": f"<image>\n{turn2_user}"},
            {"from": "gpt", "value": explanation},
        ],
    }


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
            img_path = Path(str(row.get("img_path", "")).strip())
            if not str(img_path):
                continue

            turn1_raw = str(row.get(args.turn1_target_column, "")).strip()
            if not turn1_raw:
                msg = f"Missing turn1 target in column '{args.turn1_target_column}' at row {idx + 2}."
                if args.require_turn1_target:
                    raise ValueError(msg)
                continue

            if args.turn2_regions_column:
                turn2_raw = str(row.get(args.turn2_regions_column, "")).strip() or turn1_raw
            else:
                turn2_raw = turn1_raw

            transcript = str(row.get("transcript", "")).strip()
            explanation = str(row.get("explanations", "")).strip() or args.explanations_default

            turn1_target = normalize_regions(turn1_raw, as_json_array=args.json_array_regions)
            turn2_regions_context = normalize_regions(turn2_raw, as_json_array=args.json_array_regions)
            image_field = maybe_relpath(img_path, image_folder)

            rec = build_record(
                idx=idx,
                img_path=image_field,
                turn1_target=turn1_target,
                turn2_regions_context=turn2_regions_context,
                transcript=transcript,
                explanation=explanation,
                turn1_system_prompt=args.turn1_system_prompt,
                turn1_user_prompt=args.turn1_user_prompt,
                turn2_system_prompt=args.turn2_system_prompt,
                turn2_user_prompt=args.turn2_user_prompt,
                turn2_user_template=args.turn2_user_template,
            )
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