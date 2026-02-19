#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

DEFAULT_INPUT_CSV = (
    "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_gt.csv"
)
DEFAULT_TRANSCRIPT_JSON_ROOT = (
    "/scratch3/che489/Ha/interspeech/datasets/vocv4_mfa_aligned/"
)
DEFAULT_REGION_TABLE_CSV = (
    "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/region_phone_table_grid.csv"
)
DEFAULT_GRID_EXPLANATION_ROOT = (
    "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/En/grid_explanation/"
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
            if not img_path or not regions:
                continue
            rows.append({"img_path": img_path, "regions": regions})
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


def _load_region_table(csv_path: Path) -> dict[tuple[str, int], dict[str, str]]:
    out: dict[tuple[str, int], dict[str, str]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            region_id_raw = str(row.get("region_id", "")).strip()
            if not sample_id or not region_id_raw:
                continue
            try:
                region_id = int(region_id_raw)
            except ValueError:
                continue
            out[(sample_id, region_id)] = {
                "T": str(row.get("T", "")).strip(),
                "F": str(row.get("F", "")).strip(),
                "P": str(row.get("P_type", "")).strip(),
            }
    return out


def _parse_regions_in_order(regions_raw: str) -> list[int]:
    out: list[int] = []
    for token in regions_raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out


def _build_prompt2_target(
    sample_id: str,
    region_ids: list[int],
    region_lookup: dict[tuple[str, int], dict[str, str]],
    explanation_root: Path,
) -> str | None:
    parts: list[str] = []
    for idx, rid in enumerate(region_ids, start=1):
        info = region_lookup.get((sample_id, rid))
        if info is None:
            return None

        txt_path = explanation_root / sample_id / f"{rid}.txt"
        if not txt_path.exists():
            return None
        en_text = txt_path.read_text(encoding="utf-8")

        part = (
            f"C{idx}={rid},"
            f"T={info['T']},"
            f"F={info['F']},"
            f"P={info['P']},"
            f"En={en_text}"
        )
        parts.append(part)

    return ";".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build stage-1 GT CSV with transcript from an existing img_path/regions CSV."
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
        "--fail-on-missing-transcript",
        action="store_true",
        help="Exit with error if transcript JSON is missing or cannot be parsed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_csv = Path(DEFAULT_INPUT_CSV)
    transcript_json_root = Path(DEFAULT_TRANSCRIPT_JSON_ROOT)
    region_table_csv = Path(DEFAULT_REGION_TABLE_CSV)
    explanation_root = Path(DEFAULT_GRID_EXPLANATION_ROOT)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    input_rows = _load_input_rows(input_csv)
    region_lookup = _load_region_table(region_table_csv)

    rows = []
    dropped_not_three = 0
    dropped_missing_prompt2 = 0
    missing_transcript_count = 0
    skipped_no_sample_id = 0

    for row in input_rows:
        img_path = row["img_path"]
        regions_raw = row["regions"]
        region_list = _parse_regions_in_order(regions_raw)

        if len(region_list) != 3 or len(set(region_list)) != 3:
            dropped_not_three += 1
            continue

        sample_id = _sample_id_from_img_path(img_path, args.image_suffix)
        if not sample_id:
            skipped_no_sample_id += 1
            continue

        prompt2_target = _build_prompt2_target(
            sample_id=sample_id,
            region_ids=region_list,
            region_lookup=region_lookup,
            explanation_root=explanation_root,
        )
        if prompt2_target is None:
            dropped_missing_prompt2 += 1
            continue

        transcript_json_path = transcript_json_root / f"{sample_id}.json"
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
                "prompt2_target": prompt2_target,
            }
        )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["img_path", "regions", "transcript", "prompt2_target"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to: {output_csv}")
    print(f"Dropped {dropped_not_three} rows (regions must be exactly 3 unique IDs)")
    print(f"Dropped {dropped_missing_prompt2} rows (missing T/F/P or explanation txt for at least one region)")
    if skipped_no_sample_id:
        print(f"Skipped {skipped_no_sample_id} rows (could not derive sample_id from img_path)")
    print(f"Rows with missing/invalid transcript: {missing_transcript_count}")


if __name__ == "__main__":
    main()
