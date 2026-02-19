#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build query-2-only SFT JSON by extracting turn-2 user/assistant pair from multi-turn JSON."
    )
    parser.add_argument("--input-json", required=True, help="Path to stage1_multiturn_*.json")
    parser.add_argument("--output-json", required=True, help="Path to query2-only output JSON")
    parser.add_argument(
        "--require-exact-4-turns",
        action="store_true",
        help="If set, drop samples unless conversations length is exactly 4.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    with input_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    out = []
    dropped = 0

    for rec in data:
        conversations = rec.get("conversations", [])
        if args.require_exact_4_turns and len(conversations) != 4:
            dropped += 1
            continue
        if len(conversations) < 4:
            dropped += 1
            continue

        user2 = conversations[2]
        asst2 = conversations[3]
        if user2.get("from") != "human" or asst2.get("from") != "gpt":
            dropped += 1
            continue

        out.append(
            {
                "id": f"{rec.get('id', 'sample')}_q2",
                "image": rec.get("image"),
                "conversations": [
                    {"from": "human", "value": user2.get("value", "")},
                    {"from": "gpt", "value": asst2.get("value", "")},
                ],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Loaded:  {len(data)}")
    print(f"Saved:   {len(out)} -> {output_path}")
    print(f"Dropped: {dropped}")


if __name__ == "__main__":
    main()

