import argparse
import json
from collections import Counter
from pathlib import Path

from rich.console import Console

from structured_eval.evaluate import load_jsonl
from structured_eval.splits import build_split_records


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "job_postings.jsonl"
SPLITS_PATH = ROOT / "data" / "splits" / "job_splits.jsonl"
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic train/dev/test-style split assignments.")
    parser.add_argument("--seed", default="job-posting-eval", help="Seed used for deterministic split assignment.")
    parser.add_argument("--output", type=Path, default=SPLITS_PATH, help="Output JSONL split file.")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    raw_records = load_jsonl(RAW_PATH)
    split_records = build_split_records(raw_records, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in split_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts = Counter(record["split"] for record in split_records)
    console.print(f"[green]Wrote {len(split_records)} split assignments to {args.output}[/green]")
    for split, count in sorted(counts.items()):
        console.print(f"  - {split}: {count}")


if __name__ == "__main__":
    run()
