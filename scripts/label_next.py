import json
from pathlib import Path
from typing import Any

from rich.console import Console

from structured_eval.label_assist import generate_draft_label
from structured_eval.schema import JobPostingLabel


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "job_postings.jsonl"
LABELED_PATH = ROOT / "data" / "labeled" / "labeled_jobs.jsonl"
console = Console()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def preview_text(text: str, limit: int = 800) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def prompt_edit_field(name: str, value: Any) -> Any:
    display = json.dumps(value, ensure_ascii=False)
    raw = input(f"{name} [{display}]: ").strip()
    if raw == "":
        return value
    if name in {"required_skills", "nice_to_have_skills"}:
        return [item.strip() for item in raw.split(",") if item.strip()]
    if name in {"salary_min", "salary_max"}:
        return int(raw)
    if name == "required_years_experience":
        return float(raw)
    if name in {"security_clearance_required", "sponsorship_available"}:
        lowered = raw.lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
        return None
    return raw


def run() -> None:
    raw_records = load_jsonl(RAW_PATH)
    labeled_records = load_jsonl(LABELED_PATH)
    labeled_ids = {record.get("id") for record in labeled_records}

    next_record = next((record for record in raw_records if record.get("id") not in labeled_ids), None)
    if next_record is None:
        console.print("[green]No unlabeled raw jobs found.[/green]")
        return

    job_id = next_record["id"]
    text = next_record["text"]
    console.print(f"[bold]Labeling job:[/bold] {job_id}")
    console.print(preview_text(text))

    try:
        console.print("\nGenerating draft label with the LLM...\n")
        draft = generate_draft_label(text)
        console.print(draft.model_dump_json(indent=2))
        accept = input("Accept draft label? [y/N]: ").strip().lower()
    except ValueError as exc:
        if "OPENAI_API_KEY" not in str(exc):
            raise
        console.print(f"[yellow]{exc}.[/yellow]")
        console.print("[yellow]Starting manual labeling instead.[/yellow]\n")
        draft = JobPostingLabel()
        accept = "no"

    if accept not in {"y", "yes"}:
        console.print("Editing fields. Press Enter to keep the current value.\n")
        data = draft.model_dump(mode="json")
        for field in data:
            data[field] = prompt_edit_field(field, data[field])
        draft = JobPostingLabel.model_validate(data)

    LABELED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABELED_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"id": job_id, **draft.model_dump(mode="json")}, ensure_ascii=False) + "\n")

    console.print(f"[green]Saved label for {job_id} to {LABELED_PATH}[/green]")


if __name__ == "__main__":
    run()
