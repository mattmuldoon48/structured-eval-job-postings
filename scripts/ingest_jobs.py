import json
from pathlib import Path

from rich.console import Console


RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "job_postings.jsonl"
END_MARKER = "###END###"
console = Console()


def load_max_id() -> int:
    if not RAW_PATH.exists():
        return 0
    with RAW_PATH.open("r", encoding="utf-8") as stream:
        lines = [line.strip() for line in stream if line.strip()]
        if not lines:
            return 0
        last = json.loads(lines[-1])
        job_id = last.get("id", "job-000")
        try:
            return int(job_id.split("-")[1])
        except (IndexError, ValueError):
            return len(lines)


def prompt_job_text() -> str:
    console.print(f"\n[bold]Paste job posting text[/bold] (type {END_MARKER} on its own line when done):")
    lines = []
    while True:
        line = input()
        if line.strip() == END_MARKER:
            if lines:
                break
            console.print("[yellow]Please enter some text before ending the posting.[/yellow]")
            continue
        lines.append(line)
    return "\n".join(lines)


def run() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    while True:
        max_id = load_max_id()
        next_id = f"job-{max_id + 1:03d}"

        console.print(f"[green]Next ID:[/green] {next_id}")
        text = prompt_job_text()

        record = {"id": next_id, "text": text}
        with RAW_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        console.print(f"[green]✓ Added {next_id}[/green]")

        another = input("Add another? [y/N]: ").strip().lower()
        if another not in {"y", "yes"}:
            console.print(f"[green]Done. {RAW_PATH.name} updated.[/green]")
            break


if __name__ == "__main__":
    run()
