from scripts.run_benchmark import build_eval_command


def test_build_eval_command_includes_matching_split_options():
    command = build_eval_command(
        python_executable="python",
        split="dev",
        prompt="extract_v3.txt",
        limit_per_split=2,
        min_overall=0.85,
        min_metrics=["exact_accuracy.remote_policy=0.80"],
        input_cost_per_1m=0.4,
        output_cost_per_1m=1.6,
    )

    assert command[:5] == ["python", command[1], "--split", "dev", "--prompt"]
    assert command[1].endswith("scripts/run_eval.py")
    assert command[5:] == [
        "extract_v3.txt",
        "--limit",
        "2",
        "--min-overall",
        "0.85",
        "--min-metric",
        "exact_accuracy.remote_policy=0.80",
        "--input-cost-per-1m",
        "0.4",
        "--output-cost-per-1m",
        "1.6",
    ]
