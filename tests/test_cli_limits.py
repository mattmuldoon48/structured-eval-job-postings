import sys

import pytest

from scripts import label_batch, review_drafts, run_benchmark, run_eval


@pytest.mark.parametrize(
    ("parse_args", "option"),
    [
        (run_eval.parse_args, "--limit"),
        (run_benchmark.parse_args, "--limit-per-split"),
        (label_batch.parse_args, "--limit"),
        (review_drafts.parse_args, "--limit"),
    ],
    ids=["eval", "benchmark", "batch-labeling", "draft-review"],
)
@pytest.mark.parametrize("limit", ["0", "-1"])
def test_nonpositive_workload_limits_are_usage_errors(
    monkeypatch, capsys, parse_args, option, limit
):
    monkeypatch.setattr(sys, "argv", ["command", option, limit])

    with pytest.raises(SystemExit) as error:
        parse_args()

    assert error.value.code == 2
    assert option in capsys.readouterr().err
