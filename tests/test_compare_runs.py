import math

import pytest

from scripts.compare_runs import format_value


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "-"),
        (0.0, "0.000"),
        (-0.0, "-0.000"),
        (1.23456, "1.235"),
        (-1.23456, "-1.235"),
        (42, "42"),
        (True, "True"),
        ("score", "score"),
        (["a", "b"], "['a', 'b']"),
    ],
)
def test_format_value_handles_missing_numeric_boundaries_and_other_types(value, expected):
    assert format_value(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(math.inf, "inf"), (-math.inf, "-inf"), (math.nan, "nan")],
)
def test_format_value_preserves_float_special_values(value, expected):
    assert format_value(value) == expected
