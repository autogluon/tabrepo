from __future__ import annotations

import pytest
from tabarena.benchmark.experiment.experiment_runner_api import (
    _parse_repetitions_mode_and_args,
)

RES_2_2 = [(f, r) for f in range(2) for r in range(2)]


@pytest.mark.parametrize(
    ("repetitions_mode", "repetitions_mode_args", "tasks", "expected_output"),
    [
        # Presets
        ("TabArena-Lite", None, [0, 1], [[(0, 0)], [(0, 0)]]),
        ("TabArena-Lite", (4, 3), [0, 1], [[(0, 0)], [(0, 0)]]),
        # Matrix
        ("matrix", None, [0, 1], AssertionError),  # input None
        ("matrix", [[None], [None]], [0], AssertionError),  # not same size as tasks
        ("matrix", [(), [None]], [0], AssertionError),  # not tuple (in a later element)
        ("matrix", [(1,)], [0], AssertionError),  # not len 2
        ("matrix", [(1, "str")], [0], AssertionError),  # not both str
        ("matrix", [([], "str")], [0], AssertionError),  # if list-based, not both str
        ("matrix", [([], [0])], [0], AssertionError),  # empty list
        ("matrix", [([0], ["str"])], [0], AssertionError),  # not int list
        ("matrix", "str", [0], AssertionError),  # not a tuple if not a list
        ("matrix", ([0], []), [0], AssertionError),  # empty list without list of tuples
        ("matrix", (2, 2), [0, 1], [RES_2_2, RES_2_2]),
        ("matrix", ([0, 1], [0, 1]), [0, 1], [RES_2_2, RES_2_2]),
        ("matrix", ([2], [0, 1]), [0, 1], [[(2, 0), (2, 1)], [(2, 0), (2, 1)]]),
        # Individual
        ("individual", None, [0, 1], AssertionError),  # input None
        ("individual", "str", [0], AssertionError),  # not a list
        ("individual", [[None], [None]], [0], AssertionError),  # not same size as tasks
        ("individual", [(0, 0), (0, "str")], [0], AssertionError),  # not tuples of int
        ("individual", [(0, 0), (0, 1, "str")], [0], AssertionError),  # wrong length
        ("individual", [(0, 0), None], [0], AssertionError),  # not tuples
        ("individual", [[(0, 1)], None], [0], AssertionError),  # not list of tuples
        ("individual", [[(0, 1)], [None]], [0], AssertionError),  # not list of tuples
        (
            "individual",
            [[(0, 1)], [(None,)]],
            [0],
            AssertionError,
        ),  # not list of tuples
        (
            "individual",
            [[(0, 1)], [(None, "str")]],
            [0],
            AssertionError,
        ),  # not list of tuples
        ("individual", [(0, 0)], [0, 1], [[(0, 0)], [(0, 0)]]),
        ("individual", [(0, 0), (2, 3)], [0, 1], [[(0, 0), (2, 3)], [(0, 0), (2, 3)]]),
        (
            "individual",
            [[(0, 0), (0, 1)], [(1, 0), (1, 1)]],
            [0, 1],
            [[(0, 0), (0, 1)], [(1, 0), (1, 1)]],
        ),
    ],
)
def test_parse_repetitions_mode_and_args(
    repetitions_mode, repetitions_mode_args, tasks, expected_output
):
    if isinstance(expected_output, type) and issubclass(expected_output, BaseException):
        with pytest.raises(expected_output):
            _parse_repetitions_mode_and_args(
                repetitions_mode=repetitions_mode,
                repetitions_mode_args=repetitions_mode_args,
                tasks=tasks,
            )
    else:
        assert expected_output == _parse_repetitions_mode_and_args(
            repetitions_mode=repetitions_mode,
            repetitions_mode_args=repetitions_mode_args,
            tasks=tasks,
        )
