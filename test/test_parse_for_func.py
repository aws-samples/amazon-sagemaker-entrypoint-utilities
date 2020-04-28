import argparse
import os
from pathlib import Path

import pytest

from smepu.argparse import parse_for_func, sm_protocol


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            ["--epochs", "7", "--init", "xavier", "--dict_arg", '{"seq": [1, 2]}'],
            dict(model_dir="model", train_dir="data/train", epochs=7, init="xavier", dict_arg={"seq": [1, 2]}),
        )
    ],
)
def test_good(test_input, expected):
    parser = sm_protocol()
    args, train_args = parser.parse_known_args(test_input)

    hyperopts = parse_for_func(train_args)
    result = actual_train(args.model_dir, args.train, **hyperopts)

    assert result == expected


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            ["--epochs", "7", "--init", "xavier", "--dict_arg", '{"seq": [7,8]}'],
            dict(model_dir="model", train_dir="data/train", epochs=7, init="xavier", dict_arg={"seq": [1, 2]}),
        )
    ],
)
def test_bad(test_input, expected):
    parser = sm_protocol()
    args, train_args = parser.parse_known_args(test_input)

    args, train_args = parser.parse_known_args(test_input)
    hyperopts = parse_for_func(train_args)
    result = actual_train(args.model_dir, args.train, **hyperopts)

    assert result == expected


def actual_train(model_dir, train_dir, epochs=100, init="uniform", dict_arg=None):
    return {
        "model_dir": model_dir.as_posix(),
        "train_dir": train_dir.as_posix(),
        "epochs": 7,
        "init": init,
        "dict_arg": dict_arg,
    }
