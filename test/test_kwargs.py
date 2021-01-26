# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Placeholder."""
from smepu.argparse import sm_protocol, to_kwargs

import pytest


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
    """Put a placeholder."""
    parser = sm_protocol()
    args, train_args = parser.parse_known_args(test_input)

    hyperopts = to_kwargs(train_args)
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
    """Put a placeholder."""
    parser = sm_protocol()
    args, train_args = parser.parse_known_args(test_input)

    args, train_args = parser.parse_known_args(test_input)
    hyperopts = to_kwargs(train_args)
    result = actual_train(args.model_dir, args.train, **hyperopts)

    assert result == expected


def actual_train(model_dir, train_dir, epochs=100, init="uniform", dict_arg=None):
    """Put a placeholder."""
    return {
        "model_dir": model_dir.as_posix(),
        "train_dir": train_dir.as_posix(),
        "epochs": 7,
        "init": init,
        "dict_arg": dict_arg,
    }
