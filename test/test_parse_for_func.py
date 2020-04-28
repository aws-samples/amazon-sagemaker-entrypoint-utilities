import argparse
import os
from pathlib import Path

import pytest

from smephp.argparse import parse_for_func


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            ["--epochs", "7", "--init", "xavier", "--dict_arg", '{"seq": [1,2,3]}'],
            dict(model_dir="model", train_dir="data/train", epochs=7, init="xavier", dict_arg={"seq": [1, 2, 3]}),
        )
    ],
)
def test_good(test_input, expected):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("SM_MODEL_DIR", "model"))
    parser.add_argument("--train", type=Path, default=os.environ.get("SM_CHANNEL_TRAIN", "data/train"))

    args, train_args = parser.parse_known_args(test_input)
    hyperopts = parse_for_func(train_args)
    result = actual_train(args.model_dir, args.train, **hyperopts)
    print("Result:", result)
    print("Expected:", expected)
    assert result == expected


@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            ["--epochs", "7", "--init", "xavier", "--dict_arg", '{"seq": [7,8,9]}'],
            dict(model_dir="model", train_dir="data/train", epochs=7, init="xavier", dict_arg={"seq": [1, 2, 3]}),
        )
    ],
)
def test_bad(test_input, expected):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("SM_MODEL_DIR", "model"))
    parser.add_argument("--train", type=Path, default=os.environ.get("SM_CHANNEL_TRAIN", "data/train"))

    args, train_args = parser.parse_known_args(test_input)
    hyperopts = parse_for_func(train_args)
    result = actual_train(args.model_dir, args.train, **hyperopts)
    print("Result:", result)
    print("Not Expected:", expected)
    assert result == expected


def actual_train(model_dir, train_dir, epochs=100, init="uniform", dict_arg=None):
    return {
        "model_dir": model_dir.as_posix(),
        "train_dir": train_dir.as_posix(),
        "epochs": 7,
        "init": init,
        "dict_arg": dict_arg,
    }
