import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List


def sm_protocol(
    model: str = "model",
    output: str = "output",
    channels: List[str] = ["train", "test", "validation"],
    mkdir_local: bool = True,
) -> argparse.ArgumentParser:
    """Create an arg parser that implements minimum SageMaker entrypoint protocol.

    Only model, output, and channel dirs are implemented, as this is typically bare minimum to run or test an
    entrypoint script locally, e.g., `python ./entrypoint.py`.

    See https://github.com/aws/sagemaker-containers/blob/master/README.rst#important-environment-variables.

    Args:
        model (str, optional): Model dir when not running on SageMaker. Defaults to "model".
        output (str, optional): Output dir when not running on SageMaker. Defaults to "output".
        channels (List[str], optional): Data channels. Defaults to ["train", "test", "validation"].

    Returns:
        argparse.ArgumentParser: argument parser with minimum SageMaker protocol.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("SM_MODEL_DIR", "model"))
    parser.add_argument("--output-data-dir", type=Path, default=os.environ.get("SM_OUTPUT_DATA_DIR", "output"))
    for channel in channels:
        parser.add_argument(
            f"--{channel}",
            type=Path,
            default=os.environ.get(f"SM_CHANNEL_{channel.upper()}", os.path.join("data", channel)),
        )

    return parser


# TODO: support hyperparameters that translates to an object instance.
# - CLI arg convention: "... --trainer.__class__ gluonts.trainer.Trainer --trainer.epochs 100 --trainer.1 abcd",
#   which translates to fun(..., trainer=gluonts.trainer.Trainer("abcd", epochs=100), ...).
# - Deser logic: use gluonts deser (which also defines the intermediate dict representation).
def parse_for_func(cli_args: List[str]) -> Dict[str, Any]:
    """Convert list of ['--name', 'value', ...] to {'name': val}, where 'val' will be in the nearest data type.

    Conversion follows the principle: "if it looks like a duck and quacks like a duck, then it must be a duck".
    """
    d = {}
    it = iter(cli_args)
    try:
        # Each iteration swallows ["--kwarg", "value"]
        expected = 0
        while True:
            # Get --key
            key = next(it)[2:]
            expected += 1

            # Get the value. Warn if it looks fishy.
            value = next(it)
            expected -= 1
            if value[:2] == "--":
                warnings.warn(f'Fishy cli args / hyperparams: {key}="{value}"')
            d[key] = value
    except StopIteration:
        if expected > 1:
            raise ValueError(f"CLI arg --{key} has no value, so ignored")

    # Infer data types.
    dd = {k: infer_dtype(v) for k, v in d.items()}
    return dd


def parse_for_argv(cli_args: List[str]) -> List[Any]:
    # TODO: This function converts a SageMaker-compatible CLI args to structure that the underlying function expect.
    # Intended use-case: when wrapping another function that directly access sys.argv.
    # Returns a new sys.argv-like data structure, i.e., ['--param1', 'value1', '--].
    raise NotImplementedError


def patch_sys_argv(cli_args: List[str]) -> List[Any]:
    # TODO: replace sys.argv with whatever returned by parse_hp_for_argv. Return the original sys.argv.
    ori_sys_argv = sys.argv
    sys.argv = [sys.argv[0], *parse_for_argv(cli_args)]
    return ori_sys_argv


def infer_dtype(s):
    """Auto-cast string values to nearest matching datatype.

    Conversion follows the principle: "if it looks like a duck and quacks like a duck, then it must be a duck".
    Note that python 3.6 implements PEP-515 which allows '_' as thousand separators. Hence, on Python 3.6,
    '1_000' is a valid number and will be converted accordingly.
    """
    if s == "None":
        return None
    if s == "True":
        return True
    if s == "False":
        return False

    try:
        i = float(s)
        if ("." in s) or ("e" in s.lower()):
            return i
        else:
            return int(s)
    except:
        pass

    try:
        # If string is json, deser it.
        return json.loads(s)
    except:
        return s
