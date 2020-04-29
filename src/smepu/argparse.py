import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ._gluonts_core_serde import decode


def _list(*args):
    """Wrapper to Python list, to be used by cli args."""
    return list(args)


def _set(*args):
    """Wrapper to Python set, to be used by cli args."""
    return set(args)


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


def parse_for_func(cli_args: List[str]) -> Dict[str, Any]:
    """Convert list of ['--name', 'value', ...] to {'name': val}, where 'val' will be in the nearest data type
    or a specific class as specified in the cli args.

    For the nearest types, conversion follows the principle: "if it looks like a duck and quacks like a duck,
    then it must be a duck".
    """
    # TODO: with eval() and/or exec(), the cli args can be made shorter. Is this a good idea?
    args_round_1 = _round_1(cli_args)
    args_round_2 = _round_2(args_round_1)  # Custom class in IR
    return args_round_2


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


################################################################################
# Helper utilities
################################################################################
ArgsDict = Dict[str, Any]
IR = Dict[str, "ObjectIR"]


def _round_1(cli_args: List[str]) -> ArgsDict:
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


def _round_2(d: ArgsDict) -> ArgsDict:
    """Lower CLI args to intermediate representations.

    This function aims to support cli hyperparameters that translates to an object instance, e.g.,
    "--trainer.__class__ gluonts.trainer.Trainer --trainer.epochs 100 --trainer.0 abcd" is intended to ultimately
    become some_function(trainer=gluonts.trainer.Trainer("abcd", epochs=100)).

    Args:
        d (ArgsDict): Arguments produced by round-1 parsing.

    Returns:
        ArgsDict: lowered arguments.
    """

    def split_args(d: ArgsDict) -> Tuple[IR, ArgsDict]:
        """Split input arguments into two parts: as-is, and to-be-lowered.

        Args:
            d (ArgsDict): Arguments produced by round-1 parsing.

        Returns:
            Tuple[ArgsDict, ArgsDict]: Tuple of (as-is arguments, to-be-lowered arguments).
        """
        untouched, workset = {}, {}
        for k, v in d.items():
            if "." not in k:
                untouched[k] = v
            else:
                workset[k] = v
        return untouched, workset

    def lower(d) -> IR:
        """Convert cli args dict to ObjectIR."""
        klasses, rest = ObjectIR.split(d)
        ObjectIR.scatter_args(klasses, rest)
        # [print(f"{k}:", v.klass_dict) for k, v in klasses.items()]  # type: ignore
        top_level_klasses = ObjectIR.reduce_oir(klasses)
        return top_level_klasses

    untouched, workset = split_args(d)
    ir = lower(workset)
    desered = {k: decode(v.klass_dict) for k, v in ir.items()}
    # [print(f"{k}:", v.klass_dict) for k, v in ir.items()]  # type: ignore
    return {**untouched, **desered}


class ObjectIR(object):
    @staticmethod
    def split(d: ArgsDict) -> Tuple[ArgsDict, ArgsDict]:
        klasses, rest = {}, {}
        for k, v in d.items():
            if k.endswith("__class__"):
                klasses[k[: -len(".__class__")]] = ObjectIR(v)
            else:
                # I'm args (either pos or kw).
                rest[k] = v
        return klasses, rest

    @staticmethod
    def scatter_args(klasses: IR, rest: ArgsDict):
        """Assign arguments to their ObjectIR; will modify ``klasses`` in-place."""
        for k, v in rest.items():
            k2, arg_spec = k.rsplit(".", 1)
            object_ir = klasses[k2]  # Get the owner ObjectIR.
            if ObjectIR.is_kwarg(arg_spec):
                object_ir.kwargs[arg_spec] = v
            else:
                object_ir.args_d[arg_spec] = v

    @staticmethod
    def is_kwarg(s: str) -> bool:
        # Is this a positional argument or keyword argument?
        try:
            # Python doesn't optimize away (i.e., remove) next statement,
            # unlike some compilers; or else, fun puzzling time ahead.
            int(s)
            return False
        except ValueError:
            return True

    @staticmethod
    def reduce_oir(klasses: IR) -> IR:
        """Reduced ObjectIR to just the top-level ones."""
        # After lower, we end up with this:
        #
        # callbacks: {'__kind__': 'instance', 'class': 'list', 'args': [], 'kwargs': {}}
        # callbacks.0: {'__kind__': 'instance', 'class': 'dummyest.DummyCallback', 'args': [], 'kwargs': {'name': 'EarlyStopper'}}
        # callbacks.1: {'__kind__': 'instance', 'class': 'dummyest.DummyCallback', 'args': ['Checkpointer'], 'kwargs': {}}
        #
        # So, another step needed to properly assign each ObjectIR as pos- or kw-args of another ObjectIR.
        for k, v in klasses.items():
            if "." not in k:
                # Top-level has no parent.
                continue

            k2, arg_spec = k.rsplit(".", 1)
            object_ir = klasses[k2]  # Get the parent ObjectIR
            if ObjectIR.is_kwarg(arg_spec):
                object_ir.kwargs[arg_spec] = v
            else:
                object_ir.args_d[arg_spec] = v

        return {k: v for k, v in klasses.items() if "." not in k}

    def __init__(self, klass: str) -> None:
        self.klass = klass
        self.args_d: ArgsDict = {}  # Use dict as #args unknown & may be out-of-order.
        self.kwargs: ArgsDict = {}

    @property
    def args(self) -> List[Any]:
        """Convert args dictionary to list."""
        args = [None] * len(self.args_d)
        for k, v in self.args_d.items():
            args[int(k)] = v
        return args

    @property
    def klass_dict(self) -> ArgsDict:
        """Return this IR object as a gluonts-style dictionary."""
        args_ird = [a.klass_dict if isinstance(a, ObjectIR) else a for a in self.args]
        kwargs_ird = {k: v.klass_dict if isinstance(v, ObjectIR) else v for k, v in self.kwargs.items()}
        return {"__kind__": "instance", "class": self.klass, "args": args_ird, "kwargs": kwargs_ird}
