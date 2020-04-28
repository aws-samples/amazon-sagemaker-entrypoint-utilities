import json
import sys
from typing import Any, Dict, List


# TODO: support hyperparameters that translates to an object instance.
# - CLI arg convention: "... --trainer.__class__ gluonts.trainer.Trainer --trainer.epochs 100 --trainer.1 abcd",
#   which translates to fun(..., trainer=gluonts.trainer.Trainer("abcd", epochs=100), ...).
# - Deser logic: use gluonts deser (which also defines the intermediate dict representation).
def parse_for_func(hm: List[str]) -> Dict[str, Any]:
    """Convert list of ['--name', 'value', ...] to {'name': val}, where 'val' will be in the nearest data type.

    Conversion follows the principle: "if it looks like a duck and quacks like a duck, then it must be a duck".
    """
    d = {}
    it = iter(hm)
    try:
        while True:
            key = next(it)[2:]
            value = next(it)
            d[key] = value
    except StopIteration:
        pass

    # Infer data types.
    dd = {k: infer_dtype(v) for k, v in d.items()}
    return dd


def parse_for_argv(hm: List[str]) -> List[Any]:
    # TODO: This function converts a SageMaker-compatible CLI args to structure that the underlying function expect.
    # Intended use-case: when wrapping another function that directly access sys.argv.
    # Returns a new sys.argv-like data structure, i.e., ['--param1', 'value1', '--].
    raise NotImplementedError


def patch_sys_argv(hm: List[str]) -> List[Any]:
    # TODO: replace sys.argv with whatever returned by parse_hp_for_argv. Return the original sys.argv.
    ori_sys_argv = sys.argv
    sys.argv = [sys.argv[0], *parse_for_argv(sys.argv)]
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
