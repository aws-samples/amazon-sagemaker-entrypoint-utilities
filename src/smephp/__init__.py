import pathlib

import pkg_resources

from . import argparse  # noqa
from .core import is_on_sagemaker, mkdir, pathify, setup_opinionated_logger  # noqa

_pkg_dir: pathlib.Path = pathlib.Path(__file__).resolve().parent

try:
    # Loading installed module, so read the installed version
    __version__ = pkg_resources.require(_pkg_dir.name)[0].version
except pkg_resources.DistributionNotFound:
    # Loading uninstalled module, so try to read version from ../../VERSION.
    with open(_pkg_dir / ".." / ".." / "VERSION", "r") as f:
        __version__ = f.readline().strip()


# To disable tqdm when running on SageMaker, this module MUST be imported before
# any other module that uses tqdm.
#
# https://github.com/tqdm/tqdm/issues/619#issuecomment-425234504
#
# Note that tqdm is not really completely silenced, but it's just that the
# progress bar will be printed much less frequently.

if is_on_sagemaker():
    import tqdm
    from tqdm import auto

    # Make sure this is sent to CloudWatch, regardless of how logger is setup.
    print("Env. var SM_HOSTS detected. Silencing tqdm as we're likely to run on SageMaker...")

    # Original tqdm callables
    old_auto = auto.tqdm
    old_tqdm = tqdm.tqdm
    old_trange = tqdm.trange

    def no_auto(*a, **k):
        k["disable"] = True
        return old_auto(*a, **k)

    def nop_tqdm(*a, **k):
        k["disable"] = True
        return old_tqdm(*a, **k)

    def no_trange(*a, **k):
        k["disable"] = True
        return old_trange(*a, **k)

    auto.tqdm = no_auto
    tqdm.tqdm = nop_tqdm
    tqdm.trange = no_trange
