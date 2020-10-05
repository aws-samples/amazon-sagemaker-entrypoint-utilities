# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Placeholder."""
import os

from . import argparse  # noqa
from ._version import get_versions
from .argparse import _list as list  # noqa
from .argparse import _set as set  # noqa
from .core import is_on_sagemaker, mkdir, pathify, setup_opinionated_logger  # noqa

__version__ = get_versions()["version"]
del get_versions

# To disable tqdm when running on SageMaker, this module MUST be imported before
# any other module that uses tqdm.
#
# https://github.com/tqdm/tqdm/issues/619#issuecomment-425234504
#
# Note that tqdm is not really completely silenced, but it's just that the
# progress bar will be printed much less frequently.

if is_on_sagemaker():
    # print() to ensure messages reach CloudWatch regardless of logger setup.
    print("SM_HOSTS: silencing tqdm.")
    try:
        import tqdm
        from tqdm import auto
    except ImportError:
        print("SM_HOSTS: could not import tqdm, so do nothing.")
    else:
        # Original tqdm callables
        old_auto = auto.tqdm
        old_tqdm = tqdm.tqdm
        old_trange = tqdm.trange

        def no_auto(*a, **k):
            """Put a place holder."""
            k["disable"] = True
            return old_auto(*a, **k)

        def nop_tqdm(*a, **k):
            """Put a place holder."""
            k["disable"] = True
            return old_tqdm(*a, **k)

        def no_trange(*a, **k):
            """Put a place holder."""
            k["disable"] = True
            return old_trange(*a, **k)

        auto.tqdm = no_auto
        tqdm.tqdm = nop_tqdm
        tqdm.trange = no_trange

        print("SM_HOSTS: tqdm silenced.")

    # Make spacy.convert & spacy.train output plain log.
    print("SM_HOSTS: make plain wasabi.")
    os.environ["ANSI_COLORS_DISABLE"] = "1"
    os.environ["WASABI_NO_PRETTY"] = "1"
    os.environ["WASABI_LOG_FRIENDLY"] = "1"
    # Additional setting for spacy.train to make its log plain.
    print("SM_HOSTS: make plain spacy train.")
    os.environ["LOG_FRIENDLY"] = "1"
